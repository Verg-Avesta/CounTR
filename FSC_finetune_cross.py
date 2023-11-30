import argparse
import datetime
import json
import numpy as np
import os
import time
import random
from pathlib import Path
import sys
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
import torchvision
import wandb
import timm
from tqdm import tqdm

assert "0.4.5" <= timm.__version__ <= "0.4.9"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import util.lr_sched as lr_sched
from util.FSC147 import transform_train, transform_val
import models_mae_cross


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=26, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--mask_ratio', default=0.5, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='./data/FSC147/', type=str,
                        help='dataset path')
    parser.add_argument('--anno_file', default='annotation_FSC147_384.json', type=str,
                     help='annotation json file')
    parser.add_argument('--data_split_file', default='Train_Test_Val_FSC_147.json', type=str,
                        help='data split json file')
    parser.add_argument('--class_file', default='ImageClasses_FSC147.txt', type=str,
                        help='class json file')
    parser.add_argument('--im_dir', default='images_384_VarV2', type=str,
                        help='images directory')
    parser.add_argument('--output_dir', default='./data/out/fim6_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='./data/out/pre_4_dir/checkpoint-300.pth',
                        help='resume from checkpoint')
    parser.add_argument('--do_resume', action='store_true',
                        help='Resume training (e.g. if crashed).')

    # Training parameters
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--do_aug', action='store_true',
                        help='Perform data augmentation.')
    parser.add_argument('--no_do_aug', action='store_false', dest='do_aug')
    parser.set_defaults(do_aug=True)

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # Logging parameters
    parser.add_argument("--title", default="CounTR_finetuning", type=str)
    parser.add_argument("--wandb", default="counting", type=str)
    parser.add_argument("--team", default="wsense", type=str)
    parser.add_argument("--wandb_id", default=None, type=str)

    return parser


os.environ["CUDA_LAUNCH_BLOCKING"] = '1'


class TrainData(Dataset):
    def __init__(self, args, split='train', do_aug=True):
        with open(args.anno_file) as f:
            annotations = json.load(f)
        with open(args.data_split_file) as f:
            data_split = json.load(f)

        self.img = data_split[split]
        random.shuffle(self.img)
        self.split = split
        self.img_dir = im_dir
        self.TransformTrain = transform_train(args, do_aug=do_aug)
        self.TransformVal = transform_val(args)
        self.annotations = annotations
        self.im_dir = im_dir

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        im_id = self.img[idx]
        anno = self.annotations[im_id]
        bboxes = anno['box_examples_coordinates']
        dots = np.array(anno['points'])

        rects = list()
        for bbox in bboxes:
            x1 = bbox[0][0]
            y1 = bbox[0][1]
            x2 = bbox[2][0]
            y2 = bbox[2][1]
            rects.append([y1, x1, y2, x2])

        image = Image.open('{}/{}'.format(self.im_dir, im_id))
        if image.mode == "RGBA":
            image = image.convert("RGB")
        image.load()
        m_flag = 0

        sample = {'image': image, 'lines_boxes': rects, 'dots': dots, 'id': im_id, 'm_flag': m_flag}
        sample = self.TransformTrain(sample) if self.split == "train" else self.TransformVal(sample)
        return sample['image'], sample['gt_density'], len(dots), sample['boxes'], sample['pos'], sample['m_flag'], im_id


def main(args):
    wandb_run = None
    try:
        misc.init_distributed_mode(args)

        print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
        print("{}".format(args).replace(', ', ',\n'))

        device = torch.device(args.device)

        # fix the seed for reproducibility
        seed = args.seed + misc.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        cudnn.benchmark = True

        dataset_train = TrainData(args, do_aug=args.do_aug)
        dataset_val = TrainData(args, split='val')

        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

        if global_rank == 0:
            if args.wandb is not None:
                wandb_run = wandb.init(
                    config=args,
                    resume="allow",
                    project=args.wandb,
                    name=args.title,
                    entity=args.team,
                    tags=["CounTR", "finetuning"],
                    id=args.wandb_id,
                )

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )

        # define the model
        model = models_mae_cross.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
        model.to(device)
        model_without_ddp = model
        # print("Model = %s" % str(model_without_ddp))

        eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

        if args.lr is None:  # only base_lr is specified
            args.lr = args.blr * eff_batch_size / 256

        print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
        print("actual lr: %.2e" % args.lr)

        print("accumulate grad iterations: %d" % args.accum_iter)
        print("effective batch size: %d" % eff_batch_size)

        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module

        # following timm: set wd as 0 for bias and norm layers
        param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
        print(optimizer)

        loss_scaler = NativeScaler()

        min_MAE = 99999
        print_freq = 50
        save_freq = 50

        misc.load_model_FSC_full(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

        print(f"Start training for {args.epochs - args.start_epoch} epochs   -   rank {global_rank}")
        start_time = time.time()
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)

            # train one epoch
            model.train(True)
            accum_iter = args.accum_iter

            # some parameters in training
            train_mae = torch.tensor([0], dtype=torch.float64, device=device)
            train_mse = torch.tensor([0], dtype=torch.float64, device=device)
            val_mae = torch.tensor([0], dtype=torch.float64, device=device)
            val_mse = torch.tensor([0], dtype=torch.float64, device=device)
            val_nae = torch.tensor([0], dtype=torch.float64, device=device)

            optimizer.zero_grad()

            for data_iter_step, (samples, gt_density, _, boxes, pos, m_flag, im_names) in enumerate(
                    tqdm(data_loader_train, total=len(data_loader_train),
                         desc=f"Train [e. {epoch} - r. {global_rank}]")):
                idx = data_iter_step + (epoch*len(data_loader_train))

                if data_iter_step % accum_iter == 0:
                    lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader_train) + epoch, args)

                samples = samples.to(device, non_blocking=True, dtype=torch.half)
                gt_density = gt_density.to(device, non_blocking=True, dtype=torch.half)
                boxes = boxes.to(device, non_blocking=True, dtype=torch.half)

                # If there is at least one image in the batch using Type 2 Mosaic, 0-shot is banned.
                flag = 0
                for i in range(m_flag.shape[0]):
                    flag += m_flag[i].item()
                if flag == 0:
                    shot_num = random.randint(0, 3)
                else:
                    shot_num = random.randint(1, 3)

                with torch.cuda.amp.autocast():
                    output = model(samples, boxes, shot_num)

                # Compute loss function
                mask = np.random.binomial(n=1, p=0.8, size=[384, 384])
                masks = np.tile(mask, (output.shape[0], 1))
                masks = masks.reshape(output.shape[0], 384, 384)
                masks = torch.from_numpy(masks).to(device)
                loss = (output - gt_density) ** 2
                loss = (loss * masks / (384 * 384)).sum() / output.shape[0]

                # Update information of MAE and RMSE
                with torch.no_grad():
                    pred_cnt = (output.view(len(samples), -1)).sum(1) / 60
                    gt_cnt = (gt_density.view(len(samples), -1)).sum(1) / 60
                    cnt_err = torch.abs(pred_cnt - gt_cnt).float()
                    batch_mae = cnt_err.double().mean()
                    batch_mse = (cnt_err ** 2).double().mean()

                train_mae += batch_mae
                train_mse += batch_mse

                if not torch.isfinite(loss):
                    print("Loss is {}, stopping training".format(loss))
                    sys.exit(1)

                loss /= accum_iter
                loss_scaler(loss, optimizer, parameters=model.parameters(),
                            update_grad=(data_iter_step + 1) % accum_iter == 0)
                if (data_iter_step + 1) % accum_iter == 0:
                    optimizer.zero_grad()

                lr = optimizer.param_groups[0]["lr"]
                loss_value_reduce = misc.all_reduce_mean(loss)
                if (data_iter_step + 1) % (print_freq * accum_iter) == 0 and (data_iter_step + 1) != len(data_loader_train) and data_iter_step != 0:
                    if wandb_run is not None:
                        log = {"train/loss": loss_value_reduce,
                               "train/lr": lr,
                               "train/MAE": batch_mae,
                               "train/RMSE": batch_mse ** 0.5}
                        wandb.log(log, step=idx)

            # evaluation on Validation split
            for val_samples, val_gt_density, val_n_ppl, val_boxes, val_pos, _, val_im_names in \
                tqdm(data_loader_val, total=len(data_loader_val),
                     desc=f"Val [e. {epoch} - r. {global_rank}]"):

                val_samples = val_samples.to(device, non_blocking=True, dtype=torch.half)
                val_gt_density = val_gt_density.to(device, non_blocking=True, dtype=torch.half)
                val_boxes = val_boxes.to(device, non_blocking=True, dtype=torch.half)
                val_n_ppl = val_n_ppl.to(device, non_blocking=True)
                shot_num = random.randint(0, 3)

                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        val_output = model(val_samples, val_boxes, shot_num)

                    val_pred_cnt = (val_output.view(len(val_samples), -1)).sum(1) / 60
                    val_gt_cnt = (val_gt_density.view(len(val_samples), -1)).sum(1) / 60
                    val_cnt_err = torch.abs(val_pred_cnt - val_gt_cnt).float()
                    val_mae += val_cnt_err.double().mean()
                    val_mse += (val_cnt_err ** 2).double().mean()
                    _val_nae = val_cnt_err / val_gt_cnt
                    _val_nae[_val_nae == float('inf')] = 0
                    val_nae += _val_nae.double().mean()

            # Output visualisation information to W&B
            if wandb_run is not None:
                train_wandb_densities = []
                train_wandb_bboxes = []
                val_wandb_densities = []
                val_wandb_bboxes = []
                black = torch.zeros([384, 384], device=device)

                for i in range(output.shape[0]):
                    # gt and predicted density
                    w_d_map = torch.stack([output[i], black, black])
                    gt_map = torch.stack([gt_density[i], black, black])
                    box_map = misc.get_box_map(samples[i], pos[i], device)
                    w_gt_density = samples[i] / 2 + gt_map + box_map
                    w_d_map_overlay = samples[i] / 2 + w_d_map
                    w_densities = torch.cat([w_gt_density, w_d_map, w_d_map_overlay], dim=2)
                    w_densities = torch.clamp(w_densities, 0, 1)
                    train_wandb_densities += [wandb.Image(torchvision.transforms.ToPILImage()(w_densities),
                                                          caption=f"[E#{epoch}] {im_names[i]} ({torch.sum(gt_density[i]).item()}, {torch.sum(output[i]).item()})")]

                    # exemplars
                    w_boxes = torch.cat([boxes[i][x, :, :, :] for x in range(boxes[i].shape[0])], 2)
                    train_wandb_bboxes += [wandb.Image(torchvision.transforms.ToPILImage()(w_boxes),
                                                       caption=f"[E#{epoch}] {im_names[i]}")]

                for i in range(val_output.shape[0]):
                    # gt and predicted density
                    w_d_map = torch.stack([val_output[i], black, black])
                    gt_map = torch.stack([val_gt_density[i], black, black])
                    box_map = misc.get_box_map(val_samples[i], val_pos[i], device)
                    w_gt_density = val_samples[i] / 2 + gt_map + box_map
                    w_d_map_overlay = val_samples[i] / 2 + w_d_map
                    w_densities = torch.cat([w_gt_density, w_d_map, w_d_map_overlay], dim=2)
                    w_densities = torch.clamp(w_densities, 0, 1)
                    val_wandb_densities += [wandb.Image(torchvision.transforms.ToPILImage()(w_densities),
                                                        caption=f"[E#{epoch}] {val_im_names[i]} ({torch.sum(val_gt_density[i]).item()}, {torch.sum(val_output[i]).item()})")]

                    # exemplars
                    w_boxes = torch.cat([val_boxes[i][x, :, :, :] for x in range(val_boxes[i].shape[0])], 2)
                    val_wandb_bboxes += [wandb.Image(torchvision.transforms.ToPILImage()(w_boxes),
                                                     caption=f"[E#{epoch}] {val_im_names[i]}")]

                log = {"train/loss": loss_value_reduce,
                       "train/lr": lr,
                       "train/MAE": batch_mae,
                       "train/RMSE": batch_mse ** 0.5,
                       "val/MAE": val_mae / len(data_loader_val),
                       "val/RMSE": (val_mse / len(data_loader_val)) ** 0.5,
                       "val/NAE": val_nae / len(data_loader_val),
                       "train_densitss": train_wandb_densities,
                       "val_densites": val_wandb_densities,
                       "train_boxes": train_wandb_bboxes,
                       "val_boxes": val_wandb_bboxes}
                wandb.log(log, step=idx)

            # save train status and model
            if args.output_dir and (epoch % save_freq == 0 or epoch + 1 == args.epochs) and epoch != 0:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, suffix=f"finetuning_{epoch}", upload=epoch % 100 == 0)
            elif True:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, suffix=f"finetuning_last", upload=False)
            if args.output_dir and val_mae / len(data_loader_val) < min_MAE:
                min_MAE = val_mae / len(data_loader_val)
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, suffix="finetuning_minMAE")

            print(f'[Train Epoch #{epoch}] - MAE: {train_mae.item() / len(data_loader_train):5.2f}, RMSE: {(train_mse.item() / len(data_loader_train)) ** 0.5:5.2f}', flush=True)
            print(f'[Val Epoch #{epoch}] - MAE: {val_mae.item() / len(data_loader_val):5.2f}, RMSE: {(val_mse.item() / len(data_loader_val)) ** 0.5:5.2f}, NAE: {val_nae.item() / len(data_loader_val):5.2f}', flush=True)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    finally:
        if wandb_run is not None:
            wandb.run.finish()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    data_path = Path(args.data_path)
    anno_file = data_path / args.anno_file
    data_split_file = data_path / args.data_split_file
    im_dir = data_path / args.im_dir

    if args.do_aug:
        class_file = data_path / args.class_file
    else:
        class_file = None

    args.anno_file = anno_file
    args.data_split_file = data_split_file
    args.im_dir = im_dir
    args.class_file = class_file

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
