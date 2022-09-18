import argparse
import datetime
import json
from multiprocessing import reduction
from matplotlib import image
import numpy as np
import os
import time
import random
from pathlib import Path
import math
import sys
from PIL import Image
import scipy.ndimage as ndimage

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF

import hub

import timm
assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import util.lr_sched as lr_sched
from util.FSC147 import TransformTrain
import models_mae_cross


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=8, type=int,
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
    parser.add_argument('--data_path', default='/GPFS/data/changliu/FSC147/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_CARPK_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_CARPK_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    #parser.add_argument('--resume', default='./output_pre_4_dir/checkpoint-300.pth',
    #                    help='resume from checkpoint')
    parser.add_argument('--resume', default='./output_CARPK_dir/checkpoint-6.pth',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser

os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    ds_train = hub.load("hub://activeloop/carpk-train")
    #ds_test = hub.load("hub://activeloop/carpk-test")
    dataloader_train = ds_train.pytorch(num_workers=args.num_workers, batch_size=1, shuffle=False)
    #dataloader_test = ds_test.pytorch(num_workers=args.num_workers, batch_size=1, shuffle=False)
    
    # define the model
    model = models_mae_cross.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    model.to(device)

    model_without_ddp = model

    print("Model = %s" % str(model_without_ddp))

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
    
    criterion = nn.MSELoss()
    loss_scaler = NativeScaler()

    min_MAE = 99999

    misc.load_model_FSC(args=args, model_without_ddp=model_without_ddp)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        
        # train one epoch
        model.train(True)
        metric_logger = misc.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)
        print_freq = 20
        accum_iter = args.accum_iter

        # some parameters in training
        train_mae = 0
        train_rmse = 0
        pred_cnt = 0
        gt_cnt = 0

        optimizer.zero_grad()

        if log_writer is not None:
            print('log_dir: {}'.format(log_writer.log_dir))
        
        for data_iter_step, data in enumerate(metric_logger.log_every(dataloader_train, print_freq, header)):

            # Be careful
            if data_iter_step % accum_iter == 0:
                lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(dataloader_train) + epoch, args)

            samples = (data['images']/255).to(device, non_blocking=True).half()
            labels = data['labels'].to(device, non_blocking=True)
            samples = samples.transpose(2,3).transpose(1,2)

            cnt = 0
            boxes=[]
            idx = random.randint(0,data['boxes'].shape[1]-1)

            box = data['boxes'][0][idx]

            box2 = [int(k) for k in box]
            x1, y1, x2, y2 = box2[0], box2[1], box2[0]+box2[2], box2[1]+box2[3]
            bbox = samples[0,:,y1:y2+1,x1:x2+1].squeeze(0)
            bbox = transforms.Resize((64, 64))(bbox)
            boxes.append(bbox.cpu().numpy())

            boxes = np.array(boxes)
            boxes = torch.Tensor(boxes)
            boxes = boxes.unsqueeze(0)
            boxes = boxes.to(device, non_blocking=True).half()

            samples = transforms.Resize((384, 683))(samples[0])
            samples = TF.crop(samples, 0, 0, 384, 384)
            samples = samples.unsqueeze(0)

            gt_density = np.zeros((384, 384),dtype='float32')
            for box in data['boxes'][0]:
                box2 = [int(k) for k in box]
                x, y = int(box2[0]+box2[2]/2), int(box2[1]+box2[3]/2)
                if x < 720:
                    x = int(x * 384 / 720)
                    y = int(y * 384 / 720)
                    gt_density[y][x] = 1
            gt_density = ndimage.gaussian_filter(gt_density, sigma=(1, 1), order=0)
            gt_density = gt_density * 60
            gt_density = torch.from_numpy(gt_density)
            gt_density = gt_density.to(device, non_blocking=True).half()

            shot_num = 1
            with torch.cuda.amp.autocast():
                output = model(samples,boxes,shot_num)

            mask = np.random.binomial(n=1, p=0.8, size=[384,384])
            masks = np.tile(mask,(output.shape[0],1))
            masks = masks.reshape(output.shape[0], 384, 384)
            masks = torch.from_numpy(masks).to(device)
            loss = (output - gt_density) ** 2
            # loss = (loss * masks / (384*384)).sum() / output.shape[0]
            loss = (loss / (384*384)).sum() / output.shape[0]

            #loss = criterion(output, gt_density)
            loss_value = loss.item()

            batch_mae = 0
            batch_rmse = 0
            for i in range(output.shape[0]):
                pred_cnt = torch.sum(output[i]/60).item()
                gt_cnt = labels.shape[1]
                cnt_err = abs(pred_cnt - gt_cnt)
                batch_mae += cnt_err
                batch_rmse += cnt_err ** 2

                if i == 0 :
                    print(f'{data_iter_step}/{len(dataloader_train)}: loss: {loss_value},  pred_cnt: {pred_cnt},  gt_cnt: {gt_cnt},  error: {abs(pred_cnt - gt_cnt)},  AE: {cnt_err},  SE: {cnt_err ** 2}, {shot_num}-shot ')

            train_mae += batch_mae
            train_rmse += batch_rmse
                    
            '''if log_writer is not None and data_iter_step == 0:
                fig = output[0].unsqueeze(0).repeat(3,1,1)
                f1 = gt_density.unsqueeze(0).repeat(3,1,1)

                #log_writer.add_images('bboxes', (boxes[0]), int(epoch),dataformats='NCHW')
                log_writer.add_images('gt_density', (samples[0]/2+f1/2), int(epoch),dataformats='CHW')
                #log_writer.add_images('heatmap', (fig/20), int(epoch),dataformats='CHW')
                log_writer.add_images('heatmap overlay', (samples[0]/2+fig/2), int(epoch),dataformats='CHW')'''

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            loss /= accum_iter
            loss_scaler(loss, optimizer, parameters=model.parameters(),
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()
            
            torch.cuda.synchronize()

            metric_logger.update(loss=loss_value)

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

            loss_value_reduce = misc.all_reduce_mean(loss_value)
            if log_writer is not None and (data_iter_step + 1) % accum_iter == 3:
                """ We use epoch_1000x as the x-axis in tensorboard.
                This calibrates different curves when batch size changes.
                """
                epoch_1000x = int((data_iter_step / len(dataloader_train) + epoch) * 1000)
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)
                log_writer.add_scalar('MAE', batch_mae/args.batch_size, epoch_1000x)
                log_writer.add_scalar('RMSE', (batch_rmse/args.batch_size)**0.5, epoch_1000x)

            # Only use 1 batches when overfitting
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()} 

        # save train status and model
        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
        if args.output_dir and train_mae/(len(dataloader_train) * args.batch_size) < min_MAE:
            min_MAE = train_mae/(len(dataloader_train) * args.batch_size)
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=666)

        
        # len(dataloader_train) when train on all
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'Current MAE': train_mae/(len(dataloader_train) * args.batch_size),
                        'RMSE':  (train_rmse/(len(dataloader_train) * args.batch_size))**0.5,
                        'epoch': epoch,}

        print('Current MAE: {:5.2f}, RMSE: {:5.2f} '.format( train_mae/(len(dataloader_train) * args.batch_size), (train_rmse/(len(dataloader_train) * args.batch_size))**0.5))

        
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)