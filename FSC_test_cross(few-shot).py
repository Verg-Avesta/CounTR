import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import pandas as pd

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF
import timm

assert "0.4.5" <= timm.__version__ <= "0.4.9"  # version check

import util.misc as misc
import models_mae_cross


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=1, type=int)
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
    parser.add_argument('--anno_file', default='annotation_FSC147_384.json', type=str,
                        help='annotation json file')
    parser.add_argument('--data_split_file', default='Train_Test_Val_FSC_147.json', type=str,
                        help='data split json file')
    parser.add_argument('--im_dir', default='images_384_VarV2', type=str,
                        help='images directory')
    parser.add_argument('--output_dir', default='./Image',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./Image',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='./output_fim6_dir/checkpoint-0.pth',
                        help='resume from checkpoint')
    parser.add_argument('--external', default=False,
                        help='True if using external exemplars')
    parser.add_argument('--box_bound', default=-1, type=int,
                        help='The max number of exemplars to be considered')

    # Training parameters
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.add_argument('--normalization', default=True, help='Set to False to disable test-time normalization')
    parser.set_defaults(pin_mem=True)

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser

os.environ["CUDA_LAUNCH_BLOCKING"] = '1'


class TestData(Dataset):
    def __init__(self, external: bool, box_bound: int = -1):

        self.img = data_split['test']
        self.img_dir = im_dir
        self.external = external
        self.box_bound = box_bound

        if external:
            self.external_boxes = []
            for anno in annotations:
                rects = []
                bboxes = annotations[anno]['box_examples_coordinates']

                if bboxes:
                    image = Image.open('{}/{}'.format(im_dir, anno))
                    image.load()
                    W, H = image.size

                    new_H = 384
                    new_W = 16 * int((W / H * 384) / 16)
                    scale_factor_W = float(new_W) / W
                    scale_factor_H = float(new_H) / H
                    image = transforms.Resize((new_H, new_W))(image)
                    Normalize = transforms.Compose([transforms.ToTensor()])
                    image = Normalize(image)

                    for bbox in bboxes:
                        x1 = int(bbox[0][0] * scale_factor_W)
                        y1 = int(bbox[0][1] * scale_factor_H)
                        x2 = int(bbox[1][0] * scale_factor_W)
                        y2 = int(bbox[1][1] * scale_factor_H)
                        rects.append([y1, x1, y2, x2])

                    for box in rects:
                        box2 = [int(k) for k in box]
                        y1, x1, y2, x2 = box2[0], box2[1], box2[2], box2[3]
                        bbox = image[:, y1:y2 + 1, x1:x2 + 1]
                        bbox = transforms.Resize((64, 64))(bbox)
                        self.external_boxes.append(bbox.numpy())

            self.external_boxes = np.array(self.external_boxes if self.box_bound < 0 else
                                           self.external_boxes[:self.box_bound])
            self.external_boxes = torch.Tensor(self.external_boxes)

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        im_id = self.img[idx]
        anno = annotations[im_id]
        bboxes = anno['box_examples_coordinates'] if self.box_bound < 0 else \
            anno['box_examples_coordinates'][:self.box_bound]
        dots = np.array(anno['points'])

        image = Image.open('{}/{}'.format(im_dir, im_id))
        image.load()
        W, H = image.size

        new_H = 384
        new_W = 16 * int((W / H * 384) / 16)
        scale_factor_W = float(new_W) / W
        scale_factor_H = float(new_H) / H
        image = transforms.Resize((new_H, new_W))(image)
        Normalize = transforms.Compose([transforms.ToTensor()])
        image = Normalize(image)

        boxes = list()
        if self.external:
            boxes = self.external_boxes
        else:
            rects = list()
            for bbox in bboxes:
                x1 = int(bbox[0][0] * scale_factor_W)
                y1 = int(bbox[0][1] * scale_factor_H)
                x2 = int(bbox[1][0] * scale_factor_W)
                y2 = int(bbox[1][1] * scale_factor_H)
                rects.append([y1, x1, y2, x2])

            for box in rects:
                box2 = [int(k) for k in box]
                y1, x1, y2, x2 = box2[0], box2[1], box2[2], box2[3]
                bbox = image[:, y1:y2 + 1, x1:x2 + 1]
                bbox = transforms.Resize((64, 64))(bbox)
                boxes.append(bbox.numpy())

            boxes = np.array(boxes)
            boxes = torch.Tensor(boxes)

        if self.box_bound >= 0:
            assert len(boxes) <= self.box_bound

        # Only for visualisation purpose, no need for ground truth density map indeed.
        gt_map = np.zeros((image.shape[1], image.shape[2]), dtype='float32')
        for i in range(dots.shape[0]):
            gt_map[min(new_H - 1, int(dots[i][1] * scale_factor_H))][min(new_W - 1, int(dots[i][0] * scale_factor_W))] = 1
        gt_map = ndimage.gaussian_filter(gt_map, sigma=(1, 1), order=0)
        gt_map = torch.from_numpy(gt_map)
        gt_map = gt_map * 60

        sample = {'image': image, 'dots': dots, 'boxes': boxes, 'pos': rects if self.external is False else [], 'gt_map': gt_map, 'name': im_id}
        return sample['image'], sample['dots'], sample['boxes'], sample['pos'], sample['gt_map'], sample['name']


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

    dataset_test = TestData(args.external, args.box_bound)
    print(dataset_test)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_test = %s" % str(sampler_test))
    else:
        sampler_test = torch.utils.data.RandomSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # define the model
    model = models_mae_cross.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    model.to(device)

    model_without_ddp = model

    # print("Model = %s" % str(model_without_ddp))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    misc.load_model_FSC(args=args, model_without_ddp=model_without_ddp)

    print(f"Start testing.")
    start_time = time.time()

    # test
    epoch = 0
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    # some parameters in training
    train_mae = 0
    train_rmse = 0

    loss_array = []
    gt_array = []
    pred_arr = []
    name_arr = []

    for data_iter_step, (samples, gt_dots, boxes, pos, gt_map, im_name) in \
            enumerate(metric_logger.log_every(data_loader_test, print_freq, header)):

        im_name = Path(im_name[0])
        samples = samples.to(device, non_blocking=True)
        gt_dots = gt_dots.to(device, non_blocking=True).half()
        boxes = boxes.to(device, non_blocking=True)
        num_boxes = boxes.shape[1] if boxes.nelement() > 0 else 0
        _, _, h, w = samples.shape

        r_cnt = 0
        s_cnt = 0
        for rect in pos:
            r_cnt += 1
            if r_cnt > 3:
                break
            if rect[2] - rect[0] < 10 and rect[3] - rect[1] < 10:
                s_cnt += 1

        if s_cnt >= 1:
            r_images = []
            r_densities = []
            r_images.append(TF.crop(samples[0], 0, 0, int(h / 3), int(w / 3)))
            r_images.append(TF.crop(samples[0], int(h / 3), 0, int(h / 3), int(w / 3)))
            r_images.append(TF.crop(samples[0], 0, int(w / 3), int(h / 3), int(w / 3)))
            r_images.append(TF.crop(samples[0], int(h / 3), int(w / 3), int(h / 3), int(w / 3)))
            r_images.append(TF.crop(samples[0], int(h * 2 / 3), 0, int(h / 3), int(w / 3)))
            r_images.append(TF.crop(samples[0], int(h * 2 / 3), int(w / 3), int(h / 3), int(w / 3)))
            r_images.append(TF.crop(samples[0], 0, int(w * 2 / 3), int(h / 3), int(w / 3)))
            r_images.append(TF.crop(samples[0], int(h / 3), int(w * 2 / 3), int(h / 3), int(w / 3)))
            r_images.append(TF.crop(samples[0], int(h * 2 / 3), int(w * 2 / 3), int(h / 3), int(w / 3)))

            pred_cnt = 0
            for r_image in r_images:
                r_image = transforms.Resize((h, w))(r_image).unsqueeze(0)
                density_map = torch.zeros([h, w])
                density_map = density_map.to(device, non_blocking=True)
                start = 0
                prev = -1

                with torch.no_grad():
                    while start + 383 < w:
                        output, = model(r_image[:, :, :, start:start + 384], boxes, num_boxes)
                        output = output.squeeze(0)
                        b1 = nn.ZeroPad2d(padding=(start, w - prev - 1, 0, 0))
                        d1 = b1(output[:, 0:prev - start + 1])
                        b2 = nn.ZeroPad2d(padding=(prev + 1, w - start - 384, 0, 0))
                        d2 = b2(output[:, prev - start + 1:384])

                        b3 = nn.ZeroPad2d(padding=(0, w - start, 0, 0))
                        density_map_l = b3(density_map[:, 0:start])
                        density_map_m = b1(density_map[:, start:prev + 1])
                        b4 = nn.ZeroPad2d(padding=(prev + 1, 0, 0, 0))
                        density_map_r = b4(density_map[:, prev + 1:w])

                        density_map = density_map_l + density_map_r + density_map_m / 2 + d1 / 2 + d2

                        prev = start + 383
                        start = start + 128
                        if start + 383 >= w:
                            if start == w - 384 + 128:
                                break
                            else:
                                start = w - 384

                pred_cnt += torch.sum(density_map / 60).item()
                r_densities += [density_map]
        else:
            density_map = torch.zeros([h, w])
            density_map = density_map.to(device, non_blocking=True)
            start = 0
            prev = -1
            with torch.no_grad():
                while start + 383 < w:
                    output, = model(samples[:, :, :, start:start + 384], boxes, num_boxes)
                    output = output.squeeze(0)
                    b1 = nn.ZeroPad2d(padding=(start, w - prev - 1, 0, 0))
                    d1 = b1(output[:, 0:prev - start + 1])
                    b2 = nn.ZeroPad2d(padding=(prev + 1, w - start - 384, 0, 0))
                    d2 = b2(output[:, prev - start + 1:384])

                    b3 = nn.ZeroPad2d(padding=(0, w - start, 0, 0))
                    density_map_l = b3(density_map[:, 0:start])
                    density_map_m = b1(density_map[:, start:prev + 1])
                    b4 = nn.ZeroPad2d(padding=(prev + 1, 0, 0, 0))
                    density_map_r = b4(density_map[:, prev + 1:w])

                    density_map = density_map_l + density_map_r + density_map_m / 2 + d1 / 2 + d2

                    prev = start + 383
                    start = start + 128
                    if start + 383 >= w:
                        if start == w - 384 + 128:
                            break
                        else:
                            start = w - 384

            pred_cnt = torch.sum(density_map / 60).item()

        if args.normalization:
            e_cnt = 0
            for rect in pos:
                e_cnt += torch.sum(density_map[rect[0]:rect[2] + 1, rect[1]:rect[3] + 1] / 60).item()
            e_cnt = e_cnt / 3
            if e_cnt > 1.8:
                pred_cnt /= e_cnt

        gt_cnt = gt_dots.shape[1]
        cnt_err = abs(pred_cnt - gt_cnt)
        train_mae += cnt_err
        train_rmse += cnt_err ** 2

        print(f'{data_iter_step}/{len(data_loader_test)}: pred_cnt: {pred_cnt},  gt_cnt: {gt_cnt},  error: {cnt_err},  AE: {cnt_err},  SE: {cnt_err ** 2}, id: {im_name.name}, s_cnt: {s_cnt >= 1}')

        loss_array.append(cnt_err)
        gt_array.append(gt_cnt)
        pred_arr.append(round(pred_cnt))
        name_arr.append(im_name.name)

        # compute and save images
        fig = samples[0]
        box_map = torch.zeros([fig.shape[1], fig.shape[2]], device=device)
        if args.external is False:
            for rect in pos:
                for i in range(rect[2] - rect[0]):
                    box_map[min(rect[0] + i, fig.shape[1] - 1), min(rect[1], fig.shape[2] - 1)] = 10
                    box_map[min(rect[0] + i, fig.shape[1] - 1), min(rect[3], fig.shape[2] - 1)] = 10
                for i in range(rect[3] - rect[1]):
                    box_map[min(rect[0], fig.shape[1] - 1), min(rect[1] + i, fig.shape[2] - 1)] = 10
                    box_map[min(rect[2], fig.shape[1] - 1), min(rect[1] + i, fig.shape[2] - 1)] = 10
            box_map = box_map.unsqueeze(0).repeat(3, 1, 1)
        pred = density_map.unsqueeze(0) if s_cnt < 1 else misc.make_grid(r_densities, h, w).unsqueeze(0)
        pred = torch.cat((pred, torch.zeros_like(pred), torch.zeros_like(pred))) * 5
        fig = fig + pred + box_map
        fig = torch.clamp(fig, 0, 1)

        pred_img = Image.new(mode="RGB", size=(w, h), color=(0, 0, 0))
        draw = ImageDraw.Draw(pred_img)
        draw.text((w-50, h-50), str(round(pred_cnt)), (255, 255, 255))
        pred_img = np.array(pred_img).transpose((2, 0, 1))
        pred_img = torch.tensor(np.array(pred_img), device=device) + pred
        full = torch.cat((samples[0], fig, pred_img), -1)
        torchvision.utils.save_image(full, (os.path.join(args.output_dir, f'full_{im_name.stem}__{round(pred_cnt)}{im_name.suffix}')))

        torch.cuda.synchronize()

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    log_stats = {'MAE': train_mae / (len(data_loader_test)),
                 'RMSE': (train_rmse / (len(data_loader_test))) ** 0.5}

    print('Current MAE: {:5.2f}, RMSE: {:5.2f} '.format(train_mae / (len(data_loader_test)), (
                train_rmse / (len(data_loader_test))) ** 0.5))

    if args.output_dir and misc.is_main_process():
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")

    plt.scatter(gt_array, loss_array)
    plt.xlabel('Ground Truth')
    plt.ylabel('Error')
    plt.savefig(os.path.join(args.output_dir, 'test_stat.png'))

    df = pd.DataFrame(data={'time': np.arange(data_iter_step+1)+1, 'name': name_arr, 'prediction': pred_arr})
    df.to_csv(os.path.join(args.output_dir, f'results.csv'), index=False)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Testing time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    # load data
    data_path = Path(args.data_path)
    anno_file = data_path / args.anno_file
    data_split_file = data_path / args.data_split_file
    im_dir = data_path / args.im_dir

    with open(anno_file) as f:
        annotations = json.load(f)

    with open(data_split_file) as f:
        data_split = json.load(f)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
