# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import builtins
import datetime
import os
import time
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Union

import pandas as pd
import torch
import torch.distributed as dist
import wandb
from torch._six import inf
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2
from tqdm import tqdm


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, suffix="", upload=True):
    if suffix:
        suffix = f"__{suffix}"
    output_dir = Path(args.output_dir)
    ckpt_name = f"checkpoint{suffix}.pth"
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ckpt_name]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }
            save_on_master(to_save, checkpoint_path)
            if upload and is_main_process():
                log_wandb_model(f"checkpoint{suffix}", checkpoint_path, epoch)
            print("checkpoint sent to W&B (if)")
    else:
        client_state = {'epoch': epoch}
        model.save_checkpoint(save_dir=args.output_dir, tag=ckpt_name, client_state=client_state)
        if upload and is_main_process():
            log_wandb_model(f"checkpoint{suffix}", output_dir / ckpt_name, epoch)
        print("checkpoint sent to W&B (else)")


def log_wandb_model(title, path, epoch):
    artifact = wandb.Artifact(title, type="model")
    artifact.add_file(path)
    artifact.metadata["epoch"] = epoch
    wandb.log_artifact(artifact_or_path=artifact, name=title)


def load_model(args, model_without_ddp, optimizer, loss_scaler):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        if 'pos_embed' in checkpoint['model'] and checkpoint['model']['pos_embed'].shape != model_without_ddp.state_dict()['pos_embed'].shape:
            print(f"Removing key pos_embed from pretrained checkpoint")
            del checkpoint['model']['pos_embed']

        if 'decoder_pos_embed' in checkpoint['model'] and checkpoint['model']['decoder_pos_embed'].shape != model_without_ddp.state_dict()['decoder_pos_embed'].shape:
            print(f"Removing key decoder_pos_embed from pretrained checkpoint")
            del checkpoint['model']['decoder_pos_embed']

        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")

def load_model_FSC(args, model_without_ddp):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        if 'pos_embed' in checkpoint['model'] and checkpoint['model']['pos_embed'].shape != model_without_ddp.state_dict()['pos_embed'].shape:
            print(f"Removing key pos_embed from pretrained checkpoint")
            del checkpoint['model']['pos_embed']

        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        print(f"Resume checkpoint {args.resume} ({checkpoint['epoch']})")

def load_model_FSC1(args, model_without_ddp):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
            #model = timm.create_model('vit_base_patch16_224', pretrained=True)
            #torch.save(model.state_dict(), './output_abnopre_dir/checkpoint-6657.pth')
            checkpoint1 = torch.load('./output_abnopre_dir/checkpoint-6657.pth', map_location='cpu')

        if 'pos_embed' in checkpoint['model'] and checkpoint['model']['pos_embed'].shape != model_without_ddp.state_dict()['pos_embed'].shape:
            print(f"Removing key pos_embed from pretrained checkpoint")
            del checkpoint['model']['pos_embed']

        del checkpoint1['cls_token'],checkpoint1['pos_embed']

        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        model_without_ddp.load_state_dict(checkpoint1, strict=False)
        print("Resume checkpoint %s" % args.resume)


def load_model_FSC_full(args, model_without_ddp, optimizer, loss_scaler):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        if 'pos_embed' in checkpoint['model'] and checkpoint['model']['pos_embed'].shape != \
                model_without_ddp.state_dict()['pos_embed'].shape:
            print(f"Removing key pos_embed from pretrained checkpoint")
            del checkpoint['model']['pos_embed']

        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        print("Resume checkpoint %s" % args.resume)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint and args.do_resume:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & scheduler!")


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x


def plot_counts(res_csv: Union[str, list[str]], output_dir: str, suffix: str = "", smooth: bool = False):
    if suffix:
        suffix = f"_{suffix}"
    if smooth:
        suffix = f"_smooth{suffix}"
    if type(res_csv) == str:
        res_csv = [res_csv]

    plt.figure(figsize=(15, 5))

    for res in res_csv:
        name = Path(res).parent.name
        df = pd.read_csv(res)
        print(df)

        df.sort_values(by="name", inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.index += 1
        print(df)

        if smooth:
            time_arr = df.index[5:-5]
            smooth_pred_mean = df['prediction'].iloc[5:-5].rolling(25).mean()
            smooth_pred_std = df['prediction'].iloc[5:-5].rolling(25).std()
            plt.plot(time_arr, smooth_pred_mean, label=name)
            plt.fill_between(time_arr, smooth_pred_mean + smooth_pred_std, smooth_pred_mean - smooth_pred_std, alpha=.2)
            plt.xlabel('Frame')
            plt.ylabel('Count')
        else:
            plt.plot(df.index, df['prediction'], label=name)

    plt.legend()
    plt.savefig(os.path.join(output_dir, f'counts{suffix}.png'), dpi=300)


def write_zeroshot_annotations(p: Path):
    with open(p / 'annotations.json', 'a') as split:
        split.write('{\n')
        for img in p.iterdir():
            if img.is_file():
                split.write(f'  "{img.name}": {{\n' \
                            '    "H": 960,\n' \
                            '    "W": 1280,\n' \
                            '    "box_examples_coordinates": [],\n' \
                            '    "points": []\n' \
                            '  },\n')
        split.write("}")

    with open(p / 'split.json', 'a') as split:
        split.write('{\n  "test":\n  [\n')
        for img in p.iterdir():
            if img.is_file():
                split.write(f'    "{img.name}",\n')
        split.write("  ]\n}")


def make_grid(imgs, h, w):
    assert len(imgs) == 9
    rows = []
    for i in range(0, 9, 3):
        row = torch.cat((imgs[i], imgs[i + 1], imgs[i + 2]), -1)
        rows += [row]
    grid = torch.cat((rows[0], rows[1], rows[2]), 0)
    grid = transforms.Resize((h, w))(grid.unsqueeze(0))
    return grid.squeeze(0)


def min_max(t):
    t_shape = t.shape
    t = t.view(t_shape[0], -1)
    t -= t.min(1, keepdim=True)[0]
    t /= t.max(1, keepdim=True)[0]
    t = t.view(*t_shape)
    return t


def min_max_np(v, new_min=0, new_max=1):
    v_min, v_max = v.min(), v.max()
    return (v - v_min) / (v_max - v_min) * (new_max - new_min) + new_min


def get_box_map(sample, pos, device, external=False):
    box_map = torch.zeros([sample.shape[1], sample.shape[2]], device=device)
    if external is False:
        for rect in pos:
            for i in range(rect[2] - rect[0]):
                box_map[min(rect[0] + i, sample.shape[1] - 1), min(rect[1], sample.shape[2] - 1)] = 10
                box_map[min(rect[0] + i, sample.shape[1] - 1), min(rect[3], sample.shape[2] - 1)] = 10
            for i in range(rect[3] - rect[1]):
                box_map[min(rect[0], sample.shape[1] - 1), min(rect[1] + i, sample.shape[2] - 1)] = 10
                box_map[min(rect[2], sample.shape[1] - 1), min(rect[1] + i, sample.shape[2] - 1)] = 10
        box_map = box_map.unsqueeze(0).repeat(3, 1, 1)
    return box_map


timerfunc = time.perf_counter

class measure_time(object):
    def __enter__(self):
        self.start = timerfunc()
        return self

    def __exit__(self, typ, value, traceback):
        self.duration = timerfunc() - self.start

    def __add__(self, other):
        return self.duration + other.duration

    def __sub__(self, other):
        return self.duration - other.duration
    
    def __str__(self):
        return str(self.duration)


def log_test_results(test_dir):
    test_dir = Path(test_dir)
    logs = []
    for d in test_dir.iterdir():
        if d.is_dir() and (d / "log.txt").exists():
            print(d.name)
            with open(d / "log.txt") as f:
                last = f.readlines()[-1]
                j = json.loads(last)
                j['name'] = d.name
                logs.append(j)
    df = pd.DataFrame(logs)

    df.sort_values('name', inplace=True, ignore_index=True)
    cols = list(df.columns)
    cols = cols[-1:] + cols[:-1]
    df = df[cols]

    df.to_csv(test_dir / "logs.csv", index=False)


COLORS = {
    'muted blue': '#1f77b4',
    'safety orange': '#ff7f0e',
    'cooked asparagus green': '#2ca02c',
    'brick red': '#d62728',
    'muted purple': '#9467bd',
    'chestnut brown': '#8c564b',
    'raspberry yogurt pink': '#e377c2',
    'middle gray': '#7f7f7f',
    'curry yellow-green': '#bcbd22',
    'blue-teal': '#17becf',
    'muted blue light': '#419ede',
    'safety orange light': '#ffa85b',
    'cooked asparagus green light': '#4bce4b',
    'brick red light': '#e36667'
}


def plot_test_results(test_dir):
    import plotly.graph_objects as go

    test_dir = Path(test_dir)
    df = pd.read_csv(test_dir / "logs.csv")
    df.sort_values('name', inplace=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['name'], y=df['MAE'], line_color=COLORS['muted blue'],
                        mode='lines', name='MAE'))
    fig.add_trace(go.Scatter(x=df['name'], y=df['RMSE'], line_color=COLORS['safety orange'],
                        mode='lines', name='RMSE'))
    fig.add_trace(go.Scatter(x=df['name'], y=df['NAE'], line_color=COLORS['cooked asparagus green'],
                        mode='lines', name='NAE'))

    fig.update_yaxes(type="log")
    fig.write_image(test_dir / "plot.jpeg", scale=4)
    fig.write_html(test_dir / "plot.html", auto_open=False)


def frames2vid(input_dir: str, output_file: str, pattern: str, fps: int, h=720, w=1280):
    input_dir = Path(input_dir)
    video_file = None
    files = sorted(input_dir.glob(pattern))
    video_file = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for img in tqdm(files, total=len(files)):
        frame = cv2.imread(str(img))
        frame = cv2.resize(frame, (w, h))
        video_file.write(frame)

    video_file.release()
