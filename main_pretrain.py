# Copyright (c) ByteDance, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MAE:https://github.com/facebookresearch/mae
# --------------------------------------------------------
from engine_pretrain import train_one_epoch
import models_mae
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import util.misc as misc
import timm.optim.optim_factory as optim_factory
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import shutil
import warnings
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def get_args_parser():
    parser = argparse.ArgumentParser('dBOT pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=1600, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters

    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    # multi-stage distill parameters
    parser.add_argument('--depth', nargs='+', type=int,
                    help="multi stage encoder depth e.g., 0 12 12 ")
    parser.add_argument('--init_teacher_encoder_depth', type=int, default=0,
                        help='initial encoder depth for teacher, special for multi-structure')
    parser.add_argument('--stage_epochs', nargs='+', type=int,
                    help="multi stage trainging, in ema-epochs, we drop momentum to 0, e.g., 0 800 1600")

    parser.add_argument('--tlayernorm', type=int, default=0, choices=[0, 1],
                        help="0: without teache rlayernorm \
                            1:with vit original self.norm")
    parser.add_argument('--sdpr', type=float, default=0.0,
                        help="vit drop path rate for student ")
    parser.add_argument('--tdpr', type=float, default=0.0,
                        help="vit drop path rate for teacher")
    parser.add_argument('--norm_feature_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.add_argument('--clip_grad', type=float,
                        default=None, help="clip grad when backpropogate")

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='~/data/imagenet', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--teacher_resume', default='',
                        help='teacher resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    # load dataset with num_workers
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


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    if misc.is_main_process():
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            for key, value in vars(args).items():
                f.write('%s:%s\n' % (key, value))
                print(key, value)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(
            0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)

    print(dataset_train)

    if True:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if args.resume and os.path.exists(args.resume):
        misc.load_start_epoch(args)
    init_index = misc.find_stage_index(args.start_epoch, args)
    args.teacher_encoder_depth = [
        args.init_teacher_encoder_depth] + args.depth[:-1]

    # define the model
    model = models_mae.__dict__[args.model](
        norm_feature_loss=args.norm_feature_loss,
        depth=args.depth[init_index],
        drop_path_rate=args.sdpr)

    teacher = models_mae.__dict__[args.model](
        depth=args.teacher_encoder_depth[init_index],
        decoder_depth=0,
        drop_path_rate=args.tdpr)

    model.to(device)
    teacher.to(device)

    model_without_ddp = model
    teacher_without_ddp = teacher
    if misc.is_main_process():
        print("Model = %s" % str(model_without_ddp))
        print("teacher = %s" % str(teacher_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    if misc.is_main_process():
        print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
        print("actual lr: %.2e" % args.lr)

        print("accumulate grad iterations: %d" % args.accum_iter)
        print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module


    teacher_without_ddp.load_state_dict(
        model.module.state_dict(), strict=False)

    for param in teacher.parameters():
        param.requires_grad = False

    param_groups = optim_factory.add_weight_decay(
        model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    if misc.is_main_process():
        print(optimizer)
    loss_scaler = NativeScaler()

    if args.resume and os.path.exists(args.resume):
        print("we use resume student")
        misc.load_model(args=args, model_without_ddp=model_without_ddp,
                        optimizer=optimizer, loss_scaler=loss_scaler)

    if args.teacher_resume and os.path.exists(args.teacher_resume):
        print("we use pretrain-teacher weight")
        misc.load_teacher_model(
            args=args, teacher_without_ddp=teacher_without_ddp)
    else:
        print("we use random initialized weight")

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, teacher_without_ddp, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        if args.output_dir and misc.is_main_process():
            # clean the information in log_writer
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        if misc.is_main_process() and ((epoch+1) in args.stage_epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
            misc.save_teacher_model(
                args=args, model=teacher, model_without_ddp=teacher_without_ddp, epoch=epoch)

        if (epoch+1) in args.stage_epochs and (epoch + 1) != args.epochs:
            index = misc.find_stage_index(epoch + 1, args)
            teacher = models_mae.__dict__[args.model](
                depth=args.teacher_encoder_depth[index],
                decoder_depth=0,
                drop_path_rate=args.tdpr)
            teacher.to(device)
            teacher_without_ddp = teacher
            for param in teacher.parameters():
                param.requires_grad = False

            msg = teacher_without_ddp.load_state_dict(
                model.module.state_dict(), strict=False)
            print(msg)

            model.module.initialize_weights()

        if misc.is_main_process() and (epoch+1) in args.stage_epochs:
            print("Model = %s" % str(model_without_ddp))
            print("teacher = %s" % str(teacher_without_ddp))

        # save last model for future resume
        if misc.is_main_process() and ((epoch+1) % 10 == 0):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, last=True)
            misc.save_teacher_model(
                args=args, model=teacher, model_without_ddp=teacher_without_ddp, last=1)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.log_dir = args.output_dir
    main(args)
