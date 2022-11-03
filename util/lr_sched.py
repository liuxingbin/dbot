# Copyright (c) ByteDance, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
import math

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    # after warm-up use cosine learning rate

    end_epoch = 0.0
    #compute effective epoch for multi stage
    for index, num in enumerate(args.stage_epochs):
        if epoch < num:
            start_epoch = args.stage_epochs[index-1]
            end_epoch = num - start_epoch
            epoch -= start_epoch
            break


    warmup_epochs = args.warmup_epochs
        


    if epoch < warmup_epochs:
        lr = args.lr * epoch / warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - warmup_epochs) / (end_epoch - warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
