# -*- coding: utf-8 -*-

from .checkpoint import load_checkpoint
from .layer_decay_optimizer_constructor import LayerDecayOptimizerConstructor
from .resize_transform import SETR_Resize
from .apex_runner.optimizer import DistOptimizerHook
from .train_api import train_segmentor
from .register_backbone import VisionTransformer
from .register_beit import BEiT
__all__ = ['load_checkpoint', 'LayerDecayOptimizerConstructor', 'SETR_Resize', 'DistOptimizerHook', 'train_segmentor']
