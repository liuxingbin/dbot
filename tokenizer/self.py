import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import modeling_finetune
import os

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import functional as tf
from functools import partial, reduce
from collections import OrderedDict
from timm.models.layers import drop_path, to_2tuple, trunc_normal_

class SELF(nn.Module):
    def __init__(self, weight_path, tokenizer_model, image_size, post_patch_norm=True, 
                 imagenet_default_mean_and_std=True, **kwargs):
        super().__init__()
        self.net = eval('vit_{}'.format(tokenizer_model))(img_size=[image_size, image_size], **kwargs)
        self.net.post_patch_norm = post_patch_norm
        self.embed_dim = self.net.embed_dim

        if weight_path is None or not os.path.exists(weight_path):
            print('=> Using random self.')
        else:
            self.load_from_pretrained(weight_path)
        
        for param in self.parameters():
            param.requires_grad = False

        self.mean = IMAGENET_DEFAULT_MEAN
        self.std = IMAGENET_DEFAULT_STD

    def load_from_pretrained(self, weight_path):
        sd = torch.load(weight_path, map_location='cpu')['model']

        keys = list(sd.keys())
        new_dict = OrderedDict()
        
        for key in keys:
            new_dict['net.' + key] = sd[key]
        sd = new_dict

        missing_keys, unexpected_keys = self.load_state_dict(sd, strict=False)
        print(f"Load weight of self from: {weight_path}")
        print(f"missing_keys: {missing_keys}")
        print(f"unexpected_keys: {unexpected_keys}")

    def pre_process(self, x):
        return tf.normalize(x, self.mean, self.std, False)

    def forward(self, input, discard_cls_token=True):
        input = self.pre_process(input)
        out = self.net(input)
        if discard_cls_token:
            out = out[:, 1:]
        return out

class VisionTransformer(modeling_finetune.VisionTransformer):
    """ Vision Transformer """

    def forward(self, x):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)

        if not hasattr(self, 'post_patch_norm') or self.post_patch_norm:
            x = self.norm(x)
        else:
            x[:, 0] = self.norm(x[:, 0])
        return x


def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=0, **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=0, **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=0, **kwargs)
    return model