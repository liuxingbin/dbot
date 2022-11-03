import torch
import torch.nn as nn

from torchvision.transforms import functional as tf
from clip import clip
import os

class CLIP(nn.Module):
    def __init__(self, weight_path, tokenizer_model, image_size, post_patch_norm=True):
        super().__init__()
        if tokenizer_model[0].upper() == 'B':
            self.net, _ = clip.load("ViT-B/16", jit=False, download_root=os.path.dirname(weight_path))
        elif tokenizer_model[0].upper() == 'L':
            self.net, _ = clip.load("ViT-L/14", jit=False, download_root=os.path.dirname(weight_path))
        else:
            raise NotImplementedError
        self.net.visual.post_patch_norm = post_patch_norm
        self.embed_dim = self.net.visual.output_dim 
                
        for param in self.parameters():
            param.requires_grad = False

    def pre_process(self, x):
        # follow clip https://github.com/openai/CLIP/blob/40f5484c1c74edd83cb9cf687c6ab92b28d8b656/clip/clip.py#L84
        return tf.normalize(x, (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711), False)

    def forward(self, input, caption=None, discard_cls_token=True, return_all_text_tokens=False):
        input = self.pre_process(input)
        out = self.net.encode_image(input, return_all_tokens=True)
        if discard_cls_token:
            out = out[:, 1:]
        if caption is None:
            return out
        tout = self.net.encode_text(caption, return_all_tokens=return_all_text_tokens)
        return out, tout