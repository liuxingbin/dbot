import torch
import torch.nn as nn

from torchvision.transforms import functional as tf
from clip import clip
import os

class CLIPFintune(nn.Module):
    def __init__(self, weight_path, model = 'base', num_classes=1000):
        super().__init__()
        self.net, _ = clip.load("ViT-{}/16".format(model[0].upper()), jit=False, download_root=os.path.dirname(weight_path), convert=False)
        self.net.visual.ln_post = nn.Identity()
        self.net.visual.proj = None
        self.embed_dim = 768
                
        self.fc_norm = nn.LayerNorm(768) 
        self.head = nn.Linear(768, num_classes)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}


    def pre_process(self, x):
        # follow clip https://github.com/openai/CLIP/blob/40f5484c1c74edd83cb9cf687c6ab92b28d8b656/clip/clip.py#L84
        return tf.normalize(x, (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711), False)

    def forward(self, input, caption=None, discard_cls_token=True, return_all_text_tokens=False):
        # input = self.pre_process(input)
        out = self.net.encode_image(input, return_all_tokens=True)
        # if discard_cls_token:
        #     out = out[:, 1:]
        # if caption is None:
        #     return out
        # tout = self.net.encode_text(caption, return_all_tokens=return_all_text_tokens)
        # return out, tout

        t = out[:, 1:, :]
        return self.head(self.fc_norm(t.mean(1)))