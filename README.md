# Masked Autoencoding with dBOT <img width="32" alt="dBOT Icon" src=".github/dbot.png">

This is the official PyTorch implementation of [Exploring target representations for masked autoencoders](https://arxiv.org/abs/2209.03917).
This branch is the implementation of dBOT with symmetric architecture with which we use CLIP as the pre-trained teacher.

<p align="center">
  <img src=".github/arch.png" width = "80%">
</p>


## Installation
Installation and preparation please follow [BEiT](https://github.com/microsoft/unilm/tree/master/beit) and [iBOT](https://github.com/bytedance/ibot).
This repo is based on `python==3.6`, `timm==0.4.12` and `pytorch==1.9.0`.

## Pre-training
See [pre-training instruction](PRETRAIN.md) for details.

## Downstream tasks
See [downstream instruction](DOWNSTREAM.md) for details.


## Acknowledgement

This branch is modified upon the [BEiT repository](https://github.com/microsoft/unilm/tree/master/beit).
## License

This project is under the Apache 2.0 license as found in [LICENSE](LICENSE) file.

## Citing dBOT

Please consider citing dBOT and giving a star if dBOT helps your research:
```
@article{liu2022exploring,
  title={Exploring target representations for masked autoencoders},
  author={Liu, Xingbin and Zhou, Jinghao and Kong, Tao and Lin, Xianming and Ji, Rongrong},
  journal={arXiv preprint arXiv:2209.03917},
  year={2022}
}
``` 
