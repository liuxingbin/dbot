## Dowtream tasks for dBOT
### Catalog

- [x] Fine-tuning on ImageNet-1K
- [x] COCO detection
- [x] ADE20K segmentation

### Fine-tuning on ImageNet-1K
To fine-tune ViT on ImageNet-1k with multi-node distributed training, run the following scripts on 4 nodes with 8 GPUs each.

- effective batch size is 1024. `effective_batch_size = batch_size * nnodes * ngpus`

ViT-B/16
```
./run.sh imagenet_cls \
distill_clip-vit-base_finetune \
base \
8 \
distill_clip-vit-base_pre/checkpoint-1600.pth \
--epochs 100 \
--drop_path 0.1 \
--lr 2e-4 \
--batch_size 32
```

ViT-L/16
```
./run.sh imagenet_cls \
distill_clip-vit-large_finetune \
large \
8 \
distill_clip-vit-large_pre/checkpoint-1600.pth \
--lr 1e-4 \
--model large_patch16_224 \
--epochs 50 \
--warmup_epochs 5 \
--layer_decay 0.8 \
--drop_path 0.2 \
--batch_size 32 \
```


ViT-H/14
```
./run.sh imagenet_cls \
distill_clipl-vit-huge_finetune \
huge \
8 \
distill_clipl-vit-huge_pre/checkpoint-1600.pth \
--model huge_patch14_224 \
--epochs 50 \
--warmup_epochs 5 \
--lr 5e-5 \
--layer_decay 0.85 \
--drop_path 0.3 \
--batch_size 32 \
--save_ckpt_freq 5 \
--enable_deepspeed
```


ViT-H/14-448
```
.run.sh imagenet_cls \
distill_clipl-vit-huge_448-finetune \
huge \
8 \
distill_clipl-vit-huge_finetune/checkpoint-best.pth \
--model huge_patch14_448 \
--epochs 10 \
--warmup_epochs 1 \
--lr 5e-5 \
--layer_decay 0.95 \
--drop_path 0.4 \
--batch_size 32 \
--save_ckpt_freq 1 \
--input_size 448 \
--enable_deepspeed
```

### COCO detection
The code and scripts is in `main` branch.
To fine-tune ViT on COCO with Cascade Mask-RCNN as the task layer, run the following scripts:\
**NOTE**: The model are pre-trained with abs position embedding.

ViT-B/16
```
./finetuning.sh distill_clip-vit-base_coco \
cascade_coco_det \
beit_base \
8 \
distill_clip-vit-base_pre/checkpoint-1600.pth \
data.samples_per_gpu=2 \
lr_config.step=8,11 \
runner.max_epochs=12 \
model.backbone.use_checkpoint=True \
model.backbone.use_abs_pos_emb=True \
model.backbone.use_rel_pos_bias=False \
model.backbone.use_shared_rel_pos_bias=False
```

ViT-L/16
```
./finetuning.sh distill_clip-vit-large_coco \
cascade_coco_det \
beit_large \
8 \
distill_clip-vit-large_pre/checkpoint-1600.pth \
data.samples_per_gpu=2 \
lr_config.step=8,11 \
runner.max_epochs=12 \
model.backbone.use_checkpoint=True \
model.backbone.use_abs_pos_emb=True \
model.backbone.use_rel_pos_bias=False \
model.backbone.use_shared_rel_pos_bias=False
```





### ADE20K segmentation
:dart: We opt for iBOT's implementation for semantic segmentation following [iBOT](https://github.com/bytedance/ibot/tree/main/evaluation/semantic_segmentation) codebase.

The code and scripts is in `main` branch.
To fine-tune ViT on ADE20K with UperNet as the task layer, run the following scripts:


ViT-B/16
```
./finetuning.sh distill_clip-vit-base_seg \
ade20kv_seg \
beit_base \
8 \
distill_clip-vit-base_pre/checkpoint-1600.pth \
optimizer.lr=3e-5 \
model.backbone.use_checkpoint=True \
model.backbone.drop_path_rate=0.1 \
optimizer.paramwise_cfg.layer_decay_rate=0.85
```

ViT-L/16
```
./finetuning.sh distill_clip-vit-large_seg \
ade20kv_seg \
beit_large \
8 \
distill_clip-vit-large_pre/checkpoint-1600.pth \
optimizer.lr=9e-6 \
model.backbone.use_checkpoint=True \
model.backbone.drop_path_rate=0.1 \
optimizer.paramwise_cfg.layer_decay_rate=0.9
```
