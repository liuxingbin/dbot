## Downstream tasks for dBOT
### Catalog

- [x] Fine-tuning on ImageNet-1K
- [x] Object detection and instance segmentation on COCO
- [x] Semantic segmentation on ADE20K
- [x] Transfer classification on smaller datasets


### Fine-tuning on ImageNet-1K
To fine-tune ViT on ImageNet-1k with multi-node distributed training, run the following scripts on 4 nodes with 8 GPUs each.

- `effective_batch_size` is 1024 with `effective_batch_size = batch_size * nnodes * ngpus * accum_iter`.

ViT-B/16
```
./run.sh dbot_base_finetune \
imagenet_finetune \
vit_base_patch16 \
8 \
dbot_base_pre/checkpoint-1600.pth \
--epochs 100 \
--blr 3e-4 \
--batch_size 32 \
--data_path './data/imagenet'
```

ViT-L/16
```
./run.sh dbot_large_finetune \
imagenet_finetune \
vit_large_patch16 \
8 \
dbot_large_pre/checkpoint-1600.pth \
--drop_path 0.25 \
--clip_grad 1.0 \
--blr 3e-4 \
--layer_decay 0.75 \
--epochs 50 \
--batch_size 32 \
--data_path './data/imagenet'
```


ViT-H/14
```
./run.sh dbot_huge_finetune_224 \
imagenet_finetune \
vit_huge_patch14 \
8 \
dbot_huge_pre/checkpoint-1600.pth \
--drop_path 0.3 \
--blr 3e-4 \
--layer_decay 0.75 \
--epochs 50 \
--batch_size 16 \
--accum_iter 2 \
--data_path './data/imagenet'
```


ViT-H/14-448
```
./run.sh dbot_huge_finetune_448 \
imagenet_finetune \
vit_huge_patch14 \
8 \
dbot_huge_finetune_224/checkpoint-best.pth \
--drop_path 0.4 \
--blr 7e-5 \
--layer_decay 0.9 \
--input_size 448 \ 
--batch_size 8 \
--accum_iter 4 \
--epochs 10 \
--warmup_epochs 2 \
--data_path './data/imagenet'
```


### Object detection and instance segmentation on COCO
To fine-tune ViT on COCO with Cacade Mask R-CNN as the task layer, run the following scripts:


ViT-B/16
```
./run.sh dbot_base_cascade-coco-det \
cascade_coco_det \
vit_base \
8 \
dbot_base_pre/checkpoint-1600.pth \
data.samples_per_gpu=2 \
lr_config.step=8,11 \
runner.max_epochs=12 \
optimizer.paramwise_cfg.layer_decay_rate=0.75
```

ViT-L/16
```
./run.sh dbot_large_cascade-coco-det \
cascade_coco_det \
vit_large \
8 \
dbot_large_pre/checkpoint-1600.pth \
data.samples_per_gpu=2
lr_config.step=8,11
runner.max_epochs=12
optimizer.paramwise_cfg.layer_decay_rate=0.75
```



### Semantic segmentation on ADE20K
:dart: We opt for MAE's implementation for semantic segmentation following [mae_segmentation](https://github.com/implus/mae_segmentation) codebase. Specifically, we trained relative position embeddings from scratch on top of pre-trained fixed absolute position embeddings.  We emperically find better segmentation results for both MAE and dBOT using this than using [iBOT's implementation](https://github.com/bytedance/ibot/tree/main/evaluation/semantic_segmentation).

To fine-tune ViT on ADE20K with UperNet as the task layer, run the following scripts:

ViT-B/16
```
./run.sh dbot_base_ade20k-seg \
ade20k_seg \
base 8 \
dbot_base_pre/checkpoint-1600.pth \
optimizer_config.use_fp16=True \
optimizer.lr=1e-4 \
optimizer.paramwise_cfg.layer_decay_rate=0.65 \
model.backbone.drop_path_rate=0.2 \
--deterministic
```

ViT-L/16
```
./run.sh dbot_large_ade20k-seg \
ade20k_seg \
large 8 \
dbot_large_pre/checkpoint-1600.pth \
optimizer_config.use_fp16=True \
optimizer.lr=3e-5 \
optimizer.paramwise_cfg.layer_decay_rate=0.95 \
model.backbone.drop_path_rate=0.2 \
--deterministic
```

### Transfer classification on smaller datasets
To fine-tune ViT on smaller datasets, run the following scripts:


```
./run.sh dbot_base_transfer \
cifar10_cls+cifar_cls+cars_cls+flwrs_cls+inat_cls+inat19_cls 
vit_base 8 \
dbot_base_pre/checkpoint-1600.pth \
```
