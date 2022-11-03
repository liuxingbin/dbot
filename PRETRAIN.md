## Pre-training dBOT with CLIP as teacher

To pre-train ViT with multi-node distributed training, run the following scripts on 4 nodes with 8 GPUs each.

- effective batch size is 4096.
- learning rate is 3e-3 (bs=4096).
- other parameters are explained in args's description.

ViT-B/16
```
./run.sh imagenet_pretrain \
distill_clip-vit-base_pre \
base_patch16_224_8k_vocab \
8 \
--epochs 1600 \
--tokenizer_type clip \
--input_scale 0.4 1 \
--num_mask_patches 78 \
--loss_type negcosine \
--post_patch_norm True \
--lr 3e-3 \
--drop_path 0.1 \
--batch_size 128 \
--stage_epochs 0 1600 \
--data_path "data/imagenet"
```


ViT-L/16
```
./run.sh imagenet_pretrain \
distill_clip-vit-large_pre \
large_patch16_224_8k_vocab \
8 \
--epochs 1600 \
--tokenizer_type clip \
--input_scale 0.4 1 \
--num_mask_patches 78 \
--loss_type negcosine \
--post_patch_norm True \
--tokenizer_model large \
--second_input_size 196 \
--lr 3e-3 \
--drop_path 0.1 \
--batch_size 128 \
--stage_epochs 0 1600 \
--data_path "data/imagenet"
```

ViT-H/14
```
./run.sh imagenet_pretrain \
distill_clipl-vit-huge-pre \
huge_patch14_224_8k_vocab \
8 \
--epochs 1600 \
--tokenizer_type clip \
--input_scale 0.4 1 \
--num_mask_patches 78 \
--loss_type negcosine \
--post_patch_norm True \
--tokenizer_model large \
--batch_size 128 \
--drop_path 0.1 \
--lr 3e-3 \
--stage_epochs 0 1600 \
--data_path "data/imagenet"
```