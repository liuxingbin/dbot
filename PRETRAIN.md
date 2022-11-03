## Pre-training dBOT

To pre-train ViT with multi-node distributed training, run the following scripts on 4 nodes with 8 GPUs each.

- `effective batch size` is 4096 with `effective_batch_size = batch_size * nnodes * ngpus * accum_iter`.
- `blr` is the base learning rate for `batch_size` of 256.
- Other parameters are explained in args's description.
- `teacher_resume` is the wight path for the initialized teacher. Set it `NULL` for using the random teacher and `path_to_your_model` for using pre-trained teachers. 

ViT-B/16
```
./run.sh dbot_base_pre imagenet_pretrain base_patch16 8 \
--batch_size 128 \
--epochs 1600 \
--norm_feature_loss \
--mask_ratio 0.75 \
--stage_epochs 0 800 1600 \
--depth 12 12 12 \
--init_teacher_encoder_depth 12 \
--sdpr 0.2 \
--blr 1.5e-4 \
--weight_decay 0.05 \
--clip_grad 3.0 \
--data_path /path/to/your/data
```


ViT-L/16
```
./run.sh dbot_large_pre imagenet_pretrain large_patch16 8 \
--batch_size 128 \
--epochs 1600 \
--norm_feature_loss \
--mask_ratio 0.75 \
--stage_epochs 0 800 1600 \
--depth 24 24 24 \
--init_teacher_encoder_depth 0 \
--sdpr 0.2 \
--blr 1.5e-4 \
--weight_decay 0.05 \
--data_path /path/to/your/data
```

ViT-H/14
```
./run.sh dbot_huge_pre imagenet_pretrain huge_patch14 8 \
--batch_size 128 \
--epochs 1600 \
--norm_feature_loss \
--mask_ratio 0.75 \
--stage_epochs 0 800 1600 \
--depth 32 32 32 \
--init_teacher_encoder_depth 0 \
--sdpr 0.3 \
--blr 1.5e-4 \
--weight_decay 0.05 \
--data_path /path/to/your/data
```