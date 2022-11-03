#!/usr/bin/env bash
CURDIR=$(cd $(dirname $0); pwd)
cd ${CURDIR}
echo 'The work dir is: ' ${CURDIR}

TYPE=$1
JOB_NAME=$2
ARCH=$3
ngpus=$4


if [[ $TYPE =~ imagenet_pretrain ]]; then
    echo "start pretrain ====>"
    python3 -m torch.distributed.launch --use_env \
        --nnodes=${nnodes:-1} \
        --nproc_per_node=${ngpus:-8} \
        --node_rank ${NODE_ID:-0} \
        --master_addr ${MASTER_ADDR:-127.0.0.1} \
        --master_port ${MASTER_PORT:-12345} \
        ${CURDIR}/run_pretrain.py \
        --data_path ${IMAGENET_DIR:-'~/data_path'} \
        --output_dir ${CURDIR}/${JOB_NAME} \
        --log_dir ${CURDIR}/${JOB_NAME} \
        --model beit_${ARCH} \
        --num_mask_patches 118 \
        --second_input_size 224 \
        --second_interpolation bicubic \
        --tokenizer_type dino \
        --tokenizer_model base \
        --batch_size 128 \
        --lr 1.5e-3 \
        --warmup_epochs 10 \
        --epochs 300 \
        --clip_grad 3.0 \
        --drop_path 0. \
        --layer_scale_init_value 0.1 \
        --imagenet_default_mean_and_std \
        --opt_betas 0.9 0.98 \
        --opt_eps 1e-6 \
        --loss_type l2 \
        --post_patch_norm false \
        --tokenizer_weight_path ${CURDIR}/${JOB_NAME}/checkpoint-teacher-last.pth \
        --resume ${CURDIR}/${JOB_NAME}/checkpoint-last.pth \
        ${@:5}
fi

if [[ $TYPE =~ imagenet_cls ]]; then
    echo "start finetune ====>"
    PRETRAIN_CHKPT=$5


    python3 -m torch.distributed.launch --use_env \
        --nnodes=${nnodes:-1} \
        --nproc_per_node=${ngpus:-8} \
        --node_rank ${NODE_ID:-0} \
        --master_addr ${MASTER_ADDR:-127.0.0.1} \
        --master_port ${MASTER_PORT:-12345} \
        ${CURDIR}/run_finetune.py \
        --data_path ${IMAGENET_DIR:-'~/data_path'} \
        --output_dir ${CURDIR}/${JOB_NAME} \
        --log_dir ${CURDIR}/${JOB_NAME} \
        --model base_patch16_224 \
        --weight_decay 0.05 \
        --nb_classes 1000 \
        --finetune ${PRETRAIN_CHKPT} \
        --batch_size 128 \
        --lr 1e-3 \
        --update_freq 1 \
        --warmup_epochs 20 \
        --epochs 100 \
        --layer_decay 0.65 \
        --drop_path 0.1 \
        --mixup 0.8 \
        --cutmix 1.0 \
        --imagenet_default_mean_and_std \
        --dist_eval \
        --resume ${CURDIR}/${JOB_NAME}/checkpoint-last.pth \
        ${@:6}
fi