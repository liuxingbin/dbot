#!/bin/bash

CURDIR=$(cd $(dirname $0); pwd)
cd ${CURDIR}
echo 'The work dir is: ' ${CURDIR}

JOB_DIR=$1
# finetune or eval or seg or det
TYPE=$2
ARCH=$3
ngpus=$4


echo "save dir: ${JOB_DIR}"
echo "run type: ${TYPE}"
echo "use vit: ${ARCH}"
echo "num_gpu_per_node: ${ngpus:-8}"



if [[ ${TYPE} =~ imagenet_pretrain ]]; then
    echo "==>start imagenet pretrain on imagenet-1k"
    python3 -m torch.distributed.launch --use_env \
        --nnodes=${nnodes:-1} \
        --nproc_per_node=${ngpus:-8} \
        --node_rank ${NODE_ID:-0} \
        --master_addr ${MASTER_ADDR:-127.0.0.1} \
        --master_port ${MASTER_PORT:-12345} \
        ${CURDIR}/main_pretrain.py \
        --output_dir ${CURDIR}/${JOB_DIR} \
        --accum_iter ${accum_iter:-1} \
        --model mae_vit_${ARCH} \
        --warmup_epochs 40 \
        --weight_decay 0.05 \
        --data_path './data/imagenet' \
        --resume="${CURDIR}/${JOB_DIR}/checkpoint-last.pth" \
        --teacher_resume="${CURDIR}/${JOB_DIR}/checkpoint-teacher-last.pth" \
        ${@:5}

else
    PRETRAIN_CHKPT=$5
    echo "pretrain model path: ${PRETRAIN_CHKPT}"
    if [[ ${TYPE} =~ imagenet_finetune ]]; then
        
        echo "==>start finetune on imagenet-1k"

        python3 -m torch.distributed.launch --use_env \
            --nnodes=${nnodes:-1} \
            --nproc_per_node=${ngpus:-8} \
            --node_rank ${NODE_ID:-0} \
            --master_addr ${MASTER_ADDR:-127.0.0.1} \
            --master_port ${MASTER_PORT:-12345} \
            ${CURDIR}/main_finetune.py \
            --output_dir ${CURDIR}/${JOB_DIR} \
            --accum_iter ${accum_iter:-1} \
            --batch_size 32 \
            --model vit_base_patch16 \
            --finetune ${PRETRAIN_CHKPT} \
            --epochs ${epoch:-100} \
            --blr 5e-4 --layer_decay 0.65 \
            --weight_decay 0.05 \
            --drop_path 0.1 \
            --reprob 0.25 \
            --mixup 0.8 \
            --cutmix 1.0 \
            --dist_eval \
            --data_path './data/imagenet' \
            --resume "${CURDIR}/${JOB_DIR}/checkpoint-last.pth" \
            ${@:6}
    fi



    if [[ $TYPE =~ cascade_coco_det ]]; then
        echo "Starting evaluating on coco det with cascade_coco_det."

        python3 -m torch.distributed.launch --nproc_per_node=${ngpus:-8} \
            --master_port=${MASTER_PORT:-12345} \
            --master_addr ${METIS_WORKER_0_HOST} \
            $CURDIR/evaluation/object_detection/train.py \
            $CURDIR/evaluation/object_detection/configs/cascade_rcnn/${ARCH}_giou_4conv1f_coco_3x.py \
            --launcher pytorch \
            --work-dir ${CURDIR}/${JOB_DIR} \
            --deterministic \
            --resume-from ${CURDIR}/${JOB_DIR}/latest.pth \
            --cfg-options model.backbone.use_checkpoint=True \
            model.pretrained=${PRETRAIN_CHKPT} \
            ${@:6}
        python3 -m torch.distributed.launch --nproc_per_node=${ngpus:-8} \
            --master_port=${MASTER_PORT:-12345} \
            --master_addr ${METIS_WORKER_0_HOST} \
            $CURDIR/evaluation/object_detection/test.py \
            $CURDIR/evaluation/object_detection/configs/cascade_rcnn/${ARCH}_giou_4conv1f_coco_3x.py \
            ${CURDIR}/${JOB_DIR}/latest.pth \
            --launcher pytorch \
            --eval bbox segm \
            --cfg-options model.backbone.use_checkpoint=True \
            ${@:6}

    fi


    if [[ $TYPE =~ ade20k_seg ]]; then
        echo "Starting evaluating on seg."
        CONFIGS=upernet_mae_${ARCH}_12_512_slide_160k_ade20k.py
        python3 -m torch.distributed.launch --nproc_per_node=${ngpus:-8} \
            --master_port=$[${MASTER_PORT:-29500}] \
            ${CURDIR}/evaluation/mae_segmentation/train.py \
            ${CURDIR}/evaluation/mae_segmentation/configs/mae/${CONFIGS} \
            --launcher pytorch \
            --work-dir ${CURDIR}/${JOB_DIR} \
            --options model.pretrained=${PRETRAIN_CHKPT} \
            data_root=${CURDIR}/data/ade/ADEChallengeData2016 \
            ${@:6}
        python3 -m torch.distributed.launch --nproc_per_node=${ngpus:-8} \
            --master_port=$[${MASTER_PORT:-29500}] \
            ${CURDIR}/evaluation/mae_segmentation/test.py \
            ${CURDIR}/evaluation/mae_segmentation/configs/mae/${CONFIGS} \
            ${JOB_DIR}/iter_160000.pth \
            --launcher pytorch \
            --eval mIoU \
            --options ${@:6}

    fi

    if [[ $TYPE =~ ade20kv_seg ]]; then
        echo "Starting evaluating on seg with beit structure."
        CONFIGS=${ARCH}_512_ade20k_160k.py
        python3 -m torch.distributed.launch --nproc_per_node=${ngpus:-8} \
            --master_port=$[${MASTER_PORT:-29500}] \
            ${CURDIR}/evaluation/semantic_segmentation/train.py \
            ${CURDIR}/evaluation/semantic_segmentation/configs/upernet/${CONFIGS} \
            --launcher pytorch \
            --work-dir ${CURDIR}/${JOB_DIR} \
            --deterministic \
            --options model.pretrained=${PRETRAIN_CHKPT} \
            model.backbone.use_checkpoint=True \
            data_root=${CURDIR}/data/ade/ADEChallengeData2016 \
            ${@:6}

        python3 -m torch.distributed.launch --nproc_per_node=${ngpus:-8} \
            --master_port=$[${MASTER_PORT:-29500}] \
            ${CURDIR}/evaluation/semantic_segmentation/test.py \
            ${CURDIR}/evaluation/semantic_segmentation/configs/upernet/${CONFIGS} \
            ${CURDIR}/${JOB_DIR}/iter_160000.pth \
            --launcher pytorch \
            --eval mIoU \
            --options ${@:6}
    fi

    if [[ $TYPE =~ cifar_cls ]] && [[ ! $TYPE =~ pretrain ]]; then
        echo "start eval on cifar100"
        python3 -m torch.distributed.launch --nproc_per_node=${ngpus:-8} \
            --master_port=${MASTER_PORT:-29500} \
            ${CURDIR}/evaluation/eval_cls.py \
            --pretrained_weights ${PRETRAIN_CHKPT} \
            --avgpool_patchtokens ${AVGPOOL:-0} \
            --arch ${ARCH} \
            --output_dir ${CURDIR}/${JOB_DIR}/cls/cifar100 \
            --batch-size 96 \
            --lr 7.5e-6 \
            --epochs 1000 \
            --datasource CIFAR \
            --data data/cifar \
            ${@:6}
    fi


    if [[ $TYPE =~ cifar10_cls ]] && [[ ! $TYPE =~ pretrain ]]; then
        echo "start eval on cifar10"
        python3 -m torch.distributed.launch --nproc_per_node=${ngpus:-8} \
            --master_port=${MASTER_PORT:-29500} \
            ${CURDIR}/evaluation/eval_cls.py \
            --pretrained_weights ${PRETRAIN_CHKPT} \
            --avgpool_patchtokens ${AVGPOOL:-0} \
            --arch ${ARCH} \
            --output_dir ${CURDIR}/${JOB_DIR}/cls/cifar10 \
            --batch-size 96 \
            --lr 7.5e-6 \
            --epochs 1000 \
            --datasource CIFAR10 \
            --data data/cifar \
            ${@:6}
    fi

    if [[ $TYPE =~ cars_cls ]] && [[ ! $TYPE =~ pretrain ]]; then
        echo "start eval on cars"
        python3 -m torch.distributed.launch --nproc_per_node=${ngpus:-8} \
            --master_port=${MASTER_PORT:-29500} \
            ${CURDIR}/evaluation/eval_cls.py \
            --pretrained_weights ${PRETRAIN_CHKPT} \
            --avgpool_patchtokens ${AVGPOOL:-0} \
            --arch ${ARCH} \
            --output_dir ${CURDIR}/${JOB_DIR}/cls/cars \
            --batch-size 96 \
            --lr 7.5e-6 \
            --epochs 1000 \
            --datasource Cars \
            --data data/cars \
            ${@:6}
    fi


    if [[ $TYPE =~ flwrs_cls ]] && [[ ! $TYPE =~ pretrain ]]; then
        echo "strat eval on flwrs"
        python3 -m torch.distributed.launch --nproc_per_node=${ngpus:-8} \
            --master_port=${MASTER_PORT:-29500} \
            ${CURDIR}/evaluation/eval_cls.py \
            --pretrained_weights ${PRETRAIN_CHKPT} \
            --avgpool_patchtokens ${AVGPOOL:-0} \
            --arch ${ARCH} \
            --output_dir ${CURDIR}/${JOB_DIR}/cls/flwrs \
            --batch-size 96 \
            --lr 7.5e-6 \
            --epochs 1000 \
            --datasource Flwrs \
            --data data/flwrs \
            ${@:6}
    fi


    if [[ $TYPE =~ inat_cls ]] && [[ ! $TYPE =~ pretrain ]]; then
        echo "strat eval on inat"
        python3 -m torch.distributed.launch --nproc_per_node=${ngpus:-8} \
            --master_port=${MASTER_PORT:-29500} \
            ${CURDIR}/evaluation/eval_cls.py \
            --pretrained_weights ${PRETRAIN_CHKPT} \
            --avgpool_patchtokens ${AVGPOOL:-0} \
            --arch ${ARCH} \
            --output_dir ${CURDIR}/${JOB_DIR}/cls/inat \
            --batch-size 96 \
            --lr 7.5e-6 \
            --epochs 360 \
            --reprob 0.1 \
            --datasource INAT \
            --data data/inat/2018 \
            ${@:6}
    fi


    if [[ $TYPE =~ inat19_cls ]] && [[ ! $TYPE =~ pretrain ]]; then
        echo "start rval on inat19"
        python3 -m torch.distributed.launch --nproc_per_node=${ngpus:-8} \
            --master_port=${MASTER_PORT:-29500} \
            ${CURDIR}/evaluation/eval_cls.py \
            --pretrained_weights ${PRETRAIN_CHKPT} \
            --avgpool_patchtokens ${AVGPOOL:-0} \
            --arch ${ARCH} \
            --output_dir ${CURDIR}/${JOB_DIR}/cls/inat19 \
            --batch-size 96 \
            --lr 7.5e-6 \
            --epochs 360 \
            --reprob 0.1 \
            --datasource INAT19 \
            --data data/inat/2019 \
            ${@:6}
    fi

fi