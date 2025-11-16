#!/usr/bin/env bash

for wd in 0.06 0.07; do
  for lr in 1e-3 3e-3; do
    echo "Running with weight_decay=${wd}, learning_rate=${lr}"

    python3 src/main_finetune.py \
        --mode finetune \
        --output-dir "$GCS_DATASET_DIR/CKPT" \
        --pretrained-ckpt "$GCS_MODEL_PATH" \
        --train-dataset-shards "$GCS_DATASET_DIR/imagenet-1k-wds/imagenet1k-train-{0000..1023}.tar" \
        --valid-dataset-shards "$GCS_DATASET_DIR/imagenet-1k-wds/imagenet1k-validation-{00..63}.tar" \
        --train-batch-size 1024 \
        --valid-batch-size 512 \
        --train-loader-workers 40 \
        --valid-loader-workers 10 \
        --random-crop rrc \
        --color-jitter 0.0 \
        --auto-augment "rand-m9-mstd0.5-inc1" \
        --random-erasing 0.0 \
        --augment-repeats 1 \
        --test-crop-ratio 0.875 \
        --mixup 0.8 \
        --cutmix 1.0 \
        --criterion ce \
        --label-smoothing 0.1 \
        --layers 12 \
        --dim 768 \
        --heads 12 \
        --labels 1000 \
        --patch-size 16 \
        --image-size 224 \
        --posemb sincos2d \
        --pooling cls \
        --dropout 0.0 \
        --droppath 0.1 \
        --init-seed 1 \
        --mixup-seed 1 \
        --dropout-seed 1 \
        --shuffle-seed 1 \
        --optimizer adamw \
        --learning-rate "${lr}" \
        --weight-decay "${wd}" \
        --lr-decay 0.65 \
        --clip-grad 0.0 \
        --grad-accum 1 \
        --warmup-steps $((1281167 * 10 / 1024)) \
        --training-steps $((1281167 * 110 / 1024)) \
        --log-interval 10 \
        --eval-interval $((1281167 * 1 / 1024)) \
        --project MAE-JAX \
        --name "$(basename "$0" .sh)_wd${wd}_lr${lr}" \
        --ipaddr "$(curl -s ifconfig.me)" \
        --hostname "$(hostname)"
  done
done
