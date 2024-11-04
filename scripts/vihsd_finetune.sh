#!/bin/bash
# WANDB
export WANDB_PROJECT="ViHSD"

# General variables
CACHE_DIR="/vol/tmp/truongph"
DATASET_NAME="phucdev/ViHSD"
TEXT_COLUMN="free_text"
LABEL_COLUMN="label_id"
MAX_SEQ_LENGTH=128
PER_DEVICE_TRAIN_BATCH_SIZE=32
PER_DEVICE_EVAL_BATCH_SIZE=32
GRADIENT_ACCUMULATION_STEPS=1
LEARNING_RATE=2e-5
NUM_EPOCHS=5
LR_SCHEDULER_TYPE="linear"
METRIC_NAME="phucdev/vihsd"
WARMUP_RATIO=0.1

# Variables for model
MODEL_NAME_OR_PATH="results/roberta-random_init"
OUTPUT_DIR="results/downstream/vihsd/roberta-random_init"

# Loop to run training 5 times with different random seeds
for SEED in 42 123 456 789 101112
do
    OUTPUT_DIR_WITH_SEED="${OUTPUT_DIR}/seed_${SEED}"

    echo "Running training with seed ${SEED}..."
    
    CUDA_VISIBLE_DEVICES=0 python lm_transfer/training/run_classification.py \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --dataset_name ${DATASET_NAME} \
        --cache_dir ${CACHE_DIR} \
        --shuffle_train_dataset \
        --text_column_name ${TEXT_COLUMN} \
        --label_column_name ${LABEL_COLUMN} \
        --do_train \
        --do_eval \
        --do_predict \
        --max_seq_length ${MAX_SEQ_LENGTH} \
        --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
        --per_device_eval_batch_size ${PER_DEVICE_EVAL_BATCH_SIZE} \
        --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
        --lr_scheduler_type ${LR_SCHEDULER_TYPE} \
        --learning_rate ${LEARNING_RATE} \
        --warmup_ratio ${WARMUP_RATIO} \
        --num_train_epochs ${NUM_EPOCHS} \
        --eval_strategy epoch \
        --save_strategy epoch \
        --save_total_limit 1 \
        --load_best_model_at_end \
        --metric_name ${METRIC_NAME} \
        --metric_for_best_model "accuracy" \
        --seed ${SEED} \
        --output_dir ${OUTPUT_DIR_WITH_SEED}

    echo "Finished training with seed ${SEED}."
done
