#!/bin/bash

# Variables
MODEL_NAME_OR_PATH="results/roberta-random_init"
OUTPUT_DIR="results/downstream/phoner/roberta-random_init"
CACHE_DIR="/vol/tmp/truongph"
DATASET_NAME="phucdev/PhoNER_COVID19"
DATASET_CONFIG_NAME="syllable"
MAX_SEQ_LENGTH=256
PER_DEVICE_TRAIN_BATCH_SIZE=32
PER_DEVICE_EVAL_BATCH_SIZE=32
GRADIENT_ACCUMULATION_STEPS=1
LEARNING_RATE_SCHEDULER_TYPE="linear"
LEARNING_RATE=2e-5
WARMUP_RATIO=0.1
NUM_EPOCHS=5

# Loop to run training 5 times with different random seeds
for SEED in 42 123 456 789 101112
do
    OUTPUT_DIR_WITH_SEED="${OUTPUT_DIR}/seed_${SEED}"

    echo "Running training with seed ${SEED}..."

    CUDA_VISIBLE_DEVICES=0 python lm_transfer/training/run_ner.py \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --dataset_name ${DATASET_NAME} \
        --dataset_config_name ${DATASET_CONFIG_NAME} \
        --cache_dir ${CACHE_DIR} \
        --do_train \
        --do_eval \
        --do_predict \
        --max_seq_length ${MAX_SEQ_LENGTH} \
        --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
        --per_device_eval_batch_size ${PER_DEVICE_EVAL_BATCH_SIZE} \
        --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
        --learning_rate_scheduler_type ${LEARNING_RATE_SCHEDULER_TYPE} \
        --learning_rate ${LEARNING_RATE} \
        --warmup_ratio ${WARMUP_RATIO} \
        --num_train_epochs ${NUM_EPOCHS} \
        --eval_strategy epoch \
        --save_strategy epoch \
        --save_total_limit 1 \
        --load_best_model_at_end \
        --metric_for_best_model "overall_f1" \
        --seed ${SEED} \
        --output_dir ${OUTPUT_DIR_WITH_SEED}

    echo "Finished training with seed ${SEED}."
done
