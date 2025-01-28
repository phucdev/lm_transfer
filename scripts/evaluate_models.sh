#!/bin/bash

accelerate launch lm_transfer/training/evaluate_language_modeling.py \
  --model_name_or_path=FacebookAI/xlm-roberta-base \
  --preprocessing_num_workers=4 \
  --per_device_eval_batch_size=8 \
  --language_modeling_objective=mlm \
  --output_dir=results/mlm/xlm-roberta-base \
  --validation_file data/culturax_vi/valid.json \
  --block_size 512 \
  --seed 42