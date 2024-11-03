export HF_DATASETS_CACHE="/vol/tmp/truongph/.cache/huggingface"

accelerate launch lm_transfer/training/run_language_modeling.py \
  --with_tracking \
  --report_to=wandb \
  --project_name=master-thesis \
  --run_name=roberta-random_init \
  --model_name_or_path=models/transfer/random_initialization \
  --num_train_epochs=1 \
  --preprocessing_num_workers=4 \
  --per_device_train_batch_size=16 \
  --per_device_eval_batch_size=16 \
  --language_modeling_objective=mlm \
  --output_dir=results/roberta-random_init \
  --eval_steps=300 \
  --eval_iters=50 \
  --train_file data/culturax_vi/train.json \
  --validation_file data/culturax_vi/valid.json \
  --learning_rate 6e-4 \
  --adam_epsilon 1e-6 \
  --beta1 0.9 \
  --beta2 0.98 \
  --save_preprocessed_dataset_path data/culturax_vi/preprocessed_bpe \
  --lr_scheduler_type constant_with_warmup \
  --num_warmup_steps 2613 \
  --block_size 512 \
  --seed 42