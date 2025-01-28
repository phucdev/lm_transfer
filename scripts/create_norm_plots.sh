#!/bin/bash

python scripts/calculate_emb_norms.py --input_dir models/transfer/monolingual --xlim 7.5

python scripts/calculate_emb_norms.py --input_dir models/transfer/multilingual --xlim 7.5

python scripts/calculate_emb_norms.py --input_dir results/monolingual --xlim 7.5

python scripts/calculate_emb_norms.py --input_dir results/multilingual --xlim 7.5

python scripts/calculate_emb_norms.py \
  --output_dir models/source_models/roberta-base \
  --model_name_or_path FacebookAI/roberta-base \
  --is_source_model \
  --xlim 7.5

python scripts/calculate_emb_norms.py \
  --output_dir models/source_models/xlm-roberta-base \
  --model_name_or_path FacebookAI/xlm-roberta-base \
  --is_source_model \
  --xlim 7.5