# Training
This directory contains training scripts for:
- Tokenizer training
- Continued Pre-training of a Transformer model
- Fine-tuning a Transformer model on a downstream task
  - Token Classification
  - Text Classification
  - Question Answering

## Tokenizer Training

To train a tokenizer, you can use the `train_tokenizer.py` script. 
The script takes in a dataset & an existing tokenizer and trains a new tokenizer model. 

```bash
python train_tokenizer.py \
  --dataset_name_or_path=uonlp/CulturaX \
    --dataset_config_name=vi \
    --cache_dir=data/tokenizer_training \
    --original_tokenizer=FacebookAI/xlm-roberta-base \
    --vocab_size=50000 \
    --output_dir=tokenizers/vi-spm-culturax \
    --max_num_space_separated_tokens=16000
```
You can also use `max_num_bytes` to filter out examples that are too long for your purposes.
I used this filter because long input sequences can cause issues for sentencepiece.
As I am training a new tokenizer based on the XLM-RoBERTa tokenizer, which is based on sentencepiece with the 
unigram language model segmentation algorithm, long sequences can cause overflow during the Expectation-Maximization 
algorithm step.

## Continued Pre-training


## Fine-tuning
The fine-tuning scripts are completely based on Hugging Face's `transformers` PyTorch examples: https://github.com/huggingface/transformers/tree/main/examples/pytorch

### Token Classification
For Named Entity Recognition (NER) or Part-of-Speech (POS) tagging, you can use the `run_ner.py` script.

```bash
python lm_transfer/training/run_ner.py \
  --model_name FacebookAI/xlm-roberta-base \
  --dataset_name phucdev/PhoNER_COVID19 \
  --dataset_config_name word \
  --cache_dir /vol/tmp/truongph \
  --output_dir=outputs/ner \
  --do_train \
  --do_eval \
  --per_device_train_batch_size=8 \
  --per_device_eval_batch_size=8 \
  --num_train_epochs=3 \
  --seed=42
```

The script expects the following format of the data:
```json
{
    "chunk_tags": [11, 12, 12, 21, 13, 11, 11, 21, 13, 11, 12, 13, 11, 21, 22, 11, 12, 17, 11, 21, 17, 11, 12, 12, 21, 22, 22, 13, 11, 0],
    "id": "0",
    "ner_tags": [0, 3, 4, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "pos_tags": [12, 22, 22, 38, 15, 22, 28, 38, 15, 16, 21, 35, 24, 35, 37, 16, 21, 15, 24, 41, 15, 16, 21, 21, 20, 37, 40, 35, 21, 7],
    "tokens": ["The", "European", "Commission", "said", "on", "Thursday", "it", "disagreed", "with", "German", "advice", "to", "consumers", "to", "shun", "British", "lamb", "until", "scientists", "determine", "whether", "mad", "cow", "disease", "can", "be", "transmitted", "to", "sheep", "."]
}
```

PhoNER_COVID19 has a `tokens` and a `ner_tags` field, which is the format expected by the script.
I simply loaded the JSON files and pushed them to the huggingface hub, which converts them into the parquest format.

### Text Classification
For this thesis I have included a dataset for hate speech detection in Vietnamese (ViHSD) and a natural language 
inference dataset (XNLI).

For ViHSD, you can use the `run_text_classification.py` script::
```bash
python lm_transfer/training/run_classification.py \
    --model_name_or_path FacebookAI/xlm-roberta-base \
    --dataset_name phucdev/ViHSD \
    --cache_dir /vol/tmp/truongph \
    --shuffle_train_dataset \
    --metric_name f1 \
    --text_column_name free_text \
    --label_column_name label_id \
    --do_train \
    --do_eval \
    --max_seq_length 512 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --seed 42 \
    --output_dir outputs/ViHSD/
```

For XNLI, you can use the `run_xnli.py` script:
```bash
python lm_transfer/training/run_xnli.py \
  --model_name_or_path FacebookAI/xlm-roberta-base \
  --language vi \
  --train_language vi \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 8 \
  --learning_rate 5e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 128 \
  --output_dir outputs/xnli \
  --save_steps -1
```
This should run 106 mins on a single tesla V100 16GB.
Training with the previously defined hyperparameters yields the following results on the test set:
```
acc = 0.7093812375249501
```

### Question Answering
For Question Answering, you can use the `run_qa.py` script:
```bash
python lm_transfer/training/run_qa.py \
  --model_name_or_path=FacebookAI/xlm-roberta-base \
  --dataset_name=phucdev/ViMLQA \
  --output_dir=outputs/mlqa \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 4 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --seed=42
```

Note that if your dataset contains samples with no possible answers (like SQuAD version 2), you need to pass along the 
flag `--version_2_with_negative`.
Your dataset should be structured like SQuAD to ensure compatibility with the script.