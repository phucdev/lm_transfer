# MA-Exploration
Repo for my master thesis on Cross-lingual transfer of pre-trained language models to Vietnamese

## Getting started
1. Clone the repository
    ```bash
    git clone git@github.com:phucdev/MA-Exploration.git
    ```
2. Install the requirements
    ```bash
    pip install -r requirements.txt
    ```
3. Install fast_align
    ```bash
    sudo apt-get install libgoogle-perftools-dev libsparsehash-dev build-essential
    chmod +x install_fast_align.sh
    ./scripts/install_fast_align.sh
    ```
   
## Data
In order to reproduce the [CulturaX](https://huggingface.co/datasets/uonlp/CulturaX) sample for pre-training, you need to run:
```bash
python src/data_utils/get_train_data.py \
  --dataset_name=uonlp/CulturaX \
  --dataset_config_name=vi \
  --output_dir=data/culturax_vi \
  --subsample_size_mb=4096 \
  --only_keep_text
```

## Tokenizer
You can access the Vietnamese tokenizers on the huggingface hub or reproduce them like so:
- vi-bpe-culturax-4g-sample (BPE tokenizer trained on 4GB sample of CulturaX): https://huggingface.co/phucdev/vi-bpe-culturax-4g-sample
  ```bash
    python src/training/train_tokenizer.py \
        --dataset_name=data/culturax_vi/train.json \
        --output_dir=tokenizers/vi-bpe-culturax-2048 \
        --original_tokenizer=FacebookAI/roberta-base \
        --vocab_size=50000 \
        --max_num_bytes=2048
    ```
- vi-spm-culturax-4g-sample (SentencePiece + Unigram LM tokenizer trained on 4GB sample of CulturaX): https://huggingface.co/phucdev/vi-spm-culturax-4g-sample
  ```bash
    python src/training/train_tokenizer.py \
        --dataset_name=uonlp/CulturaX \
        --dataset_config_name=vi \
        --output_dir=tokenizers/vi-bpe-culturax-4g-sample \
        --original_tokenizer=FacebookAI/roberta-base \
        --vocab_size=50000
    ```
- vi-bpe-culturax-2048 (BPE tokenizer trained on CulturaX examples with <= 2048 bytes): https://huggingface.co/phucdev/vi-bpe-culturax-2048
  ```bash
    python src/training/train_tokenizer.py \
        --dataset_name=uonlp/CulturaX \
        --dataset_config_name=vi \
        --output_dir=tokenizers/vi-bpe-culturax-2048 \
        --original_tokenizer=FacebookAI/roberta-base \
        --vocab_size=50000 \
        --max_num_bytes=2048
    ```
- vi-spm-culturax-2048 (SentencePiece + Unigram LM tokenizer trained on CulturaX examples with <= 2048 bytes): https://huggingface.co/phucdev/vi-spm-culturax-2048
  ```bash
    python src/training/train_tokenizer.py \
        --dataset_name=uonlp/CulturaX \
        --dataset_config_name=vi \
        --output_dir=tokenizers/vi-spm-culturax-2048 \
        --original_tokenizer=FacebookAI/xlm-roberta-base \
        --vocab_size=50000 \
        --max_num_bytes=2048
    ```

## Embedding Initialization/Tokenizer Transfer

The script: src/mf_initialization.py
transfers word embeddings from a source model to the target language by using the CLP-Transfer method, but combines it
with matrix factorization. The main idea is to decompose the original word embeddings matrix into two matrices, 
a lower dimensional word embedding matrix and an up-projection matrix that encodes general linguistic information.
The new lower dimensional word embedding matrix is initialized with the CLP-Transfer method and is then up-projected
to the original dimensionality. The up-projection matrix is reused, which means that we leverage information from the
entire source language word embedding matrix, not just from the overlapping words.

The script: src/initialization_with_bilingual_dictionary_frequency.py
We walk through the dictionary and tokenize the source language word with the source language tokenizer and the target 
language word with the target language tokenizer. We track how often a source language word occurs with a target 
language word in translation pairs. We use this normalized frequency matrix to determine how much a source language
word embeddings contributes to the initialization of a target language word embedding. 
the token text (source and target) and using our token to indices list. That information is already present in the TokenClass. 

## Continued Pre-training
TODO

## Training and Evaluation on downstream tasks
TODO
