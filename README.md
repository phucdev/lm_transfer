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
    pip install -e .
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
python lm_transfer/data_utils/get_train_data.py \
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
    python lm_transfer/training/train_tokenizer.py \
        --dataset_name=data/culturax_vi/train.json \
        --output_dir=tokenizers/vi-bpe-culturax-2048 \
        --original_tokenizer=FacebookAI/roberta-base \
        --vocab_size=50000 \
        --max_num_bytes=2048
    ```
- vi-spm-culturax-4g-sample (SentencePiece + Unigram LM tokenizer trained on 4GB sample of CulturaX): https://huggingface.co/phucdev/vi-spm-culturax-4g-sample
  ```bash
    python lm_transfer/training/train_tokenizer.py \
        --dataset_name=uonlp/CulturaX \
        --dataset_config_name=vi \
        --output_dir=tokenizers/vi-bpe-culturax-4g-sample \
        --original_tokenizer=FacebookAI/roberta-base \
        --vocab_size=50000
    ```
- vi-bpe-culturax-2048 (BPE tokenizer trained on CulturaX examples with <= 2048 bytes): https://huggingface.co/phucdev/vi-bpe-culturax-2048
  ```bash
    python lm_transfer/training/train_tokenizer.py \
        --dataset_name=uonlp/CulturaX \
        --dataset_config_name=vi \
        --output_dir=tokenizers/vi-bpe-culturax-2048 \
        --original_tokenizer=FacebookAI/roberta-base \
        --vocab_size=50000 \
        --max_num_bytes=2048
    ```
- vi-spm-culturax-2048 (SentencePiece + Unigram LM tokenizer trained on CulturaX examples with <= 2048 bytes): https://huggingface.co/phucdev/vi-spm-culturax-2048
  ```bash
    python lm_transfer/training/train_tokenizer.py \
        --dataset_name=uonlp/CulturaX \
        --dataset_config_name=vi \
        --output_dir=tokenizers/vi-spm-culturax-2048 \
        --original_tokenizer=FacebookAI/xlm-roberta-base \
        --vocab_size=50000 \
        --max_num_bytes=2048
    ```

## Embedding Initialization/Tokenizer Transfer

### RAMEN (Tran, 2020)
lm_transfer/embedding_initialization/ramen_transfer.py

This approach leverages parallel data to transfer word embeddings from a source model to the target language.
The parallel data is tokenized with the source and target tokenizer and `fast_align` is used to align the tokens.
The alignment is used to calculate word translation probabilities and the embeddings are then initialized based on the
translation probabilities.
For overlapping special tokens the embeddings are copied directly from the source model.
In contrast to some of the other approaches they explicitly set the output embeddings to the input embeddings.
They also calculate the output bias of the language model based on the output bias of the source model weighted by the
translation probabilities.
The output bias is only relevant in older models such as BERT. For the purpose of this thesis we ignore the output bias
because the output bias in RoBERTa-base and XLM-RoBERTa-base are zero vectors.
Another thing that sets this approach apart is that for tokens that are not in the parallel data they initialize them
randomly with a normal distribution that does NOT leverage information from the source embeddings with
`nn.init.normal_(tgt_embs, mean=0, std=emb_dim ** -0.5)`

Later approaches use the mean and the standard deviation of the source embeddings for the random initialization of the
target embeddings.

### WECHSEL (Minixhofer et al., 2022)
lm_transfer/embedding_initialization/wechsel_transfer.py

This approach leverages bilingual dictionaries to align existing FastText embeddings for the source and target language 
using the Orthogonal Procrustes method. This makes it possible to embed the tokens of the source and target language in
the same auxiliary embedding space. For each target token its embedding is initialized as the weighted average of the
source embeddings of its `n` nearest source token neighbors in the auxiliary embedding space.
They also copy embeddings for overlapping special tokens directly from the source model.

We added to other approaches that are based on WECHSEL.
- WECHSEL+overlap: This approach follows WECHSEL, but it additionally copies the embeddings of overlapping tokens
  directly from the source model. 
- WECHSEL+aligned: This approach follows WECHSEL, but uses already aligned embeddings and uses the method described in 
  appendix D (WECHSEL+TFR) to calculate subword embeddings without subword information based on word frequencies. 
  They apply the tokenizer to every word resulting in a set of subwords for each word. The embedding of the subword is 
  the average of the embeddings of words whose tokenization contains the subword, weighted by their word frequencies.

### FVT (Gee et al., 2022)
lm_transfer/embedding_initialization/fvt_transfer.py

This approach only works for the transfer of multilingual source models.
For overlapping tokens the embeddings are copied directly from the source model.
For non-overlapping tokens in the target vocabulary each token is tokenized using the source tokenizer and the 
source embeddings of the subwords are averaged to initialize the target token embeddings.

We think that there is potential to improve the initialization of non-overlapping tokens by using different 
aggregation methods compared to simply averaging the embeddings of the subwords of the decomposition.

### CLP-Transfer (Ostendorff & Rehm, 2023)
lm_transfer/embedding_initialization/clp_transfer.py

This method transfers word embeddings from a source model to the target language by using the CLP-Transfer method.
The approach leverages overlapping words between the source and target language to initialize the target language word 
embeddings. For overlapping words the source embeddings are used directly, for non-overlapping words the embeddings are
initialized based on the weighted average of the embeddings of the overlapping tokens. 
They use a helper model in the target language to determine the weights for the weighted average.
The weights are calculated as the cosine similarity between the target language word and the words in the overlap 
in the helper model embedding space.

Notes:
- It is assumed that the helper model already uses the target tokenizer. 
- The original approach assumes that the input and output embeddings are tied. In order to support more recent models 
  with separate input and output embeddings, we simply repeat the transfer process for the output embeddings.

### FOCUS (Dobler & de Melo, 2023)
lm_transfer/embedding_initialization/focus_transfer.py

This method transfers word embeddings from a source model to the target language by using the FOCUS method.
It also leverages overlapping words between the source and target language to initialize the target language word
embeddings. For overlapping words the source embeddings are used directly, for non-overlapping words the embeddings are
also initialized based on the weighted average of the embeddings of the overlapping tokens.
Instead of an existing helper model, they train FastText embeddings on the target language where they pre-tokenize the 
text using the target tokenizer and use these embeddings as the auxiliary embedding space to determine the weights for
the weighted average.

Notes:
- In order to find overlapping tokens this approach performs a canonicalization step in order to match the tokens 
  between the source and target language even for different tokenization techniques. This is sets this approach apart 
  from the other approaches that directly compare the source and target vocabularies. 
  It also has an option for fuzzy matching.
- While not mentioned in the paper, we strongly recommend training the auxiliary FastText model on the same data that 
  the target tokenizer was trained on to minimize the amount of randomly initialized embeddings.
- The original approach assumes that the input and output embeddings are tied. We had modify the code in order to 
  support more recent models with separate input and output embeddings.

### ZeTT (Minixhofer et al., 2024)
For this approach you need to clone the original repository and run the transfer script.
(Integrating the approach into this repository would be too complicated.)

1. Clone the repository from: https://github.com/bminixhofer/zett
    ```
    git clone https://github.com/bminixhofer/zett.git
    ```
2. Follow the [instructions in the README.md of the repository](https://github.com/bminixhofer/zett?tab=readme-ov-file#using-a-pretrained-hypernetwork) to create a conda environment and install the requirements
    ```
    conda create -n zett Python=3.11
    conda activate zett
    
    pip install -r requirements.txt
    pip install -U "jax[cuda12_pip]==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html # adjust based on your CUDA version
    pip install -e .
    ```
3. Transfer the model, e.g. for Vietnamese:
    ```
    git clone https://huggingface.co/benjamin/zett-hypernetwork-xlm-roberta-base
    
    python3 scripts/transfer.py \
        --target_model=FacebookAI/xlm-roberta-base \
        --tokenizer_name=phucdev/vi-spm-culturax-4g-sample \
        --output=xlm-r \
        --model_class=AutoModelForMaskedLM \
        --lang_code=vi \
        --checkpoint_path=zett-hypernetwork-xlm-roberta-base \
        --save_pt # otherwise saves only Flax weights
    ```
This method is quite different from the previous methods.
The approach performs zero-shot transfer of tokenizers by training a hypernetwork that predicts new embeddings for any
given tokenizer. The hypernetwork learns to merge embeddings of tokens decomposed by the original tokenizer into a 
single embedding for tokens of the target tokenizer.

### Matrix Factorization Transfer
lm_transfer/embedding_initialization/matrix_factorization_transfer.py

This approach transfers word embeddings from a source model to the target language by using any of the existing 
transfer methods, but combine it with matrix factorization. 
The main idea is to decompose the original word embeddings matrix into two matrices, a lower dimensional word embedding 
matrix and an up-projection matrix that encodes general linguistic information.
The new lower dimensional word embedding matrix is initialized with the respective method and is then up-projected
to the original dimensionality. The up-projection matrix is reused, which means that we leverage information from the
entire source language word embedding matrix, not just from the overlapping words.

### Bilingual Dictionary Transfer
lm_transfer/embedding_initialization/bilingual_dictionary_transfer.py

We walk through the dictionary and tokenize the source language word with the source language tokenizer and the target 
language word with the target language tokenizer. We track how often a source language word occurs with a target 
language word in translation pairs. We use this normalized frequency matrix to determine how much a source language
word embeddings contributes to the initialization of a target language word embedding. 



## Continued Pre-training
TODO

## Training and Evaluation on downstream tasks
TODO
