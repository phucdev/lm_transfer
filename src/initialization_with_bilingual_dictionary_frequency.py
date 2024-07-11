import sys
import logging
import os
import math
import csv
import torch
import fire
import entmax

import numpy as np
from scipy.special import softmax
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
from tqdm.auto import tqdm
from focus import get_overlapping_tokens

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def xavier_normal(tensor):
    """Fills the input Tensor with values according to the method described in Understanding the difficulty of
    training deep feedforward neural networks - Glorot, X. & Bengio, Y. (2010), using a normal distribution.
    https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py#L122"""
    return torch.nn.init.xavier_normal_(tensor)


def small_init(tensor, dim):
    """Fills the input Tensor with values according to the method described in Transformers without Tears: Improving
    the Normalization of Self-Attention - Nguyen, T. & Salazar, J. (2010), using a normal distribution.
    https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py#L138"""
    # dim is hidden size: in our case it is 1024 for pythia-410m
    std = math.sqrt(2 / (5 * dim))
    return torch.nn.init.normal_(tensor, mean=0.0, std=std)


def apply_clp(
        source_model_name_or_path,
        target_model_path,
        target_tokenizer_name_or_path=None,
        seed=42,
        override: bool = False,
        copy_overlapping_tokens: bool = False,
        prioritize_overlap: bool = False,
        exact_match_all: bool = False,
        match_symbols: bool = False,
        fuzzy_match_all: bool = True,
        bilingual_dictionary: str = None,
        skip_phrases: bool = False,  # phrases could be noisy
):
    """
    All methods have the following steps in common:
    - Load source model and tokenizer
    - Optionally factorize source embeddings into lower dimensional word embeddings F with token specific information
        and orthogonal up-projection matrix G that encodes general linguistic information and is shared by all tokens
    - Load target tokenizer (learned or helper model)
    - Initialize target embeddings with source embeddings for overlapping tokens
    - Optionally initialize target embeddings for missing tokens with a weighted average of overlapping token embeddings
      - Similarities for missing tokens are calculated in the helper token embedding space/ aligned FastText
        embeddings that were readily available/ learned FastText embeddings for the target language
    - Initialize target model with transformer weights from source model and replace embeddings with target embeddings
    """
    if os.path.exists(target_model_path) and not override:
        raise FileExistsError(f'Output exists already at {target_model_path} fix with --override')

    logger.info(f"Loading source model: {source_model_name_or_path}")

    if "bert" in source_model_name_or_path:
        source_model = AutoModelForMaskedLM.from_pretrained(source_model_name_or_path)
    else:
        source_model = AutoModelForCausalLM.from_pretrained(source_model_name_or_path)
    source_tokenizer = AutoTokenizer.from_pretrained(source_model_name_or_path)
    source_embeddings = source_model.get_input_embeddings().weight.detach().numpy()

    logger.info(f'{source_embeddings.shape=}')

    logger.info(f'Loading helper tokenizer: {target_tokenizer_name_or_path}')

    target_tokenizer = AutoTokenizer.from_pretrained(target_tokenizer_name_or_path)

    target_tokens = set(target_tokenizer.get_vocab().keys())
    target_tokens_list = list(target_tokenizer.get_vocab().keys())
    source_tokens_list = list(source_tokenizer.get_vocab().keys())

    source_token_to_idx = {t: i for t, i in source_tokenizer.get_vocab().items()}
    source_idx_to_token = [t for t, i in source_token_to_idx.items()]
    target_token_to_idx = {t: i for t, i in target_tokenizer.get_vocab().items()}
    target_idx_to_token = [t for t, i in target_token_to_idx.items()]

    # Load bilingual dictionary
    dict_pairs = []
    with open(bilingual_dictionary) as f:
        csv_reader = csv.reader(f, delimiter='\t')
        for row in csv_reader:
            dict_pairs.append((row[0], row[1]))
    # Count co-occurrences of tokens in translation pairs
    token_freqmatrix = np.zeros((len(target_tokens_list), len(source_tokens_list)), dtype=np.float32)
    source_token_freqs = np.zeros(len(source_tokens_list))
    for en, vi in tqdm(dict_pairs, desc="Counting co-occurrences of tokens in translation pairs"):
        if skip_phrases and len(en.split()) > 3:  # heuristic to filter out phrases
            continue
        en_token_ids = source_tokenizer(en, add_special_tokens=False, return_tensors='pt')['input_ids'][0]
        vi_token_ids = target_tokenizer(vi, add_special_tokens=False, return_tensors='pt')['input_ids'][0]
        # debug
        en_tokens = source_tokenizer.convert_ids_to_tokens(en_token_ids)
        vi_tokens = target_tokenizer.convert_ids_to_tokens(vi_token_ids)
        for vi_t in vi_token_ids:
            for en_t in en_token_ids:
                token_freqmatrix[vi_t][en_t] += 1 / len(en_token_ids)  # adjust by decomposition lengths
        source_token_freqs[en_token_ids] += 1

    # Adding a small number to avoid division by zero, if necessary
    row_sums = np.sum(token_freqmatrix, axis=1).reshape(-1, 1) + 1e-9  # adding a small constant
    normalized_matrix = token_freqmatrix / row_sums  # relative frequencies
    softmax_probs = softmax(token_freqmatrix, axis=1)
    # sparsemax_probs = entmax.sparsemax(torch.tensor(token_freqmatrix), dim=1).numpy()
    # normalized_source_token_freqs = source_token_freqs / np.sum(source_token_freqs)  # all close to zero
    # adjusted_matrix = normalized_matrix / (normalized_source_token_freqs + 1e-9)  # adjusted by source token frequencies

    np.random.seed(seed)

    logger.info("Initializing target embeddings using normal distribution with mean and std of source embeddings")
    # Random init target embeddings with mean+std of source embeddings
    target_embeddings = np.random.normal(
        np.mean(source_embeddings, axis=0),
        np.std(source_embeddings, axis=0),
        (
            len(target_tokens),
            source_embeddings.shape[1]
        )
    )

    overlapping_token_indices = []
    # Copy special token embeddings
    source_tokenizer_special_tokens = source_tokenizer.all_special_tokens
    target_tokenizer_special_tokens = target_tokenizer.all_special_tokens
    bert_to_roberta = {
        '[CLS]': '<s>',
        '[SEP]': '</s>',
        '[PAD]': '<pad>',
        '[UNK]': '<unk>',  # assuming an unk token needs mapping
        '[MASK]': '<mask>'  # assuming a mask token needs mapping
    }
    if (all(t.startswith("[") for t in source_tokenizer_special_tokens) and
            all(t.startswith("<") for t in target_tokenizer_special_tokens)):
        special_tokens_mapping = bert_to_roberta
    elif (all(t.startswith("<") for t in source_tokenizer_special_tokens) and
          all(t.startswith("[") for t in target_tokenizer_special_tokens)):
        special_tokens_mapping = {v: k for k, v in bert_to_roberta.items()}
    else:
        # Identity mapping
        special_tokens_mapping = {t: t for t in source_tokenizer_special_tokens}
    for special_token in source_tokenizer.all_special_tokens:
        target_special_token = special_tokens_mapping[special_token]
        if target_special_token in target_tokens:
            special_token_idx = target_token_to_idx[target_special_token]
            target_embeddings[special_token_idx] = source_embeddings[source_token_to_idx[special_token]]
            overlapping_token_indices.append(special_token_idx)
    logger.info(f'Copied embeddings for these special tokens: '
                f'{target_tokenizer.convert_ids_to_tokens(overlapping_token_indices)}')

    if copy_overlapping_tokens:
        # Overlapping tokens
        logger.info(f'Matching for overlapping tokens: {match_symbols=}; {exact_match_all=}; {fuzzy_match_all=}')
        # overlapping tokens keys are the native form of the corresponding target token
        overlapping_tokens, missing_tokens = get_overlapping_tokens(target_tokenizer, source_tokenizer,
                                                                    match_symbols=match_symbols,
                                                                    exact_match_all=exact_match_all,
                                                                    fuzzy_match_all=fuzzy_match_all)
        overlapping_tokens_list_source = []
        overlapping_tokens_list_target = list(overlapping_tokens.keys())
        for t, overlapping_token in overlapping_tokens.items():
            overlapping_tokens_list_source.append(overlapping_token.source[0].native_form)

        logger.info(f'{len(overlapping_tokens)=}; {len(missing_tokens)=}')

        if not overlapping_tokens:
            raise ValueError('No overlapping tokens found')
        # Set overlapping tokens
        for source_t, target_t in tqdm(zip(overlapping_tokens_list_source, overlapping_tokens_list_target),
                                       desc="Initialize target embeddings for overlapping tokens"):
            overlapping_token_idx = target_token_to_idx[target_t]
            target_embeddings[overlapping_token_idx] = source_embeddings[source_token_to_idx[source_t]]
            overlapping_token_indices.append(overlapping_token_idx)

    dictionary_token_indices = []
    logger.info(f"Initializing target embeddings for missing tokens with translations "
                f"({copy_overlapping_tokens=}, {prioritize_overlap=})")
    for i in tqdm(range(normalized_matrix.shape[0]), desc="Initialize target embeddings for missing tokens with "
                                                          "translations"):
        if i in target_tokenizer.all_special_ids:
            continue
        elif copy_overlapping_tokens and prioritize_overlap and i in overlapping_token_indices:
            continue
        # Find those whose entry is non-zero: has a translation
        relevant_source_embedding_indices = np.nonzero(normalized_matrix[i, :])[0]
        relevant_source_embeddings = source_embeddings[[t for t in relevant_source_embedding_indices], :]

        norm_freqs = normalized_matrix[i, relevant_source_embedding_indices]
        norm_freqs_sum = norm_freqs.sum()
        # adjusted_weights = adjusted_matrix[i, relevant_source_embedding_indices]
        if norm_freqs_sum == 0.0:
            continue
        weights = norm_freqs
        target_vec = np.average(relevant_source_embeddings, axis=0, weights=weights)
        target_embeddings[i] = target_vec

        regular_sum = weights.sum()
        softmax_sum = softmax_probs[i, relevant_source_embedding_indices].sum()

        # debugging
        abs_freqs = token_freqmatrix[i, relevant_source_embedding_indices]
        softmaxed_relevant_source_embedding_indices = np.nonzero(softmax_probs[i, :])[0]
        softmaxed_freqs = softmax_probs[i, softmaxed_relevant_source_embedding_indices]
        softmaxed_relevant_tokens = source_tokenizer.convert_ids_to_tokens(softmaxed_relevant_source_embedding_indices)
        # sparsemaxed_freqs = sparsemax_probs[i, relevant_source_embedding_indices]
        target_token = target_tokenizer.convert_ids_to_tokens([i])[0]
        relevant_source_tokens = source_tokenizer.convert_ids_to_tokens(relevant_source_embedding_indices)
        sorted_relevant_source_tokens = [
            (w, f, t) for w, f, t in
            sorted(zip(weights, abs_freqs, relevant_source_tokens),
                   key=lambda pair: pair[0], reverse=True)
        ]
        softmaxed_sorted_relevant_source_tokens = [
            (w, t) for w, t in
            sorted(zip(softmaxed_freqs, softmaxed_relevant_tokens),
                   key=lambda pair: pair[0], reverse=True)
        ]
        # Debug
        if target_token == '▁của':
            logger.debug(f'{target_token=}; {relevant_source_tokens=}')
            logger.debug(f'{list(sorted_relevant_source_tokens)}')

        dictionary_token_indices.append(i)

    logger.info(f'Initialized {len(dictionary_token_indices)} target embeddings with translations')
    logger.info(f'Initialized {len(overlapping_token_indices)} target embeddings with overlapping tokens')
    logger.info(f'Initialized {len(dictionary_token_indices) + len(overlapping_token_indices)}/'
                f'{target_embeddings.shape[0]} target embeddings using heuristics in total')
    logger.info(f'{target_embeddings.shape=}')

    # Save target model
    target_model = source_model
    target_tokenizer = target_tokenizer
    target_model.resize_token_embeddings(len(target_tokenizer))
    target_model.get_input_embeddings().weight.data = torch.from_numpy(target_embeddings)

    target_model.save_pretrained(target_model_path)
    target_tokenizer.save_pretrained(target_model_path)
    logger.info(f'Saved to {target_model_path}')


if __name__ == '__main__':
    fire.Fire(apply_clp)
    sys.exit(0)
