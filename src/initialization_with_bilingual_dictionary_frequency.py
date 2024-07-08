import sys
import logging
import os
import math
import csv
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

import fire
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
        helper_model_name_or_path,
        target_model_path,
        helper_tokenizer_name_or_path=None,
        seed=42,
        override: bool = False,
        copy_overlapping_tokens: bool = False,
        exact_match_all: bool = True,
        match_symbols: bool = False,
        fuzzy_match_all: bool = False,
        bilingual_dictionary: str = None
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

    # CLP
    if not helper_tokenizer_name_or_path:
        helper_tokenizer_name_or_path = helper_model_name_or_path

    logger.info(f'Loading helper model: {helper_model_name_or_path}')
    logger.info(f'Loading helper tokenizer: {helper_tokenizer_name_or_path}')

    if "bert" in helper_model_name_or_path:
        helper_model = AutoModelForMaskedLM.from_pretrained(helper_model_name_or_path)
    else:
        helper_model = AutoModelForCausalLM.from_pretrained(helper_model_name_or_path)
    helper_tokenizer = AutoTokenizer.from_pretrained(helper_tokenizer_name_or_path)
    helper_embeddings = helper_model.get_input_embeddings().weight.detach().numpy()

    target_tokens = set(helper_tokenizer.get_vocab().keys())
    target_tokens_list = list(helper_tokenizer.get_vocab().keys())
    source_tokens_list = list(source_tokenizer.get_vocab().keys())

    # Load bilingual dictionary
    dict_pairs = []
    with open(bilingual_dictionary) as f:
        csv_reader = csv.reader(f, delimiter='\t')
        for row in csv_reader:
            dict_pairs.append((row[0], row[1]))
    # Count co-occurrences of tokens in translation pairs
    token_freqmatrix = np.zeros((len(target_tokens_list), len(source_tokens_list)))
    for en, vi in tqdm(dict_pairs, desc="Counting co-occurrences of tokens in translation pairs"):
        en_tokens = source_tokenizer.tokenize(en)
        vi_tokens = helper_tokenizer.tokenize(vi)
        for vi_t in vi_tokens:
            for en_t in en_tokens:
                token_freqmatrix[target_tokens_list.index(vi_t)][source_tokens_list.index(en_t)] += 1
    # Adding a small number to avoid division by zero, if necessary
    column_sums = np.sum(token_freqmatrix, axis=0) + 1e-9  # adding a small constant
    normalized_matrix = token_freqmatrix / column_sums

    np.random.seed(seed)

    source_token_to_idx = {t: i for t, i in source_tokenizer.get_vocab().items()}
    helper_token_to_idx = {t: i for t, i in helper_tokenizer.get_vocab().items()}

    # Random init target embeddings with mean+std of source embeddings
    target_embeddings = np.random.normal(
        np.mean(source_embeddings, axis=0),
        np.std(source_embeddings, axis=0),
        (
            len(target_tokens),
            source_embeddings.shape[1]
        )
    )

    if copy_overlapping_tokens:
        # Overlapping tokens
        logger.info(f'Matching for overlapping tokens: {match_symbols=}; {exact_match_all=}; {fuzzy_match_all=}')
        # overlapping tokens keys are the native form of the corresponding target token
        overlapping_tokens, missing_tokens = get_overlapping_tokens(helper_tokenizer, source_tokenizer,
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
        for source_t, target_t in zip(overlapping_tokens_list_source, overlapping_tokens_list_target):
            target_embeddings[helper_token_to_idx[target_t]] = source_embeddings[source_token_to_idx[source_t]]

    for i in tqdm(range(normalized_matrix.shape[0]), desc="Applying lexicon walking"):
        # Find those whose entry is non-zero: has a translation
        relevant_source_embedding_indices = np.nonzero(normalized_matrix[i, :])[0]
        relevant_source_embeddings = source_embeddings[[t for t in relevant_source_embedding_indices], :]
        weights = normalized_matrix[i, relevant_source_embedding_indices]
        if weights.sum() == 0.0:
            continue
        target_vec = np.average(relevant_source_embeddings, axis=0,
                                weights=normalized_matrix[i, relevant_source_embedding_indices])
        target_embeddings[i] = target_vec

    logger.info(f'{target_embeddings.shape=}')

    # Save target model
    target_model = source_model
    target_tokenizer = helper_tokenizer
    target_model.resize_token_embeddings(len(target_tokenizer))
    target_model.get_input_embeddings().weight.data = torch.from_numpy(target_embeddings)

    target_model.save_pretrained(target_model_path)
    target_tokenizer.save_pretrained(target_model_path)
    logger.info(f'Saved to {target_model_path}')


if __name__ == '__main__':
    fire.Fire(apply_clp)
    sys.exit(0)
