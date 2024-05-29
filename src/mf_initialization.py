import sys
import logging
import os
import math
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

import fire
from tqdm.auto import tqdm
from utils import perform_factorize
from snmf import SNMF
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
        random_init: bool = False,
        random_init_method: str = None,
        keep_dim: int = 100,
        mf_method: str = None,
        exact_match_all: bool = True,
        match_symbols: bool = False,
        fuzzy_match_all: bool = False
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

    # Factorize source embeddings into lower dimensional word embeddings F with token specific information and
    # orthogonal up-projection matrix G that encodes general linguistic information and is shared by all tokens
    logger.info(f'{source_embeddings.shape=}')
    if mf_method == "svd":
        # SVD method as in OFA by Liu et al. (2024)
        # f is described as lower embeddings or coordinates
        # g is described as primitive embeddings
        logger.info(f"Performing SVD factorization as in OFA by Liu et al. (2024)")
        g, f = perform_factorize(source_embeddings, keep_dim=keep_dim)
        logger.info(f'{f.shape=}; {g.shape=}')
    elif mf_method == "snmf":
        # Semi Non-Negative Matrix Factorization similar to Pfeiffer et al. (2021)
        logger.info(f"Performing SNMF factorization similar to Pfeiffer et al. (2021)")
        snmf_mdl = SNMF(source_embeddings, niter=3000, num_bases=keep_dim)
        snmf_mdl.factorize()
        f = snmf_mdl.W
        g = snmf_mdl.H
        logger.info(f'{f.shape=}; {g.shape=}')
    else:
        # No factorization
        logger.info(f"No matrix factorization")
        f = source_embeddings
        g = np.eye(source_embeddings.shape[1])

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
    # source_tokens = set(source_tokenizer.get_vocab().keys())

    # Overlapping tokens
    logger.info(f'Matching for overlapping tokens: {match_symbols=}; {exact_match_all=}; {fuzzy_match_all=}')
    # overlapping tokens keys are the native form of the corresponding target token
    overlapping_tokens, missing_tokens = get_overlapping_tokens(helper_tokenizer, source_tokenizer,
                                                                match_symbols=match_symbols,
                                                                exact_match_all=exact_match_all,
                                                                fuzzy_match_all=fuzzy_match_all)
    # overlapping_tokens = sorted(overlapping_tokens.items(), key=lambda x: x[1].target.id)
    # missing_tokens = sorted(missing_tokens.items(), key=lambda x: x[1].target.id)
    missing_tokens_list = list(missing_tokens.keys())
    overlapping_tokens_list_source = []
    overlapping_tokens_list_target = list(overlapping_tokens.keys())
    for t, overlapping_token in overlapping_tokens.items():
        overlapping_tokens_list_source.append(overlapping_token.source[0].native_form)

    logger.info(f'{len(overlapping_tokens)=}; {len(missing_tokens)=}')

    if not overlapping_tokens:
        raise ValueError('No overlapping tokens found')

    source_token_to_idx = {t: i for t, i in source_tokenizer.get_vocab().items()}
    # target_token_to_idx = {t: i for t, i in target_tokenizer.get_vocab().items()}
    helper_token_to_idx = {t: i for t, i in helper_tokenizer.get_vocab().items()}

    overlapping_tokens_idxs = [source_token_to_idx[t] for t in overlapping_tokens_list_source]
    # overlapping_token_vecs = source_embeddings[overlapping_tokens_idxs, :]
    overlapping_f = f[overlapping_tokens_idxs, :]

    logger.info(f'{overlapping_f.shape=}')

    # Target embeddings
    np.random.seed(seed)
    if random_init:
        logger.info(f'Use randomly initialized target embeddings')
        # Random init target embeddings with mean+std of source embeddings
        if random_init_method == 'xavier':
            target_embeddings = xavier_normal(torch.empty(len(target_tokens), f.shape[1])).numpy()
        elif random_init_method == 'small_init':
            target_embeddings = small_init(torch.empty(len(target_tokens), f.shape[1]),
                                           f.shape[1]).numpy()
        elif random_init_method == 'normal_leverage_source':
            target_embeddings = np.random.normal(
                np.mean(f, axis=0),
                np.std(f, axis=0),
                (
                    len(target_tokens),
                    f.shape[1]
                )
            )
        else:
            target_embeddings = np.random.normal(size=(len(target_tokens), f.shape[1]))
    else:
        # Random init target embeddings with mean+std of source embeddings
        target_embeddings = np.random.normal(
            np.mean(f, axis=0),
            np.std(f, axis=0),
            (
                len(target_tokens),
                f.shape[1]
            )
        )
        # Set overlapping tokens only if we plan on using factorized embeddings as in Pfeiffer et al. (2021)
        # for source_t, target_t in zip(overlapping_tokens_list_source, overlapping_tokens_list_target):
        #   target_embeddings[helper_token_to_idx[target_t]] = source_embeddings[source_token_to_idx[source_t]]

        if missing_tokens:
            helper_missing_tokens_vecs = helper_embeddings[[helper_token_to_idx[t] for t in missing_tokens_list], :]
            helper_overlapping_token_vecs = helper_embeddings[
                                            [helper_token_to_idx[t] for t in overlapping_tokens_list_target], :]

            # Similarities for missing tokens
            sims = cosine_similarity(helper_missing_tokens_vecs, helper_overlapping_token_vecs)

            # similar = 1 => high weight
            # dissimilar = 0 => low weight

            for ti, t in enumerate(tqdm(missing_tokens_list)):
                # distances to overlapping tokens
                token_sims = sims[ti]
                norm_sims = token_sims / token_sims.sum()

                # weighted average of overlapping token embeddings with weight from similarity in helper token
                # embedding space
                target_vec = np.average(overlapping_f, axis=0, weights=norm_sims)
                target_embeddings[helper_token_to_idx[t]] = target_vec
        else:
            logger.warning('No missing tokens')

    # Multiply target embeddings with g to get full-sized embedding matrix
    target_embeddings = np.dot(target_embeddings, g)
    for source_t, target_t in zip(overlapping_tokens_list_source, overlapping_tokens_list_target):
        target_embeddings[helper_token_to_idx[target_t]] = source_embeddings[source_token_to_idx[source_t]]
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
