"""
Code adapted from https://github.com/konstantinjdobler/focus/blob/main/src/deepfocus/vocab_helper.py
and https://github.com/konstantinjdobler/focus/blob/main/src/deepfocus/focus.py
From the paper:
@inproceedings{dobler-de-melo-2023-focus,
    title = "{FOCUS}: Effective Embedding Initialization for Monolingual Specialization of Multilingual Models",
    author = "Dobler, Konstantin  and
      de Melo, Gerard",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.829",
    doi = "10.18653/v1/2023.emnlp-main.829",
    pages = "13440--13454",
}
"""

import string
import logging
from dataclasses import dataclass
from typing import Union

import numpy as np
import regex
from torch import Tensor
from tqdm import tqdm
from transformers import PreTrainedTokenizer


logger = logging.getLogger(__name__)

BPE_WHITESPACE = "Ġ"
XLMR_WHITESPACE = "▁"


def get_token_standardization_func(input_tokenizer: PreTrainedTokenizer):
    """Standardize tokens from different tokenizers.
    Standard output format should be Unicode-like output for non-ASCII chars.
    Beginning of word tokens should be prefixed with a space.

    We have to use .decode() to get "standardized" tokens (e.g. BytePairBPE represents non-ASCII tokens non-UNIcode-like internally).
    But XLM-R's tokenizer removes leading whitespace from tokens when using .decode().
    Se we add those back in manually.
    """

    def decode(tokenizer: PreTrainedTokenizer, token_id: int):
        """For BPE tokenizer and fallback"""
        return tokenizer.decode(token_id)

    def replace_space(tokenizer: PreTrainedTokenizer, token_id: int):
        """For XLM-R tokenizer (sentencepiece-style)"""
        return tokenizer.convert_ids_to_tokens(token_id).replace(XLMR_WHITESPACE, " ")

    def wordpiece(tokenizer: PreTrainedTokenizer, token_id: int):
        """For wordpiece (e.g. BERT or mBERT)"""
        token = tokenizer.decode(token_id)
        if token.startswith("##"):
            return token[2:]
        else:
            return " " + token

    # simple heuristics to avoid false positive
    if (
        len([k for k in input_tokenizer.get_vocab().keys() if k[0] == XLMR_WHITESPACE])
        > 100
    ):
        standardize_token = replace_space
    # simple heuristics to avoid false positive
    elif len([k for k in input_tokenizer.get_vocab().keys() if k[:2] == "##"]) > 100:
        standardize_token = wordpiece
    else:
        standardize_token = decode

    return standardize_token


@dataclass
class TokenClass:
    native_form: str
    id: int
    canonical_form: str
    fuzzy_form: str
    uncased_form: str
    no_whitespace_form: str
    is_beginning_of_word: bool
    descriptor: str = ""


@dataclass
class NewToken:
    target: TokenClass
    auxiliary_embedding: Union[Tensor, np.ndarray] = None
    descriptor: str = ""


@dataclass
class OverlappingToken:
    source: list[TokenClass]
    target: TokenClass
    source_embedding: Tensor = None
    auxiliary_embedding: Union[Tensor, np.ndarray] = None
    descriptor: str = ""
    use_for_focus: bool = True


def get_canonicalize_token_fn(input_tokenizer: PreTrainedTokenizer):
    """Standardize tokens from different tokenizers."""

    def decode(tokenizer: PreTrainedTokenizer, token_id: int):
        """For BPE tokenizer and fallback"""
        decoded_token = tokenizer.decode(token_id)
        token = tokenizer.convert_ids_to_tokens(token_id)
        is_beginning_of_word = token.startswith(BPE_WHITESPACE)
        if is_beginning_of_word:
            return XLMR_WHITESPACE + decoded_token.lstrip(), True
        else:
            return decoded_token.lstrip(), False

    def replace_space(tokenizer: PreTrainedTokenizer, token_id: int):
        """For XLM-R tokenizer (sentencepiece-style)"""
        decoded_token = tokenizer.decode(token_id)
        token = tokenizer.convert_ids_to_tokens(token_id)

        # For sentencepiece ByteFallback tokens used in Llama, Mistral et al.
        if regex.match(r"<0x[0-9,A-F]{2}>", token):
            return token, False

        is_beginning_of_word = token.startswith(XLMR_WHITESPACE)
        if is_beginning_of_word:
            return XLMR_WHITESPACE + decoded_token.lstrip(), True
        else:
            return decoded_token.lstrip(), False

    def wordpiece(tokenizer: PreTrainedTokenizer, token_id: int):
        """For wordpiece (e.g. BERT or mBERT)"""
        token = tokenizer.decode(token_id)
        if token.startswith("##"):
            return token[2:], False
        else:
            return XLMR_WHITESPACE + token, True

    # simple heuristics to avoid false positive
    if len([k for k in input_tokenizer.get_vocab().keys() if k[0] == XLMR_WHITESPACE]) > 100:
        logger.debug(f"Using sentencepiece canonicalization for {input_tokenizer}")
        return replace_space
    elif len([k for k in input_tokenizer.get_vocab().keys() if k[:2] == "##"]) > 100:
        logger.debug(f"Using wordpiece canonicalization for {input_tokenizer}")
        return wordpiece
    else:
        logger.debug(f"Using default canonicalization for {input_tokenizer}")
        return decode


def is_numerical_symbol_etc(token: str, tokenizer: PreTrainedTokenizer):
    if token in tokenizer.all_special_tokens:
        return True
    return token.isnumeric() or all(c in string.punctuation for c in token) or token.isspace()


def canonicalize_vocab(vocab, tokenizer, descriptor):
    canonical_vocab: dict[str, TokenClass] = {}
    canonicalize_token = get_canonicalize_token_fn(tokenizer)
    for token, token_idx in tqdm(vocab.items(), desc=f"Canonicalizing {descriptor} vocab", leave=False):
        canonical_form, is_beginning_of_word = canonicalize_token(tokenizer, token_idx)
        token_info = TokenClass(
            native_form=token,
            canonical_form=canonical_form,
            fuzzy_form=canonical_form.replace(XLMR_WHITESPACE, "").lower(),
            uncased_form=canonical_form.lower(),
            no_whitespace_form=canonical_form.replace(XLMR_WHITESPACE, ""),
            id=token_idx,
            is_beginning_of_word=is_beginning_of_word,
            descriptor=descriptor,
        )

        canonical_vocab[token] = token_info
    return canonical_vocab


def construct_vocab_view(vocab: dict[str, TokenClass], key: str):
    view: dict[str, list[TokenClass]] = {}

    # sort to ensure deterministic order.
    for token, token_info in sorted(vocab.items(), key=lambda x: x[1].id):
        cur_key_value = token_info.__getattribute__(key)
        if view.get(cur_key_value):
            if cur_key_value == token_info.__getattribute__("canonical_form"):
                # ensure canonical form is always first
                view[cur_key_value].insert(0, token_info)
            else:
                view[cur_key_value].append(token_info)
        else:
            view[cur_key_value] = [token_info]
    return view


def get_overlapping_tokens(
    target_tokenizer: PreTrainedTokenizer,
    source_tokenizer: PreTrainedTokenizer,
    match_symbols: bool = True,
    exact_match_all: bool = False,
    fuzzy_match_all: bool = False,
):
    """Returns overlapping tokens between two tokenizers. There are several options to select which tokens count as overlapping tokens.

    Args:
        target_tokenizer (PreTrainedTokenizer): The target tokenizer.
        source_tokenizer (PreTrainedTokenizer): The source tokenizer.
        match_symbols (bool): Tokens that satisfy `token.isnumeric() or all(c in string.punctuation for c in token) or token.isspace()` are considered.
        exact_match_all (bool): All tokens that match exactly are considered.
        fuzzy_match_all (bool): All tokens that match ignoring whitespace and case are considered.

    Returns:
        `(dict[str, OverlappingToken], dict[str, NewToken])`: A tuple with (1) information about overlapping tokens and (2) additional tokens in the target tokenizer.
    """
    target_vocab = target_tokenizer.get_vocab()
    source_vocab = source_tokenizer.get_vocab()

    canonical_source_vocab = canonicalize_vocab(source_vocab, source_tokenizer, "source")
    canonical_target_vocab = canonicalize_vocab(target_vocab, target_tokenizer, "target")

    overlap: dict[str, OverlappingToken] = {}
    additional_tokens: dict[str, NewToken] = {}
    exact_src_vocab = construct_vocab_view(canonical_source_vocab, "canonical_form")
    fuzzy_src_vocab = construct_vocab_view(canonical_source_vocab, "fuzzy_form")

    for _, target_token_info in tqdm(
        canonical_target_vocab.items(),
        desc="Getting overlapping tokens...",
        leave=False,
    ):
        # Exact match for symbols
        if (
            match_symbols
            and is_numerical_symbol_etc(target_token_info.fuzzy_form, target_tokenizer)
            and (exact_src_vocab.get(target_token_info.canonical_form) or fuzzy_src_vocab.get(target_token_info.fuzzy_form))
        ):
            overlap[target_token_info.native_form] = OverlappingToken(
                target=target_token_info,
                source=(
                    exact_src_vocab.get(target_token_info.canonical_form) or fuzzy_src_vocab.get(target_token_info.fuzzy_form)
                ),
                descriptor="numerical_symbol",
            )
        # General exact match
        elif exact_match_all and exact_src_vocab.get(target_token_info.canonical_form):
            overlap[target_token_info.native_form] = OverlappingToken(
                target=target_token_info,
                source=exact_src_vocab[target_token_info.canonical_form],
                descriptor="exact_match",
            )
        # General fuzzy match
        elif fuzzy_match_all and fuzzy_src_vocab.get(target_token_info.fuzzy_form):
            overlap[target_token_info.native_form] = OverlappingToken(
                target=target_token_info,
                source=fuzzy_src_vocab[target_token_info.fuzzy_form],
                descriptor="fuzzy_match",
            )
        # No match - it's a NewToken
        else:
            additional_tokens[target_token_info.native_form] = NewToken(target=target_token_info)
    return overlap, additional_tokens
