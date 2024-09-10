import json
from pathlib import Path

import torch
import logging
import math
import numpy as np
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    BertTokenizerFast,
    RobertaTokenizerFast,
    XLMRobertaTokenizerFast
)
from .overlap_utils import get_overlapping_tokens
from .special_token_mappings import get_bert_special_tokens, roberta_special_tokens, xlm_roberta_special_tokens

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class TokenizerTransfer:
    def __init__(
            self,
            source_model_name_or_path: str,
            target_tokenizer_name_or_path: str,
            target_model_path: str = None,
            **kwargs
    ):
        """
        Class for transferring embeddings from one tokenizer to another.
        :param source_model_name_or_path: Name or path of the source model
        :param target_tokenizer_name_or_path: Name or path of the target tokenizer
        :param target_model_path: Path to save the transferred model
        """
        self.source_model_name_or_path = source_model_name_or_path
        self.target_tokenizer_name = target_tokenizer_name_or_path
        self.target_model_path = target_model_path

        self.source_tokenizer = AutoTokenizer.from_pretrained(source_model_name_or_path)
        if "bert" in self.source_model_name_or_path:
            self.source_model = AutoModelForMaskedLM.from_pretrained(source_model_name_or_path)
        else:
            self.source_model = AutoModelForCausalLM.from_pretrained(source_model_name_or_path)
        self.source_embeddings = self.source_model.get_input_embeddings().weight.detach().numpy()
        self.source_token_to_idx = {t: i for t, i in self.source_tokenizer.get_vocab().items()}
        self.target_tokenizer = AutoTokenizer.from_pretrained(target_tokenizer_name_or_path)
        self.target_tokens = self.target_tokenizer.get_vocab()
        self.target_token_to_idx = {t: i for t, i in self.target_tokenizer.get_vocab().items()}

        # Information about the transfer
        self.overlap_based_initialized_tokens = 0
        self.cleverly_initialized_tokens = 0

    def save_parameters_to_dict(self):
        parameters_dict = {
            "source_model_name_or_path": self.source_model_name_or_path,
            "target_tokenizer_name_or_path": self.target_tokenizer_name,
            "target_model_path": self.target_model_path,
            "source_tokenizer_vocab_size": len(self.source_tokenizer.get_vocab()),
            "target_tokenizer_vocab_size": len(self.target_tokenizer.get_vocab()),
            "source_embeddings_shape": self.source_embeddings.shape,
            "cleverly_initialized_tokens": self.cleverly_initialized_tokens,
            "overlap_based_initialized_tokens": self.overlap_based_initialized_tokens
        }
        return parameters_dict

    def initialize_embeddings(self, **kwargs):
        """
        Method that initializes the embeddings of a LM with a target tokenizer given a source LM.

        :param kwargs: no kwargs

        :return: The initialized embedding matrix
        """
        raise NotImplementedError


    def update_model_embeddings(self, in_matrix, **kwargs):
        """
        Method that replaces the embeddings of a given LM with in_matrix.

        :param in_matrix: (2-d np.ndarray) The new embedding matrix.
        :param kwargs: no kwargs

        :return: A new LM model with replaced embeddings
        """

        # Change the model's embedding matrix
        target_model = self.source_model
        target_model.resize_token_embeddings(len(self.target_tokenizer))
        target_model.get_input_embeddings().weight.data = torch.from_numpy(in_matrix)

        tie_weights = kwargs.get('tie_weights', False)
        if tie_weights:
            # Tie the model's weights
            target_model.tie_weights()

        return target_model

    def transfer(self, **kwargs):
        """
        Method that creates a new LM model with transferred embeddings.
        :param kwargs: no kwargs

        :return: A new in_domain model
        """

        in_matrix = self.initialize_embeddings(**kwargs)
        target_model = self.update_model_embeddings(in_matrix, **kwargs)

        if self.target_model_path:
            self.target_tokenizer.save_pretrained(self.target_model_path)
            target_model.save_pretrained(self.target_model_path)
            with open(Path(self.target_model_path) / "transfer_information.json", "w") as f:
                information_dict = self.save_parameters_to_dict()
                json.dump(information_dict, f, indent=2)
        return target_model


class RandomInitializationTokenizerTransfer(TokenizerTransfer):
    def __init__(
            self,
            source_model_name_or_path: str,
            target_tokenizer_name_or_path: str,
            target_model_path: str = None,
            init_method: str = "smart",
            **kwargs
    ):
        """
        Class for transferring embeddings from one tokenizer to another using random initialization.
        :param source_model_name_or_path:
        :param target_tokenizer_name_or_path: Name or path of the target tokenizer
        :param target_model_path: Path to save the transferred model
        """
        super().__init__(source_model_name_or_path, target_tokenizer_name_or_path, target_model_path, **kwargs)
        self.init_method = init_method

    def save_parameters_to_dict(self):
        parameters_dict = super().save_parameters_to_dict()
        parameters_dict["init_method"] = self.init_method
        return parameters_dict

    @staticmethod
    def xavier_normal(tensor):
        """Fills the input Tensor with values according to the method described in Understanding the difficulty of
        training deep feedforward neural networks - Glorot, X. & Bengio, Y. (2010), using a normal distribution.
        https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py#L122"""
        return torch.nn.init.xavier_normal_(tensor)

    @staticmethod
    def small_init(tensor, dim):
        """Fills the input Tensor with values according to the method described in Transformers without Tears: Improving
        the Normalization of Self-Attention - Nguyen, T. & Salazar, J. (2010), using a normal distribution.
        https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py#L138"""
        # dim is hidden size: in our case it is 1024 for pythia-410m
        std = math.sqrt(2 / (5 * dim))
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    def initialize_random_embeddings(self):
        if self.init_method == "smart":
            target_embeddings = np.random.normal(
                np.mean(self.source_embeddings, axis=0),
                np.std(self.source_embeddings, axis=0),
                (
                    len(self.target_tokens),
                    self.source_embeddings.shape[1]
                )
            )
        elif self.init_method == "normal":
            # Optional: RAMEN uses a mean=0 and std=emb_dim ** -0.5
            target_embeddings = np.random.normal(
                size=(len(self.target_tokens),
                self.source_embeddings.shape[1])
            )
        elif self.init_method == "xavier":
            target_embeddings = self.xavier_normal(
                torch.empty(len(self.target_tokens),
                self.source_embeddings.shape[1])
            ).numpy()
        elif self.init_method == "small_init":
            target_embeddings = self.small_init(
                torch.empty(len(self.target_tokens),
                self.source_embeddings.shape[1]),
                self.source_embeddings.shape[1]
            ).numpy()
        else:
            raise ValueError("Invalid initialization method")
        return target_embeddings

    def initialize_embeddings(self, **kwargs):
        """
        Method that randomly initializes the embeddings of a LM with a target tokenizer given a source LM.

        :param kwargs: no kwargs

        :return: The initialized embedding matrix
        """
        target_embeddings = self.initialize_random_embeddings()
        return target_embeddings


class OverlapTokenizerTransfer(RandomInitializationTokenizerTransfer):
    def __init__(
            self,
            source_model_name_or_path: str,
            target_tokenizer_name_or_path: str,
            target_model_path: str = None,
            exact_match_all: bool = True,
            match_symbols: bool = False,
            fuzzy_match_all: bool = False,
            **kwargs
    ):
        """
        Class for transferring embeddings from one tokenizer to another using random initialization.
        :param source_model_name_or_path:
        :param target_tokenizer_name_or_path:
        :param target_model_path:
        :param exact_match_all: Match all overlapping tokens if they are an exact match. Defaults to True.
        :param match_symbols: Match overlapping symbolic tokens. Defaults to False.
        :param fuzzy_match_all: Match all overlapping tokens with fuzzy matching (whitespace and case). Defaults to False.
        """
        super().__init__(source_model_name_or_path, target_tokenizer_name_or_path, target_model_path, **kwargs)
        self.exact_match_all = exact_match_all
        self.match_symbols = match_symbols
        self.fuzzy_match_all = fuzzy_match_all

        self.overlapping_tokens = None
        self.missing_tokens = None
        self.fasttext_model = None

    def save_parameters_to_dict(self):
        parameters_dict = super().save_parameters_to_dict()
        parameters_dict["exact_match_all"] = self.exact_match_all
        parameters_dict["match_symbols"] = self.match_symbols
        parameters_dict["fuzzy_match_all"] = self.fuzzy_match_all
        return parameters_dict

    def is_very_rare_token(self, token):
        """
        We want to filter out some "bad" tokens.
        These are tokens that are so rare that they did not get an embedding in the fasttext model.
        If using pretrained word embeddings, these are tokens where no subwords are part of the pretrained word fasttext model.
        These tokens will be initialized with a random embedding.
        """
        if self.fasttext_model is not None:
            return token not in self.fasttext_model or np.absolute(self.fasttext_model[token]).sum() == 0
        else:
            return False

    def get_overlapping_tokens(self):
        # Identify overlapping tokens between the source and target vocabularies
        overlapping_tokens, missing_tokens = get_overlapping_tokens(
            target_tokenizer=self.target_tokenizer,
            source_tokenizer=self.source_tokenizer,
            exact_match_all=self.exact_match_all,
            match_symbols=self.match_symbols,
            fuzzy_match_all=self.fuzzy_match_all
        )
        # Sort to ensure same order every time (especially important when executing on multiple ranks)
        overlapping_tokens = sorted(overlapping_tokens.items(), key=lambda x: x[1].target.id)
        missing_tokens = sorted(missing_tokens.items(), key=lambda x: x[1].target.id)
        logger.debug(f"Found {len(overlapping_tokens)} overlapping tokens.")
        return overlapping_tokens, missing_tokens

    def copy_overlapping_tokens(self, target_embeddings, return_overlapping_token_indices=True):
        """
        Method that copies source embeddings for overlapping tokens to the target embeddings.
        :param target_embeddings:
        :param return_overlapping_token_indices:
        :return:
        """
        logger.info(f'{len(self.overlapping_tokens)=}; {len(self.missing_tokens)=}')

        overlapping_token_indices = []
        if not self.overlapping_tokens:
            raise ValueError('No overlapping tokens found')
        # Set overlapping tokens
        for token, overlapping_token_info in tqdm(self.overlapping_tokens,
                                                  desc="Initialize target embeddings for overlapping tokens"):
            if self.fasttext_model is not None:
                if self.is_very_rare_token(token):
                    overlapping_token_info.use_for_focus = False
                    continue
                else:
                    overlapping_token_info.auxiliary_embedding = self.fasttext_model[token]
            target_token_idx = overlapping_token_info.target.id
            source_token_idx = overlapping_token_info.source[0].id
            target_embeddings[target_token_idx] = self.source_embeddings[source_token_idx]
            if target_token_idx not in overlapping_token_indices:
                overlapping_token_indices.append(target_token_idx)

        if return_overlapping_token_indices:
            return target_embeddings, overlapping_token_indices
        else:
            return target_embeddings

    def copy_special_tokens(self, target_embeddings, return_overlapping_token_indices=True):
        overlapping_token_indices = []
        if isinstance(self.source_tokenizer, BertTokenizerFast):
            source_tokenizer_special_tokens = get_bert_special_tokens()
        elif isinstance(self.source_tokenizer, RobertaTokenizerFast):
            source_tokenizer_special_tokens = roberta_special_tokens()
        elif isinstance(self.source_tokenizer, XLMRobertaTokenizerFast):
            source_tokenizer_special_tokens = xlm_roberta_special_tokens()
        else:
            source_tokenizer_special_tokens = None
        if isinstance(self.target_tokenizer, BertTokenizerFast):
            target_tokenizer_special_tokens = get_bert_special_tokens()
        elif isinstance(self.target_tokenizer, RobertaTokenizerFast):
            target_tokenizer_special_tokens = roberta_special_tokens()
        elif isinstance(self.target_tokenizer, XLMRobertaTokenizerFast):
            target_tokenizer_special_tokens = xlm_roberta_special_tokens()
        else:
            target_tokenizer_special_tokens = None
        copied_special_tokens = []
        if source_tokenizer_special_tokens and target_tokenizer_special_tokens:
            for target_k, target_v in target_tokenizer_special_tokens.items():
                if target_k in source_tokenizer_special_tokens:
                    source_v = source_tokenizer_special_tokens[target_k]
                    target_token_idx = self.target_token_to_idx[target_v]
                    target_embeddings[self.target_token_to_idx[target_v]] = self.source_embeddings[
                        self.source_token_to_idx[source_v]]
                    copied_special_tokens.append(
                        (target_v, source_v)
                    )
                    if target_token_idx not in overlapping_token_indices:
                        overlapping_token_indices.append(target_token_idx)
            logger.info(f'Copied embeddings for special tokens: {copied_special_tokens}')

        if return_overlapping_token_indices:
            return target_embeddings, overlapping_token_indices
        else:
            return target_embeddings

    def initialize_embeddings(self, **kwargs):
        """
        Method that initializes the embeddings of a LM with a target tokenizer given a source LM.
        Leverages overlap between the source vocabulary and the target vocabulary to directly copy source embeddings
        and randomly initializes the rest.

        :param kwargs: no kwargs

        :return: The initialized embedding matrix
        """
        target_embeddings = self.initialize_random_embeddings()

        # Get overlapping tokens and missing tokens
        self.overlapping_tokens, self.missing_tokens = self.get_overlapping_tokens()
        # Copy source embeddings for overlapping tokens
        target_embeddings, overlapping_token_indices = self.copy_overlapping_tokens(target_embeddings)
        target_embeddings, overlapping_special_token_indices = self.copy_special_tokens(target_embeddings)
        for special_token_idx in overlapping_special_token_indices:
            if special_token_idx not in overlapping_token_indices:
                overlapping_token_indices.append(special_token_idx)

        logger.info(f"Copied source embeddings for {len(overlapping_token_indices)}/{len(self.target_tokens)} "
                    f"target tokens by leveraging the overlap between the source and the target vocabularies")
        self.cleverly_initialized_tokens = len(overlapping_token_indices)

        return target_embeddings
