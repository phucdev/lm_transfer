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
from sklearn.metrics.pairwise import cosine_similarity
from .overlap import get_overlapping_tokens
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
            target_model_path: str = None
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

    def initialize_embeddings(self, **kwargs):
        """
        Method that initializes the embeddings of a given LM with the source embeddings.

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
        return target_model


class RandomInitializationTokenizerTransfer(TokenizerTransfer):
    def __init__(
            self,
            source_model_name_or_path: str,
            target_tokenizer_name_or_path: str,
            target_model_path: str = None,
            init_method: str = "smart"
    ):
        """
        Class for transferring embeddings from one tokenizer to another using random initialization.
        :param source_model_name_or_path:
        :param target_tokenizer_name_or_path: Name or path of the target tokenizer
        :param target_model_path: Path to save the transferred model
        """
        super().__init__(source_model_name_or_path, target_tokenizer_name_or_path, target_model_path)
        self.init_method = init_method

    def initialize_embeddings(self, **kwargs):
        """
        Method that initializes the embeddings of a given LM with the source embeddings.

        :param kwargs: no kwargs

        :return: The initialized embedding matrix
        """
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


class OverlapTokenizerTransfer(RandomInitializationTokenizerTransfer):
    def __init__(
            self,
            source_model_name_or_path: str,
            target_tokenizer_name_or_path: str,
            target_model_path: str = None,
            exact_match_all: bool = True,
            match_symbols: bool = False,
            fuzzy_match_all: bool = False,
    ):
        """
        Class for transferring embeddings from one tokenizer to another using random initialization.
        :param source_model_name_or_path:
        :param target_tokenizer_name_or_path:
        :param target_model_path:
        :param exact_match_all:
        :param match_symbols:
        :param fuzzy_match_all:
        """
        super().__init__(source_model_name_or_path, target_tokenizer_name_or_path, target_model_path)
        self.exact_match_all = exact_match_all
        self.match_symbols = match_symbols
        self.fuzzy_match_all = fuzzy_match_all

        self.overlapping_tokens = None
        self.missing_tokens = None

    def copy_overlapping_tokens(
            self, target_embeddings, overlapping_tokens, missing_tokens, return_overlapping_token_indices=True
    ):
        overlapping_tokens_list_source = []
        overlapping_tokens_list_target = list(overlapping_tokens.keys())
        for t, overlapping_token in overlapping_tokens.items():
            overlapping_tokens_list_source.append(overlapping_token.source[0].native_form)

        logger.info(f'{len(overlapping_tokens)=}; {len(missing_tokens)=}')

        overlapping_token_indices = []
        if not overlapping_tokens:
            raise ValueError('No overlapping tokens found')
        # Set overlapping tokens
        for source_t, target_t in tqdm(zip(overlapping_tokens_list_source, overlapping_tokens_list_target),
                                       desc="Initialize target embeddings for overlapping tokens"):
            overlapping_token_idx = self.target_token_to_idx[target_t]
            target_embeddings[overlapping_token_idx] = self.source_embeddings[self.source_token_to_idx[source_t]]
            if overlapping_token_idx not in overlapping_token_indices:
                overlapping_token_indices.append(overlapping_token_idx)

        # Copy source embeddings for special tokens
        if isinstance(self.source_tokenizer, BertTokenizerFast):
            source_tokenizer_special_tokens = get_bert_special_tokens()
        elif isinstance(self.source_tokenizer, RobertaTokenizerFast):
            source_tokenizer_special_tokens = roberta_special_tokens
        elif isinstance(self.source_tokenizer, XLMRobertaTokenizerFast):
            source_tokenizer_special_tokens = xlm_roberta_special_tokens
        else:
            source_tokenizer_special_tokens = None
        if isinstance(self.target_tokenizer, BertTokenizerFast):
            target_tokenizer_special_tokens = get_bert_special_tokens()
        elif isinstance(self.target_tokenizer, RobertaTokenizerFast):
            target_tokenizer_special_tokens = roberta_special_tokens
        elif isinstance(self.target_tokenizer, XLMRobertaTokenizerFast):
            target_tokenizer_special_tokens = xlm_roberta_special_tokens
        else:
            target_tokenizer_special_tokens = None
        copied_special_tokens = []
        if source_tokenizer_special_tokens and target_tokenizer_special_tokens:
            for target_t in target_tokenizer_special_tokens:
                if target_t in source_tokenizer_special_tokens:
                    source_t = source_tokenizer_special_tokens[target_t]
                    overlapping_token_idx = self.target_token_to_idx[target_t]
                    target_embeddings[self.target_token_to_idx[target_t]] = self.source_embeddings[
                        self.source_token_to_idx[source_t]]
                    copied_special_tokens.append(
                        (target_tokenizer_special_tokens[target_t], source_tokenizer_special_tokens[source_t])
                    )
                    if overlapping_token_idx not in overlapping_token_indices:
                        overlapping_token_indices.append(overlapping_token_idx)
            logger.info(f'Copied embeddings for special tokens: {copied_special_tokens}')

        if return_overlapping_token_indices:
            return target_embeddings, overlapping_token_indices
        else:
            return target_embeddings

    def initialize_embeddings(self, **kwargs):
        """
        Method that initializes the embeddings of a given LM with the source embeddings.

        :param kwargs: no kwargs

        :return: The initialized embedding matrix
        """
        target_embeddings = super().initialize_embeddings(**kwargs)

        # Identify overlapping tokens between the source and target vocabularies
        self.overlapping_tokens, self.missing_tokens = get_overlapping_tokens(
            target_tokenizer=self.target_tokenizer,
            source_tokenizer=self.source_tokenizer,
            exact_match_all=self.exact_match_all,
            match_symbols=self.match_symbols,
            fuzzy_match_all=self.fuzzy_match_all
        )
        # Copy source embeddings for overlapping tokens
        target_embeddings, overlapping_token_indices = self.copy_overlapping_tokens(
            target_embeddings, self.overlapping_tokens, self.missing_tokens
        )

        logger.info(f"Copied source embeddings for {len(overlapping_token_indices)}/{len(self.target_tokens)} "
                    f"target tokens by leveraging the overlap between the source and the target vocabularies")

        return target_embeddings


class CLPTokenizerTransfer(OverlapTokenizerTransfer):
    def __init__(
            self,
            source_model_name_or_path: str,
            target_tokenizer_name_or_path: str,
            helper_model_name_or_path: str,
            helper_tokenizer_name_or_path: str = None,
            target_model_path: str = None
    ):
        """
        Class for transferring embeddings from one tokenizer to another using CLP method by Ostendorff & Rehm (2023).
        :param source_model_name_or_path:
        :param target_tokenizer_name_or_path:
        :param target_model_path:
        :param helper_model_name_or_path:
        :param helper_tokenizer_name_or_path:
        """
        super().__init__(source_model_name_or_path, target_tokenizer_name_or_path, target_model_path)
        self.helper_model_name_or_path = helper_model_name_or_path
        self.helper_tokenizer_name_or_path = helper_tokenizer_name_or_path if helper_tokenizer_name_or_path else helper_model_name_or_path


        if "bert" in self.source_model_name_or_path:
            self.helper_model = AutoModelForMaskedLM.from_pretrained(self.helper_model_name_or_path)
        else:
            self.helper_model = AutoModelForCausalLM.from_pretrained(self.helper_model_name_or_path)
        self.helper_tokenizer = AutoTokenizer.from_pretrained(self.helper_tokenizer_name_or_path)
        self.helper_embeddings = self.helper_model.get_input_embeddings().weight.detach().numpy()
        self.helper_token_to_idx = {t: i for t, i in self.helper_tokenizer.get_vocab().items()}

    def initialize_embeddings(self, **kwargs):
        """
        Method that initializes the embeddings of a given LM with the source embeddings.

        :param kwargs: no kwargs

        :return: The initialized embedding matrix
        """
        target_embeddings = super().initialize_embeddings(**kwargs)

        # Initialize the rest using the helper embeddings
        if self.missing_tokens:
            missing_tokens_list = list(self.missing_tokens.keys())
            overlapping_tokens_list_source = []
            overlapping_tokens_list_target = list(self.overlapping_tokens.keys())
            for t, overlapping_token in self.overlapping_tokens.items():
                overlapping_tokens_list_source.append(overlapping_token.source[0].native_form)
            overlapping_tokens_idxs = [self.source_token_to_idx[t] for t in overlapping_tokens_list_source]
            overlapping_token_vecs = self.source_embeddings[overlapping_tokens_idxs, :]

            helper_missing_tokens_vecs = self.helper_embeddings[[self.helper_token_to_idx[t] for t in missing_tokens_list], :]
            helper_overlapping_token_vecs = self.helper_embeddings[[self.helper_token_to_idx[t] for t in overlapping_tokens_list_target], :]

            # Calculate similarities for each pair of missing and overlapping token embeddings
            sims = cosine_similarity(helper_missing_tokens_vecs, helper_overlapping_token_vecs)

            # similar = 1 => high weight
            # dissimilar = 0 => low weight

            for ti, t in enumerate(tqdm(missing_tokens_list, desc="Initialize target embeddings for missing tokens")):
                # distances to overlapping tokens
                token_sims = sims[ti]
                norm_sims = token_sims / token_sims.sum()

                # weighted average of overlapping token embeddings with weight from similarity in helper token embedding space
                target_vec = np.average(overlapping_token_vecs, axis=0, weights=norm_sims)
                target_embeddings[self.helper_token_to_idx[t]] = target_vec
        return target_embeddings
