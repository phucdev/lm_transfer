import json
import logging
from pathlib import Path

import numpy as np
import torch
from overrides import override
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer
)
from sklearn.metrics.pairwise import cosine_similarity
from .tokenizer_transfer import OverlapTokenizerTransfer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class CLPTokenizerTransfer(OverlapTokenizerTransfer):
    def __init__(
            self,
            source_model_name_or_path: str,
            target_tokenizer_name_or_path: str,
            helper_model_name_or_path: str,
            helper_tokenizer_name_or_path: str = None,
            target_model_path: str = None,
            **kwargs
    ):
        """
        Class for transferring embeddings from one tokenizer to another using CLP method by Ostendorff & Rehm (2023).
        Adapted from https://github.com/malteos/clp-transfer/blob/main/clp.py
        @misc{Ostendorff2023clp,
          doi = {10.48550/ARXIV.2301.09626},
          author = {Ostendorff, Malte and Rehm, Georg},
          title = {Efficient Language Model Training through Cross-Lingual and Progressive Transfer Learning},
          publisher = {arXiv},
          year = {2023}
        }
        :param source_model_name_or_path:
        :param target_tokenizer_name_or_path:
        :param target_model_path:
        :param helper_model_name_or_path:
        :param helper_tokenizer_name_or_path:
        """
        super().__init__(source_model_name_or_path, target_tokenizer_name_or_path, target_model_path, **kwargs)
        self.helper_model_name_or_path = helper_model_name_or_path
        self.helper_tokenizer_name_or_path = helper_tokenizer_name_or_path if helper_tokenizer_name_or_path else helper_model_name_or_path


        if "bert" in self.source_model_name_or_path:
            self.helper_model = AutoModelForMaskedLM.from_pretrained(self.helper_model_name_or_path)
        else:
            self.helper_model = AutoModelForCausalLM.from_pretrained(self.helper_model_name_or_path)
        self.helper_tokenizer = AutoTokenizer.from_pretrained(self.helper_tokenizer_name_or_path)
        self.helper_embeddings = self.helper_model.get_input_embeddings().weight.detach().numpy()
        self.helper_token_to_idx = {t: i for t, i in self.helper_tokenizer.get_vocab().items()}
        self.helper_output_embeddings = self.helper_model.get_output_embeddings().weight.detach().numpy()

        self.transfer_method = "clp"

    @override
    def save_parameters_to_dict(self):
        """
        Method that saves the parameters of the CLPTokenizerTransfer object to a dictionary.

        :return: A dictionary containing the parameters of the CLPTokenizerTransfer object.
        """
        parameters = super().save_parameters_to_dict()
        parameters["helper_model_name_or_path"] = self.helper_model_name_or_path
        parameters["helper_tokenizer_name_or_path"] = self.helper_tokenizer_name_or_path
        return parameters

    @override
    def initialize_embeddings(self, source_embeddings, helper_embeddings, **kwargs):
        """
        Method that initializes the embeddings of a LM with a target tokenizer given a source LM.
        Leverages overlap between the source vocabulary and the target vocabulary to directly copy source embeddings
        and uses a helper model to initialize the rest.

        :param source_embeddings: The source embeddings (either the input embeddings or the output embeddings)
        :param helper_embeddings: The helper embeddings (either the input embeddings or the output embeddings)
        :param kwargs: no kwargs

        :return: The initialized embedding matrix
        """
        logger.info("(1/2) Create random fallback matrix for target embeddings and copy source embeddings for overlapping tokens...")
        target_embeddings = super().initialize_embeddings(source_embeddings=source_embeddings, **kwargs)

        logger.info("(2/2) Initialize the rest based on the overlap and the helper embeddings with the CLP method")
        if self.missing_tokens:
            missing_tokens_idxs = [missing_token_info.target.id for token, missing_token_info in self.missing_tokens]
            overlapping_tokens_source_idxs = []
            overlapping_tokens_target_idxs = []
            for token, overlapping_token_info in self.overlapping_tokens:
                target_token_idx = overlapping_token_info.target.id
                source_token_idx = overlapping_token_info.source[0].id
                overlapping_tokens_source_idxs.append(source_token_idx)
                overlapping_tokens_target_idxs.append(target_token_idx)
            overlapping_tokens_idxs = [
                overlapping_token_info.source[0].id for t, overlapping_token_info in self.overlapping_tokens
            ]
            overlapping_token_vecs = source_embeddings[overlapping_tokens_idxs, :]

            helper_missing_tokens_vecs = helper_embeddings[missing_tokens_idxs, :]
            helper_overlapping_token_vecs = helper_embeddings[overlapping_tokens_target_idxs, :]

            # Calculate similarities for each pair of missing and overlapping token embeddings
            sims = cosine_similarity(helper_missing_tokens_vecs, helper_overlapping_token_vecs)

            # similar = 1 => high weight
            # dissimilar = 0 => low weight

            for ti, helper_token_idx in enumerate(tqdm(missing_tokens_idxs, desc="Initialize target embeddings for missing tokens")):
                # distances to overlapping tokens
                token_sims = sims[ti]
                norm_sims = token_sims / token_sims.sum()

                # weighted average of overlapping token embeddings with weight from similarity in helper token embedding space
                target_vec = np.average(overlapping_token_vecs, axis=0, weights=norm_sims)
                target_embeddings[helper_token_idx] = target_vec
                self.cleverly_initialized_tokens += 1

        logger.info(f"Initialized {self.cleverly_initialized_tokens}/{len(self.target_tokens)} tokens with the CLP method.")
        return target_embeddings

    @override
    def transfer(self, **kwargs):
        """
        Method that creates a new LM model with transferred embeddings.
        :param kwargs: no kwargs

        :return: A new in_domain model
        """

        target_embeddings = self.initialize_embeddings(
            source_embeddings=self.source_embeddings,
            helper_embeddings=self.helper_embeddings,
            **kwargs
        )

        target_model = self.source_model
        target_model.resize_token_embeddings(len(self.target_tokenizer))
        target_model.get_input_embeddings().weight.data = torch.from_numpy(target_embeddings)

        # For models with separate embedding and unembedding matrix we need to repeat the process
        # for the unembedding matrix
        if not target_model.config.tie_word_embeddings:
            target_output_embeddings = self.initialize_embeddings(
                source_embeddings=self.source_output_embeddings,
                helper_embeddings=self.helper_output_embeddings,
                **kwargs
            )
            target_model.get_output_embeddings().weight.data = torch.from_numpy(target_output_embeddings)

        if self.target_model_path:
            self.target_tokenizer.save_pretrained(self.target_model_path)
            target_model.save_pretrained(self.target_model_path)
            with open(Path(self.target_model_path) / "transfer_information.json", "w") as f:
                information_dict = self.save_parameters_to_dict()
                json.dump(information_dict, f, indent=2)
        return target_model
