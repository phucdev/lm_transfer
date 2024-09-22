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
from .tokenizer_transfer import TokenizerTransfer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class CLPTokenizerTransfer(TokenizerTransfer):
    def __init__(
            self,
            source_model_name_or_path: str,
            target_tokenizer_name_or_path: str,
            helper_model_name_or_path: str,
            helper_tokenizer_name_or_path: str = None,
            target_model_path: str = None,
            seed: int = 42,
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
        :param seed:
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
        self.seed = seed

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
        parameters["seed"] = self.seed
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
        # Overlapping tokens
        target_tokens = set(self.helper_tokenizer.get_vocab().keys())
        source_tokens = set(self.source_tokenizer.get_vocab().keys())

        overlapping_tokens = target_tokens & source_tokens
        missing_tokens = target_tokens - source_tokens

        missing_tokens_list = list(missing_tokens)
        overlapping_tokens_list = list(overlapping_tokens)

        logger.info(f'{len(overlapping_tokens)=}; {len(missing_tokens)=}')

        if not overlapping_tokens:
            raise ValueError("No overlapping tokens found")

        source_token_to_idx = {t: i for t, i in self.source_tokenizer.get_vocab().items()}
        # target_token_to_idx = {t: i for t, i in target_tokenizer.get_vocab().items()}
        helper_token_to_idx = {t: i for t, i in self.helper_tokenizer.get_vocab().items()}

        overlapping_tokens_idxs = [source_token_to_idx[t] for t in overlapping_tokens_list]
        overlapping_token_vecs = source_embeddings[overlapping_tokens_idxs, :]

        logger.info(f"{overlapping_token_vecs.shape=}")

        # Target embeddings

        # Random init target embeddings with mean+std of source embeddings
        target_embeddings = np.random.normal(
            np.mean(source_embeddings, axis=0),
            np.std(source_embeddings, axis=0),
            (
                len(target_tokens),
                source_embeddings.shape[1]
            )
        )

        # Set overlapping tokens
        self.overlap_based_initialized_tokens = 0
        for t in overlapping_tokens:
            target_embeddings[helper_token_to_idx[t]] = source_embeddings[source_token_to_idx[t]]
            self.overlap_based_initialized_tokens += 1
        self.cleverly_initialized_tokens = self.overlap_based_initialized_tokens

        if missing_tokens:

            helper_missing_tokens_vecs = helper_embeddings[[helper_token_to_idx[t] for t in missing_tokens_list], :]
            helper_overlapping_token_vecs = helper_embeddings[[helper_token_to_idx[t] for t in overlapping_tokens_list],
                                            :]

            # Similarities for missing tokens
            sims = cosine_similarity(helper_missing_tokens_vecs, helper_overlapping_token_vecs)

            # similar = 1 => high weight
            # dissimilar = 0 => low weight

            for ti, t in enumerate(tqdm(missing_tokens_list)):  # 1:14hrs (12min with batch sim)
                # distances to overlapping tokens
                token_sims = sims[ti]
                norm_sims = token_sims / token_sims.sum()

                # weighted average of overlapping token embeddings with weight from similarity in helper token embedding space
                target_vec = np.average(overlapping_token_vecs, axis=0, weights=norm_sims)
                target_embeddings[helper_token_to_idx[t]] = target_vec
                self.cleverly_initialized_tokens += 1
        else:
            logger.warning("No missing tokens")

        logger.info(
            f"Initialized {self.cleverly_initialized_tokens}/{len(self.target_tokens)} tokens with the CLP method."
        )
        return target_embeddings

    @override
    def transfer(self, **kwargs):
        """
        Method that creates a new LM model with transferred embeddings.
        :param kwargs: no kwargs

        :return: A new in_domain model
        """
        np.random.seed(self.seed)
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
