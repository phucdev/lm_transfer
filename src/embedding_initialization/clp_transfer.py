import logging
import numpy as np
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
