import logging
import re
import numpy as np

from tqdm import tqdm
from .tokenizer_transfer import OverlapTokenizerTransfer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class FVTTokenizerTransfer(OverlapTokenizerTransfer):
    def __init__(
            self,
            source_model_name_or_path: str,
            target_tokenizer_name_or_path: str,
            target_model_path: str = None,
            **kwargs
    ):
        """
        Class for transferring embeddings from one tokenizer to another using FVT method by Geee et al. (2022).
        Code adapted from https://github.com/LeonidasY/fast-vocabulary-transfer/blob/main/fvt/
        From the paper:
        @inproceedings{gee-etal-2022-fast,
            title = "Fast Vocabulary Transfer for Language Model Compression",
            author = "Gee, Leonidas  and
              Zugarini, Andrea  and
              Rigutini, Leonardo  and
              Torroni, Paolo",
            booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing: Industry Track",
            month = dec,
            year = "2022",
            address = "Abu Dhabi, UAE",
            publisher = "Association for Computational Linguistics",
            url = "https://aclanthology.org/2022.emnlp-industry.41",
            pages = "409--416",
        }
        :param source_model_name_or_path:
        :param target_tokenizer_name_or_path:
        :param target_model_path:
        """
        super().__init__(source_model_name_or_path, target_tokenizer_name_or_path, target_model_path, **kwargs)

    def initialize_embeddings(self, **kwargs):
        """
        Method that initializes the embeddings of a given LM with the source embeddings.

        :param kwargs: no kwargs

        :return: The initialized embedding matrix
        """
        target_embeddings = super().initialize_embeddings(**kwargs)
        ngram_vocab = self.target_tokenizer.ngram_vocab if hasattr(self.target_tokenizer, 'ngram_vocab') else {}

        # Initialize the rest by partitioning the target token into source tokens using the source tokenizer
        # and averaging the source embeddings of tokens in the partition
        if self.missing_tokens:
            missing_tokens_list = [token for token, missing_token_info in self.missing_tokens]

            for target_token in tqdm(missing_tokens_list, desc="Initialize target embeddings for missing tokens"):
                normalized_target_token = re.sub('^(##|Ġ|▁)', '', target_token)
                if normalized_target_token in ngram_vocab:
                    partition_token_idxs = self.source_tokenizer(
                        normalized_target_token.split('‗'), is_split_into_words=True, add_special_tokens=False, return_tensors='pt'
                    )["input_ids"][0]
                else:
                    partition_token_idxs = self.source_tokenizer(
                        normalized_target_token, add_special_tokens=False, return_tensors='pt'
                    )["input_ids"][0]
                target_token_idx = self.target_token_to_idx[target_token]
                target_embeddings[target_token_idx] = np.mean(self.source_embeddings[partition_token_idxs], axis=0)
        return target_embeddings
