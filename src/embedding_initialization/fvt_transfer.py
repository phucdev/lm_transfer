import logging
import re
import numpy as np

from overrides import override
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

        self.transfer_method = "fvt"

    @override
    def save_parameters_to_dict(self):
        """
        Method that saves the parameters of the FVT transfer method to a dictionary.
        :return: The dictionary containing the parameters of the FVT transfer method.
        """
        parameters = super().save_parameters_to_dict()
        parameters["transfer_method"] = self.transfer_method
        return parameters

    @override
    def initialize_embeddings(self, source_embeddings, **kwargs):
        """
        Method that initializes the embeddings of a given LM with the source embeddings.
        For target tokens that exist in the source vocabulary, the embeddings are copied from the source model.
        The rest are tokenized using the source tokenizer and the corresponding source embeddings are averaged.
        Compared to the original implementation we use the matching strategy of FOCUS (Dobler & de Melo, 2023) to
        find overlapping tokens between the source and target vocabularies.
        This involves canonicalizing the tokens before matching them.
        Another difference is that we also copy the embeddings of the special tokens from the source model.

        :param source_embeddings: The source embeddings to initialize the target embeddings with.
        :param kwargs: no kwargs

        :return: The initialized embedding matrix
        """
        logger.info("(1/2) Create random fallback matrix for target embeddings and copy source embeddings for overlapping tokens...")
        target_embeddings = super().initialize_embeddings(source_embeddings=source_embeddings, **kwargs)
        ngram_vocab = self.target_tokenizer.ngram_vocab if hasattr(self.target_tokenizer, 'ngram_vocab') else {}

        logger.info("(2/2) Initialize target embeddings for missing tokens using FVT method...")
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
                # TODO instead of averaging, we could use some other method for aggregating the embeddings
                target_embeddings[target_token_idx] = np.mean(source_embeddings[partition_token_idxs], axis=0)
                self.cleverly_initialized_tokens += 1
        logger.info(f"Initialized {self.cleverly_initialized_tokens}({len(self.target_tokens)} target embeddings using FVT method.")
        return target_embeddings
