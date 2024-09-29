import logging
import re
import torch

from overrides import override
from .tokenizer_transfer import TokenizerTransfer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class FVTTokenizerTransfer(TokenizerTransfer):
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

        :param source_embeddings: The source embeddings to initialize the target embeddings with.
        :param kwargs: no kwargs

        :return: The initialized embedding matrix
        """
        # For compatibility with the original code
        gen_tokenizer = self.source_tokenizer
        in_tokenizer = self.target_tokenizer

        # tokens_mapping: maps new token indices to old token indices
        # https://github.com/LeonidasY/fast-vocabulary-transfer/blob/9ecbbf2454cff8a27c298e3efc047c29efd32836/fvt/fvt.py#L12
        gen_vocab = gen_tokenizer.get_vocab()
        in_vocab = in_tokenizer.get_vocab()
        ngram_vocab = in_tokenizer.ngram_vocab if hasattr(in_tokenizer, "ngram_vocab") else {}

        self.overlap_based_initialized_tokens = 0
        tokens_map = {}
        for new_token, new_index in in_vocab.items():
            if new_token in gen_vocab:
                # if the same token exists in the old vocabulary, take its embedding
                old_index = gen_vocab[new_token]
                tokens_map[new_index] = [old_index]
                self.overlap_based_initialized_tokens += 1
            else:
                # if not, tokenize the new token using the old vocabulary
                new_token = re.sub('^(##|Ġ|▁)', '', new_token)
                # we modified the call to the gen_tokenizer in order to directly get the input_ids
                if new_token in ngram_vocab:
                    token_partition = gen_tokenizer(
                        new_token.split('‗'), is_split_into_words=True, add_special_tokens=False, return_tensors='pt'
                    )["input_ids"][0]
                else:
                    token_partition = gen_tokenizer(
                        new_token, add_special_tokens=False, return_tensors='pt'
                    )["input_ids"][0]
                tokens_map[new_index] = token_partition

        # embeddings_assignment: assigns the embeddings to the new embedding matrix
        # https://github.com/LeonidasY/fast-vocabulary-transfer/blob/9ecbbf2454cff8a27c298e3efc047c29efd32836/fvt/fvt.py#L50
        # originally: gen_model.get_input_embeddings().weight, but we want to use the passed source_embeddings
        # that can either be the input embeddings or the output embeddings (unembedding matrix)
        gen_matrix = torch.from_numpy(source_embeddings)
        in_matrix = torch.zeros(len(tokens_map), gen_matrix.shape[1])

        self.cleverly_initialized_tokens = 0
        for new_index, old_indices in tokens_map.items():
            # in the original code: old_embedding = torch.mean(gen_matrix[old_indices], axis=0)
            old_indices = torch.tensor(old_indices, dtype=torch.long)
            old_embedding = torch.mean(gen_matrix[old_indices], dim=0)
            in_matrix[new_index] = old_embedding
            self.cleverly_initialized_tokens += 1

        target_embeddings = in_matrix.detach().cpu().numpy()
        logger.info(f"Initialized {self.cleverly_initialized_tokens}({len(self.target_tokens)} target embeddings using FVT method.")
        return target_embeddings
