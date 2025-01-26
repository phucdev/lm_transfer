import json
import logging
import re
from pathlib import Path

import torch
import math

from collections import Counter
from datasets import load_dataset
from overrides import override
from typing import Optional

from tqdm import tqdm

from lm_transfer.embedding_initialization.tokenizer_transfer import TokenizerTransfer

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
            seed: int = 42,
            device="cpu",
            aggregation_method="mean",
            target_training_data_path: Optional[str] = None,
            num_proc: Optional[int] = None,
            log_scale: bool = False,
            rescale: bool = False,
            minimize_punctuation_weight: bool = False,
            freq_dict_path: Optional[str] = None,
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
        :param seed:
        :param device:
        :param aggregation_method:
        :param target_training_data_path:
        :param num_proc:
        :param log_scale:
        :param rescale:
        :param minimize_punctuation_weight:
        :param freq_dict_path:
        """
        super().__init__(source_model_name_or_path, target_tokenizer_name_or_path, target_model_path, **kwargs)
        self.seed = seed
        self.device = device
        self.aggregation_method = aggregation_method
        self.target_training_data_path = target_training_data_path
        self.num_proc = num_proc
        self.log_scale = log_scale
        self.rescale = rescale
        self.minimize_punctuation_weight = minimize_punctuation_weight
        self.freq_dict_path = freq_dict_path
        self.gen = torch.Generator(device=self.device).manual_seed(self.seed)
        self.id_to_source_token = {v: k for k, v in self.source_tokenizer.get_vocab().items()}
        self.id_to_target_token = {v: k for k, v in self.target_tokenizer.get_vocab().items()}
        self.transfer_method = "fvt"

    @override
    def save_parameters_to_dict(self):
        """
        Method that saves the parameters of the FVT transfer method to a dictionary.
        :return: The dictionary containing the parameters of the FVT transfer method.
        """
        parameters = super().save_parameters_to_dict()
        parameters["seed"] = self.seed
        parameters["aggregation_method"] = self.aggregation_method
        parameters["target_training_data_path"] = self.target_training_data_path
        parameters["num_proc"] = self.num_proc
        parameters["log_scale"] = self.log_scale
        parameters["rescale"] = self.rescale
        parameters["minimize_punctuation_weight"] = self.minimize_punctuation_weight
        parameters["transfer_method"] = self.transfer_method
        return parameters

    def get_token_frequencies(self, tokenizer):
        train_data = load_dataset("json", data_files={"train": self.target_training_data_path}, split="train")

        def tokenize_function(example):
            """
            Tokenize each text in the batch using old_tokenizer.tokenize()
            and store the list of tokens in a new field called 'tokens'.
            """
            all_tokens = []
            for text in example["text"]:
                # old_tokenizer.tokenize() returns a list of subword strings
                tokens = tokenizer.tokenize(text)
                all_tokens.append(tokens)
            return {"tokens": all_tokens}

        tokenized = train_data.map(tokenize_function, batched=True, num_proc=self.num_proc).remove_columns("text")
        freq_counter = Counter()

        # We iterate over the dataset and extend a Python list with all tokens
        for example in tqdm(tokenized, desc="Counting token frequencies"):
            # tokens_list is a list of subwords for one example
            tokens_list = example["tokens"]
            freq_counter.update(tokens_list)

        return freq_counter

    @staticmethod
    def is_punctuation(token):
        return not any(c.isalpha() for c in token)

    def compute_weighted_mean(
        self,
        old_indices,
        gen_matrix,
        freq_counter=None,
        punc_weight_factor=0.01,
        avg_source_norm=None
    ):
        """Compute weights for the weighted mean of the embeddings.
        old_indices: list of old/source token indices of the decomposed new token
        gen_matrix: source embedding matrix
        freq_counter: dictionary mapping old_subword -> frequency
        punc_weight_factor: multiply weight for punctuation tokens with this factor
        avg_source_norm: average L2 norm of the source embeddings

        Returns the weighted mean of the embeddings of the old/source tokens and the weights.
        """
        old_tokens = [self.id_to_source_token[int(i)] for i in old_indices]
        if len(old_indices) == 1:
            # Direct copy for overlapping tokens
            old_indices = old_indices.to(torch.long)
            old_embedding = gen_matrix[old_indices]
            return old_embedding, [1.0]
        else:
            if freq_counter is not None and self.aggregation_method == "freq_weighted":
                # Compute weights
                freqs = []
                for s in old_tokens:
                    if self.is_punctuation(s) and self.minimize_punctuation_weight:
                        # Completely reduce the weight of punctuation tokens
                        freqs.append(1)
                    else:
                        freqs.append(freq_counter.get(s, 0))
                total = sum(freqs)
                if total == 0:
                    # fallback to unweighted mean
                    old_indices = old_indices.to(torch.long)
                    old_embedding = torch.mean(gen_matrix[old_indices], dim=0)
                    weights = [1.0 / len(old_indices)] * len(old_indices)
                else:
                    if self.log_scale:
                        # To handle huge disparity between frequencies we can use log scale
                        weights = [math.log(f + 1) for f in freqs]
                    else:
                        weights = [max(f, 1) for f in freqs]
                    # Normalize weights
                    weights = [w / sum(weights) for w in weights]

                    # Weighted sum
                    old_embedding = torch.zeros(gen_matrix.size(1))
                    for w, s in zip(weights, old_indices):
                        if s is not None:
                            old_embedding += w * gen_matrix[s]
            elif self.aggregation_method == "subword_length_weighted":
                # Use the relative length of the subwords as weights
                # The first token often contains some prefix which should not count
                subword_lengths = [max(1, len(re.sub("^(##|Ġ|▁)", "", subword))) for subword in old_tokens]
                if self.log_scale:
                    subword_lengths = [math.log(l + 1) for l in subword_lengths]
                weights = [w / sum(subword_lengths) for w in subword_lengths]
                old_embedding = torch.zeros(gen_matrix.size(1))
                for w, s in zip(weights, old_indices):
                    if s is not None:
                        old_embedding += w * gen_matrix[s]
            else:
                # Original FVT approach simply calculates the mean of the embeddings
                if self.minimize_punctuation_weight:
                    # Use punc_weight for punctuation tokens when calculating the mean
                    avg_weight = 1 / len(old_indices)
                    weights = [punc_weight_factor * avg_weight if self.is_punctuation(old_token) else avg_weight for old_token in old_tokens]
                    weights = [w / sum(weights) for w in weights]
                    old_embedding = torch.zeros(gen_matrix.size(1))
                    for w, s in zip(weights, old_indices):
                        if s is not None:
                            old_embedding += w * gen_matrix[s]
                else:
                    # in the original code: old_embedding = torch.mean(gen_matrix[old_indices], axis=0)
                    old_indices = old_indices.to(torch.long)
                    old_embedding = torch.mean(gen_matrix[old_indices], dim=0)
                    weights = [1.0 / len(old_indices)] * len(old_indices)
            # In case we average multiple source embeddings, we can optionally rescale the resulting embedding to match
            # the average L2 norm of the source embeddings
            if len(old_indices) > 1 and self.rescale and avg_source_norm is not None:
                old_embedding *= avg_source_norm / old_embedding.norm()
            return old_embedding, weights

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

        if self.aggregation_method == "freq_weighted":
            if self.freq_dict_path is not None and Path(self.freq_dict_path).exists():
                with open(self.freq_dict_path, "r") as f:
                    freq_counter = json.load(f)
                logger.info(f"Token frequencies loaded from {self.freq_dict_path}.")
            else:
                freq_counter = self.get_token_frequencies(in_tokenizer)
                logger.info(f"Token frequencies collected for {len(freq_counter)} tokens.")
                if self.freq_dict_path is not None:
                    with open(self.freq_dict_path, "w") as f:
                        f.write(json.dumps(freq_counter))
                    logger.info(f"Token frequencies saved to {self.freq_dict_path}.")
        else:
            freq_counter = {}

        self.overlap_based_initialized_tokens = 0
        tokens_map = {}
        for new_token, new_index in in_vocab.items():
            if new_token in gen_vocab:
                # if the same token exists in the old vocabulary, take its embedding
                old_index = gen_vocab[new_token]
                tokens_map[new_index] = torch.tensor([old_index], dtype=torch.long)
                self.overlap_based_initialized_tokens += 1
            else:
                # if not, tokenize the new token using the old vocabulary
                tmp_new_token = new_token
                new_token = re.sub("^(##|Ġ|▁)", "", new_token)
                if new_token == "":
                    new_token = tmp_new_token   # reverse substitution if the token is empty
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
                if len(token_partition) < 1:
                    logger.warning(f"Token {tmp_new_token} could not be tokenized with source vocabulary.")

        # embeddings_assignment: assigns the embeddings to the new embedding matrix
        # https://github.com/LeonidasY/fast-vocabulary-transfer/blob/9ecbbf2454cff8a27c298e3efc047c29efd32836/fvt/fvt.py#L50
        # originally: gen_model.get_input_embeddings().weight, but we want to use the passed source_embeddings
        # that can either be the input embeddings or the output embeddings (unembedding matrix)
        gen_matrix = torch.from_numpy(source_embeddings).to(self.device)
        in_matrix = torch.zeros(len(tokens_map), gen_matrix.shape[1], device=self.device)

        emb_mean = gen_matrix.mean(dim=0)
        emb_std = gen_matrix.std(dim=0)

        if self.rescale:
            avg_source_norm = gen_matrix.norm(dim=1).mean()  # average L2 norm
            logger.info("Rescaling target embeddings to match source embeddings' average L2 norm.")
        else:
            avg_source_norm = None

        self.cleverly_initialized_tokens = 0
        for new_index, old_indices in tokens_map.items():
            if len(old_indices) > 0:
                old_embedding, weights = self.compute_weighted_mean(
                    old_indices,
                    gen_matrix,
                    freq_counter,
                    avg_source_norm=avg_source_norm
                )
                in_matrix[new_index] = old_embedding
                self.cleverly_initialized_tokens += 1

                self.sources[self.id_to_target_token[new_index]] = (
                    [self.id_to_source_token[int(i)] for i in old_indices],
                    old_indices.tolist(),
                    weights
                )
            else:
                # Random initialization for tokens that could not be found in the source vocabulary
                in_matrix[new_index] = torch.normal(emb_mean, emb_std, generator=self.gen)

        target_embeddings = in_matrix.detach().cpu().numpy()
        logger.info(f"Initialized {self.cleverly_initialized_tokens}/{len(self.target_tokens)} target embeddings using FVT method.")
        return target_embeddings
