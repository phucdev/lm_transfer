import os
import logging
import numpy as np

from pathlib import Path
from typing import Optional
from overrides import override
from tqdm import tqdm

from lm_transfer.utils.download_utils import download
from lm_transfer.embedding_initialization.tokenizer_transfer import OverlapTokenizerTransfer


CACHE_DIR = (
    (Path(os.getenv("XDG_CACHE_HOME", "~/.cache")) / "wechsel").expanduser().resolve()
)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class BilingualDictionaryTransfer(OverlapTokenizerTransfer):
    def __init__(
            self,
            source_model_name_or_path: str,
            target_tokenizer_name_or_path: str,
            target_model_path: str = None,
            bilingual_dictionary_path: Optional[str] = None,
            source_language_identifier: Optional[str] = None,
            target_language_identifier: Optional[str] = None,
            copy_overlap: bool = False,
            prioritize_overlap: bool = False,
            skip_phrases: bool = False,
            **kwargs):
        """
        Transfer method based on co-occurrences in a bilingual dictionary.
        :param source_model_name_or_path:
        :param target_tokenizer_name_or_path:
        :param target_model_path:
        :param bilingual_dictionary_path:
        :param source_language_identifier:
        :param target_language_identifier:
        :param copy_overlap: Copy embeddings for overlapping tokens.
        :param prioritize_overlap: Directly copy embedding for overlapping tokens.
        :param skip_phrases: Skip dictionary pair if encountering a phrase longer than 3 words.
        :param kwargs:
        """
        super().__init__(source_model_name_or_path, target_tokenizer_name_or_path, target_model_path, **kwargs)
        self.bilingual_dictionary_path = bilingual_dictionary_path
        self.source_language_identifier = source_language_identifier
        self.target_language_identifier = target_language_identifier
        self.copy_overlap = copy_overlap
        self.prioritize_overlap = prioritize_overlap
        self.skip_phrases = skip_phrases

        self.transfer_method = "bilingual_dictionary"

    @override
    def save_parameters_to_dict(self):
        params = super().save_parameters_to_dict()
        params.update({
            "bilingual_dictionary_path": self.bilingual_dictionary_path,
            "source_language_identifier": self.source_language_identifier,
            "target_language_identifier": self.target_language_identifier,
            "copy_overlap": self.copy_overlap,
            "prioritize_overlap": self.prioritize_overlap,
            "skip_phrases": self.skip_phrases,
        })
        return params

    @override
    def initialize_embeddings(self, source_embeddings, skip_phrases=False, **kwargs):
        # Initialize target embeddings randomly
        target_embeddings = self.initialize_random_embeddings(source_embeddings=source_embeddings)

        overlapping_token_indices = []
        if self.copy_overlap:
            # Get overlapping tokens and missing tokens
            self.overlapping_tokens, self.missing_tokens = self.get_overlapping_tokens()
            # Copy source embeddings for overlapping tokens
            target_embeddings, overlapping_token_indices = self.copy_overlapping_tokens(
                source_embeddings=source_embeddings,
                target_embeddings=target_embeddings,
            )

        # Copy embeddings for overlapping special tokens
        target_embeddings, overlapping_special_token_indices = self.copy_special_tokens(
            source_embeddings=source_embeddings,
            target_embeddings=target_embeddings,
        )
        for special_token_idx in overlapping_special_token_indices:
            if special_token_idx not in overlapping_token_indices:
                overlapping_token_indices.append(special_token_idx)

        self.overlap_based_initialized_tokens = len(overlapping_token_indices)
        self.cleverly_initialized_tokens = self.overlap_based_initialized_tokens

        # Load and read bilingual dictionary
        if self.bilingual_dictionary_path is None:
            raise ValueError(
                "`bilingual_dictionary_path` must not be `None` if `align_strategy` is 'bilingual_dictionary'."
            )
        if not os.path.exists(self.bilingual_dictionary_path):
            self.bilingual_dictionary_path = download(
                f"https://raw.githubusercontent.com/CPJKU/wechsel/main/dicts/data/{self.bilingual_dictionary_path}.txt",
                CACHE_DIR / f"{self.bilingual_dictionary_path}.txt",
            )
        dictionary = []

        for line in open(self.bilingual_dictionary_path, "r"):
            line = line.strip()
            try:
                source_word, target_word = line.split("\t")
            except ValueError:
                source_word, target_word = line.split()
            dictionary.append((source_word, target_word))

        # Count co-occurrences of tokens in translation pairs
        token_freqmatrix = np.zeros((len(self.target_tokens), len(self.source_tokens)), dtype=np.float32)
        source_token_freqs = np.zeros(len(self.source_tokens))
        for en, vi in tqdm(dictionary, desc="Counting co-occurrences of tokens in translation pairs"):
            if skip_phrases and len(en.split()) > 3:  # heuristic to filter out phrases
                continue
            en_token_ids = self.source_tokenizer(en, add_special_tokens=False, return_tensors='pt')['input_ids'][0]
            vi_token_ids = self.target_tokenizer(vi, add_special_tokens=False, return_tensors='pt')['input_ids'][0]
            # TODO handle cases where are direct translations differently
            # debug
            # en_tokens = self.source_tokenizer.convert_ids_to_tokens(en_token_ids)
            # vi_tokens = self.target_tokenizer.convert_ids_to_tokens(vi_token_ids)
            for vi_t in vi_token_ids:
                for en_t in en_token_ids:
                    token_freqmatrix[vi_t][en_t] += 1 / len(en_token_ids)  # adjust by decomposition lengths
            source_token_freqs[en_token_ids] += 1

        # Adding a small number to avoid division by zero, if necessary
        row_sums = np.sum(token_freqmatrix, axis=1).reshape(-1, 1) + 1e-9  # adding a small constant
        normalized_matrix = token_freqmatrix / row_sums  # relative frequencies
        # softmax_probs = softmax(token_freqmatrix, axis=1)
        # sparsemax_probs = entmax.sparsemax(torch.tensor(token_freqmatrix), dim=1).numpy()
        # normalized_source_token_freqs = source_token_freqs / np.sum(source_token_freqs)  # all close to zero
        # adjusted_matrix = normalized_matrix / (normalized_source_token_freqs + 1e-9)  # adjusted by source token frequencies

        # Initialize target embeddings based on bilingual dictionary
        dictionary_token_indices = []
        logger.info(f"Initializing target embeddings for missing tokens with translations "
                    f"({self.copy_overlap=}, {self.prioritize_overlap=})")
        for i in tqdm(range(normalized_matrix.shape[0]), desc="Initialize target embeddings for missing tokens with "
                                                              "translations"):
            if i in self.target_tokenizer.all_special_ids:
                continue
            elif self.copy_overlap and self.prioritize_overlap and i in overlapping_token_indices:
                continue
            # Find those whose entry is non-zero: has a translation
            relevant_source_embedding_indices = np.nonzero(normalized_matrix[i, :])[0]
            relevant_source_embeddings = source_embeddings[[t for t in relevant_source_embedding_indices], :]

            norm_freqs = normalized_matrix[i, relevant_source_embedding_indices]
            norm_freqs_sum = norm_freqs.sum()
            # adjusted_weights = adjusted_matrix[i, relevant_source_embedding_indices]
            if norm_freqs_sum == 0.0:
                continue
            weights = norm_freqs
            target_vec = np.average(relevant_source_embeddings, axis=0, weights=weights)
            target_embeddings[i] = target_vec

            # regular_sum = weights.sum()
            # softmax_sum = softmax_probs[i, relevant_source_embedding_indices].sum()

            # debugging
            abs_freqs = token_freqmatrix[i, relevant_source_embedding_indices]
            # softmaxed_relevant_source_embedding_indices = np.nonzero(softmax_probs[i, :])[0]
            # softmaxed_freqs = softmax_probs[i, softmaxed_relevant_source_embedding_indices]
            # softmaxed_relevant_tokens = source_tokenizer.convert_ids_to_tokens(softmaxed_relevant_source_embedding_indices)
            # sparsemaxed_freqs = sparsemax_probs[i, relevant_source_embedding_indices]
            target_token = self.target_tokenizer.convert_ids_to_tokens([i])[0]
            relevant_source_tokens = self.source_tokenizer.convert_ids_to_tokens(relevant_source_embedding_indices)
            sorted_relevant_source_tokens = [
                (w, f, t) for w, f, t in
                sorted(zip(weights, abs_freqs, relevant_source_tokens),
                       key=lambda pair: pair[0], reverse=True)
            ]
            # softmaxed_sorted_relevant_source_tokens = [
            #     (w, t) for w, t in
            #     sorted(zip(softmaxed_freqs, softmaxed_relevant_tokens),
            #            key=lambda pair: pair[0], reverse=True)
            # ]
            if target_token == '▁của':
                logger.debug(f'{target_token=}; {relevant_source_tokens=}')
                logger.debug(f'{list(sorted_relevant_source_tokens)}')

            dictionary_token_indices.append(i)
            self.cleverly_initialized_tokens += 1

        logger.info(f"Initialized {len(dictionary_token_indices)} target embeddings with translations")
        logger.info(f"Initialized {len(overlapping_token_indices)} target embeddings with overlapping tokens")
        logger.info(f"Initialized {self.cleverly_initialized_tokens}/"
                    f"{target_embeddings.shape[0]} target embeddings using heuristics in total")

        return target_embeddings
