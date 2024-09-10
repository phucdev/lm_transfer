import logging
import os
import math

import fasttext
import fasttext.util
import numpy as np

from pathlib import Path
from typing import Literal, Optional
from scipy.linalg import orthogonal_procrustes
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from tqdm import tqdm
from .tokenizer_transfer import OverlapTokenizerTransfer
from ..training.fasttext_embs import load_embeddings
from ..utils.download_utils import download


CACHE_DIR = (
    (Path(os.getenv("XDG_CACHE_HOME", "~/.cache")) / "wechsel").expanduser().resolve()
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def softmax(x, axis=-1):
    return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)


class WordEmbedding:
    """
    Uniform interface to fastText models and gensim Word2Vec models.
    """

    def __init__(self, model):
        self.model = model

        if isinstance(model, fasttext.FastText._FastText):
            self.kind = "fasttext"
        elif isinstance(model, Word2Vec):
            self.kind = "word2vec"
        else:
            raise ValueError(
                f"{model} seems to be neither a fastText nor Word2Vec model."
            )

    def has_subword_info(self):
        return self.kind == "fasttext"

    def get_words_and_freqs(self):
        if self.kind == "fasttext":
            return self.model.get_words(include_freq=True, on_unicode_error="ignore")
        elif self.kind == "word2vec":
            return self.model.wv.index_to_key, self.model.wv.expandos["count"]

    def get_dimension(self):
        if self.kind == "fasttext":
            return self.model.get_dimension()
        elif self.kind == "word2vec":
            return self.model.wv.vector_size

    def get_word_vector(self, word):
        if self.kind == "fasttext":
            return self.model.get_word_vector(word)
        elif self.kind == "word2vec":
            return self.model.wv[word]

    def get_word_id(self, word):
        if self.kind == "fasttext":
            return self.model.get_word_id(word)
        elif self.kind == "word2vec":
            return self.model.wv.key_to_index.get(word, -1)


class WechselTokenizerTransfer(OverlapTokenizerTransfer):
    def __init__(
            self,
            source_model_name_or_path: str,
            target_tokenizer_name_or_path: str,
            target_model_path: str = None,
            bilingual_dictionary_path: Optional[str] = None,
            source_language_identifier: Optional[str] = None,
            target_language_identifier: Optional[str] = None,
            align_with_bilingual_dictionary: bool = True,
            use_subword_info: bool = True,
            max_n_word_vectors: Optional[int] = None,
            neighbors: int = 10,
            temperature: float = 0.1,
            auxiliary_embedding_mode: Literal["fasttext-tokenlevel", "fasttext-wordlevel"] = "fasttext-wordlevel",
            source_training_data_path: Optional[str] = None,
            target_training_data_path: Optional[str] = None,
            source_fasttext_model_path: Optional[str] = None,
            target_fasttext_model_path: Optional[str] = None,
            fasttext_model_epochs: int = 3,
            fasttext_model_dim: int = 100,
            fasttext_model_min_count: int = 10,
            processes: Optional[int] = None,
            seed: int = 42,
            device="cpu",
            verbosity: Literal["debug", "info", "silent"] = "info",
            **kwargs
    ):
        """
        Class for transferring embeddings from one tokenizer to another using WECHSEL method by Minixhofer et al. (2022)
        Code adapted from https://github.com/cpjku/wechsel
        From the paper:
        @inproceedings{minixhofer-etal-2022-wechsel,
            title = "{WECHSEL}: Effective initialization of subword embeddings for cross-lingual transfer of monolingual language models",
            author = "Minixhofer, Benjamin  and
              Paischer, Fabian  and
              Rekabsaz, Navid",
            booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
            month = jul,
            year = "2022",
            address = "Seattle, United States",
            publisher = "Association for Computational Linguistics",
            url = "https://aclanthology.org/2022.naacl-main.293",
            pages = "3992--4006",
            abstract = "Large pretrained language models (LMs) have become the central building block of many NLP applications. Training these models requires ever more computational resources and most of the existing models are trained on English text only. It is exceedingly expensive to train these models in other languages. To alleviate this problem, we introduce a novel method {--} called WECHSEL {--} to efficiently and effectively transfer pretrained LMs to new languages. WECHSEL can be applied to any model which uses subword-based tokenization and learns an embedding for each subword. The tokenizer of the source model (in English) is replaced with a tokenizer in the target language and token embeddings are initialized such that they are semantically similar to the English tokens by utilizing multilingual static word embeddings covering English and the target language. We use WECHSEL to transfer the English RoBERTa and GPT-2 models to four languages (French, German, Chinese and Swahili). We also study the benefits of our method on very low-resource languages. WECHSEL improves over proposed methods for cross-lingual parameter transfer and outperforms models of comparable size trained from scratch with up to 64x less training effort. Our method makes training large language models for new languages more accessible and less damaging to the environment. We make our code and models publicly available.",
        }
        :param source_model_name_or_path:
        :param target_tokenizer_name_or_path:
        :param target_model_path:
        :param bilingual_dictionary_path: Path to a file containing bilingual dictionary for the source and target languages. Defaults to None.
        :param source_language_identifier: Two-letter language code for downloading pretrained fasttext word embeddings if using `fasttext-wordlevel`. Defaults to None.
        :param target_language_identifier: Two-letter language code for downloading pretrained fasttext word embeddings if using `fasttext-wordlevel`. Defaults to None.
        :param align_with_bilingual_dictionary:
                - If `False`, this will load already aligned fasttext embeddings.
                - If `True`, a bilingual dictionary must be passed
                    which will be used to align the embeddings using the Orthogonal Procrustes method.
        :param use_subword_info: Whether to use fastText subword information. Defaults to True.
        :param max_n_word_vectors: Maximum number of vectors to consider (only relevant if `use_subword_info` is False). Defaults to None.
        :param neighbors: Number of neighbors to consider for initializing embeddings. Defaults to 10.
        :param auxiliary_embedding_mode ("fasttext-tokenlevel" or "fasttext-wordlevel"): The type of auxiliary embeddings to use. Defaults to "fasttext-tokenlevel".
        :param source_training_data_path: Path to a file containing lines of text in the source language for training a fasttext model. Only necessary if using `fasttext-tokenlevel`. Defaults to None.
        :param target_training_data_path: Path to a file containing lines of text in the target language for training a fasttext model. Only necessary if using `fasttext-tokenlevel`. Defaults to None.
        :param source_fasttext_model_path: Path to a pretrained fasttext model for the source tokenizer. Defaults to None.
        :param target_fasttext_model_path: Path to a pretrained fasttext model for the target tokenizer. Defaults to None.
        :param fasttext_model_epochs: Number of epochs if training a custom fasttext model. Defaults to 3.
        :param fasttext_model_dim: Dimension size if training a custom fasttext model. Defaults to 100.
        :param fasttext_model_min_count: Minimum number of occurrences for a token to be included if training a custom fasttext model. Defaults to 10.
        :param processes: Number of processes for parallelized workloads. Defaults to None, which uses heuristics based on available hardware.
        :param seed: Defaults to 42.
        :param device: Defaults to "cpu".
        :param verbosity: Defaults to "info".
        """
        super().__init__(source_model_name_or_path, target_tokenizer_name_or_path, target_model_path, **kwargs)
        self.bilingual_dictionary_path = bilingual_dictionary_path
        self.source_language_identifier = source_language_identifier
        self.target_language_identifier = target_language_identifier
        self.align_with_bilingual_dictionary = align_with_bilingual_dictionary
        self.use_subword_info = use_subword_info
        self.max_n_word_vectors = max_n_word_vectors
        self.neighbors = neighbors
        self.temperature = temperature
        # Additional fasttext parameters in case we want to train fasttext embeddings from scratch.
        # Those are not used at the moment.
        self.auxiliary_embedding_mode = auxiliary_embedding_mode
        self.source_training_data_path = source_training_data_path
        self.target_training_data_path = target_training_data_path
        self.source_fasttext_model_path = source_fasttext_model_path
        self.target_fasttext_model_path = target_fasttext_model_path
        self.fasttext_model_epochs = fasttext_model_epochs
        self.fasttext_model_dim = fasttext_model_dim
        self.fasttext_model_min_count = fasttext_model_min_count

        self.processes = processes
        self.seed = seed
        self.device = device
        self.verbosity = verbosity
        self.fasttext_source_embeddings = None
        self.fasttext_target_embeddings = None
        self.source_transform = lambda x: x
        self.target_transform = lambda x: x

    def load_auxiliary_embeddings(self):
        # This loads pre-trained fasttext embeddings
        # TODO: add the possibility to train fasttext embeddings from scratch similar to FOCUS
        fasttext_source_embeddings = WordEmbedding(load_embeddings(
            identifier=self.source_language_identifier,
            aligned=not self.align_with_bilingual_dictionary
        ))
        fasttext_target_embeddings = WordEmbedding(load_embeddings(
            identifier=self.target_language_identifier,
            aligned=not self.align_with_bilingual_dictionary
        ))
        min_dim = min(
            fasttext_source_embeddings.get_dimension(), fasttext_target_embeddings.get_dimension()
        )
        if fasttext_source_embeddings.get_dimension() != min_dim:
            fasttext.util.reduce_model(fasttext_source_embeddings.model, min_dim)
        if fasttext_target_embeddings.get_dimension() != min_dim:
            fasttext.util.reduce_model(fasttext_target_embeddings.model, min_dim)
        self.fasttext_source_embeddings = fasttext_source_embeddings
        self.fasttext_target_embeddings = fasttext_target_embeddings

    @staticmethod
    def compute_align_matrix_from_dictionary(
        source_embeddings, target_embeddings, dictionary
    ):
        correspondences = []

        for source_word, target_word in dictionary:
            for src_w in (source_word, source_word.lower(), source_word.title()):
                for trg_w in (target_word, target_word.lower(), target_word.title()):
                    src_id = source_embeddings.get_word_id(src_w)
                    trg_id = target_embeddings.get_word_id(trg_w)

                    if src_id != -1 and trg_id != -1:
                        correspondences.append(
                            [
                                source_embeddings.get_word_vector(src_w),
                                target_embeddings.get_word_vector(trg_w),
                            ]
                        )

        correspondences = np.array(correspondences)

        align_matrix, _ = orthogonal_procrustes(
            correspondences[:, 0], correspondences[:, 1]
        )

        return align_matrix

    def set_embedding_transformations(self):
        if self.align_with_bilingual_dictionary:
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

            align_matrix = self.compute_align_matrix_from_dictionary(
                source_embeddings=self.fasttext_source_embeddings,
                target_embeddings=self.fasttext_target_embeddings,
                dictionary=dictionary
            )
            self.source_transform = lambda matrix: matrix @ align_matrix
            self.target_transform = lambda x: x
        else:
            self.source_transform = lambda x: x
            self.target_transform = lambda x: x

    @staticmethod
    def get_subword_embeddings_in_word_embedding_space(
            tokenizer, model, max_n_word_vectors=None, use_subword_info=True, verbose=True
    ):
        words, freqs = model.get_words_and_freqs()

        if max_n_word_vectors is None:
            max_n_word_vectors = len(words)

        sources = {}
        embs_matrix = np.zeros((len(tokenizer), model.get_dimension()))

        if use_subword_info:
            if not model.has_subword_info():
                raise ValueError("Can not use subword info of model without subword info!")

            for i in range(len(tokenizer)):
                token = tokenizer.decode(i).strip()

                # `get_word_vector` returns zeros if not able to decompose
                embs_matrix[i] = model.get_word_vector(token)
        else:
            embs = {value: [] for value in tokenizer.get_vocab().values()}

            for i, word in tqdm(
                    enumerate(words[:max_n_word_vectors]),
                    total=max_n_word_vectors,
                    disable=not verbose,
            ):
                for tokenized in [
                    tokenizer.encode(word, add_special_tokens=False),
                    tokenizer.encode(" " + word, add_special_tokens=False),
                ]:
                    for token_id in set(tokenized):
                        embs[token_id].append(i)

            for i in range(len(embs_matrix)):
                if len(embs[i]) == 0:
                    continue

                weight = np.array([freqs[idx] for idx in embs[i]])
                weight = weight / weight.sum()

                vectors = [model.get_word_vector(words[idx]) for idx in embs[i]]

                sources[tokenizer.convert_ids_to_tokens([i])[0]] = embs[i]
                embs_matrix[i] = (np.stack(vectors) * weight[:, np.newaxis]).sum(axis=0)

        return embs_matrix, sources

    @staticmethod
    def create_target_embeddings(
            source_subword_embeddings,
            target_subword_embeddings,
            source_tokenizer,
            target_tokenizer,
            source_embeddings,
            target_embeddings,
            neighbors=10,
            temperature=0.1,
            verbose=True,
    ):
        def get_n_closest(token_id, similarities, top_k):
            if (target_subword_embeddings[token_id] == 0).all():
                return None

            best_indices = np.argpartition(similarities, -top_k)[-top_k:]
            best_tokens = source_tokenizer.convert_ids_to_tokens(best_indices)

            best = sorted(
                [
                    (token, similarities[idx])
                    for token, idx in zip(best_tokens, best_indices)
                ],
                key=lambda x: -x[1],
            )

            return best

        source_vocab = source_tokenizer.vocab

        batch_size = 1024
        n_matched = 0

        not_found = []
        sources = {}

        for i in tqdm(
                range(int(math.ceil(len(target_embeddings) / batch_size))), disable=not verbose
        ):
            start, end = (
                i * batch_size,
                min((i + 1) * batch_size, len(target_embeddings)),
            )

            similarities = cosine_similarity(
                target_subword_embeddings[start:end], source_subword_embeddings
            )
            for token_id in range(start, end):
                closest = get_n_closest(token_id, similarities[token_id - start], neighbors)

                if closest is not None:
                    # Calculate weighted average of source embeddings of k nearest neighbors in auxiliary embedding space
                    tokens, sims = zip(*closest)
                    weights = softmax(np.array(sims) / temperature, 0)

                    sources[target_tokenizer.convert_ids_to_tokens(token_id)] = (
                        tokens,
                        weights,
                        sims,
                    )

                    emb = np.zeros(target_embeddings.shape[1])

                    for i, close_token in enumerate(tokens):
                        emb += source_embeddings[source_vocab[close_token]] * weights[i]

                    target_embeddings[token_id] = emb

                    n_matched += 1
                else:
                    # Fall back on random initialization
                    not_found.append(target_tokenizer.convert_ids_to_tokens([token_id])[0])

        logging.info(
            f"Matching token found for {n_matched} of {len(target_embeddings)} tokens."
        )
        return target_embeddings, not_found, sources

    def initialize_embeddings(self, **kwargs):
        """
        Method that initializes the embeddings of a LM with a target tokenizer given a source LM.
        Leverages overlap between the source vocabulary and the target vocabulary to directly copy source embeddings
        and uses a helper model to initialize the rest.

        1. Load fast text embeddings for source language and target language
         --> load_auxiliary_embeddings() method
        2. Compute alignment for word embeddings using Orthogonal Procrustes method
         --> compute_align_matrix_from_dictionary() method to get self.source_transform and self.target_transform
        3. Compute subword embeddings for source language and target language -> sum of embeddings of all occurring n-grams
          subwords with no known n-grams are initialized to zero
          --> get_subword_embeddings_in_word_embedding_space() method
        4. Align source and target subword embeddings using alignment matrix from word embeddings
          --> apply self.source_transform and self.target_transform to source and target subword embeddings
        5. Compute cosine similarity between source and target subword embeddings and initialize target embeddings as
          weighted mean of k nearest source embeddings according to similarity in auxiliary embedding space

        :param kwargs: no kwargs

        :return: The initialized embedding matrix
        """
        target_embeddings = self.initialize_random_embeddings()     # Random initialization

        # Load FastText embeddings for source and target languages
        self.load_auxiliary_embeddings()
        # Compute alignment for word embeddings using Orthogonal Procrustes method
        self.set_embedding_transformations()

        # Compute subword embeddings for source and target languages
        source_subword_embeddings, source_subword_to_word = self.get_subword_embeddings_in_word_embedding_space(
            self.source_tokenizer, self.fasttext_source_embeddings, verbose=self.verbosity == "debug"
        )
        target_subword_embeddings, target_subword_to_word = self.get_subword_embeddings_in_word_embedding_space(
            self.target_tokenizer, self.fasttext_target_embeddings, verbose=self.verbosity == "debug"
        )
        # Align source and target subword embeddings using alignment matrix from word embeddings
        source_subword_embeddings = self.source_transform(source_subword_embeddings)
        target_subword_embeddings = self.target_transform(target_subword_embeddings)
        source_subword_embeddings /= (
                np.linalg.norm(source_subword_embeddings, axis=1)[:, np.newaxis] + 1e-8
        )
        target_subword_embeddings /= (
                np.linalg.norm(target_subword_embeddings, axis=1)[:, np.newaxis] + 1e-8
        )
        # Calculate target embeddings based on similarities in the auxiliary embedding space
        target_embeddings, not_found, sources = self.create_target_embeddings(
            source_subword_embeddings,
            target_subword_embeddings,
            self.source_tokenizer,
            self.target_tokenizer,
            self.source_embeddings.copy(),
            target_embeddings,
            neighbors=self.neighbors,
            temperature=self.temperature,
        )
        # not_found contains all tokens that could not be matched and whose embeddings were initialized randomly
        # sources contains the source tokens for each target token whose embeddings were used for initialization

        # Copy special tokens
        target_embeddings, overlapping_token_indices = self.copy_special_tokens(
            target_embeddings, return_overlapping_token_indices=True
        )

        return target_embeddings
