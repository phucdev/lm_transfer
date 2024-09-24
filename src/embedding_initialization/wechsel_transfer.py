import logging
import os
import math
import tempfile
import multiprocessing
import nltk
import fasttext
import fasttext.util
import numpy as np

from pathlib import Path
from typing import Literal, Optional
from overrides import override
from scipy.linalg import orthogonal_procrustes
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from tqdm import tqdm
from functools import partial
from datasets import load_dataset
from .tokenizer_transfer import TokenizerTransfer
from ..utils.download_utils import download, decompress_archive as gunzip


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
    https://github.com/CPJKU/wechsel/blob/395e3d446ecc1f000aaf4dea2da2003d16203f0b/wechsel/__init__.py#L33-L75
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


def get_subword_embeddings_in_word_embedding_space(
        tokenizer, model, max_n_word_vectors=None, use_subword_info=True, verbose=True
):
    """
    Utility function to compute subword embeddings in the word embedding space.
    https://github.com/CPJKU/wechsel/blob/395e3d446ecc1f000aaf4dea2da2003d16203f0b/wechsel/__init__.py#L78-L125
    :param tokenizer:
    :param model:
    :param max_n_word_vectors:
    :param use_subword_info:
    :param verbose:
    :return:
    """
    words, freqs = model.get_words_and_freqs()

    if max_n_word_vectors is None:
        max_n_word_vectors = len(words)

    sources = {}
    embs_matrix = np.zeros((len(tokenizer), model.get_dimension()))

    if use_subword_info:
        if not model.has_subword_info():
            raise ValueError("Can not use subword info of model without subword info!")

        for i in range(len(tokenizer)):
            # Uses FastText OOV method to calculate subword embedding: decompose token into n-grams and sum
            # the embeddings of the n-grams
            token = tokenizer.decode(i).strip()

            # `get_word_vector` returns zeros if not able to decompose
            embs_matrix[i] = model.get_word_vector(token)
    else:
        # Mentioned in Appendix D of the paper for models without subword information:
        #  The embedding of a subword (target token) is defined as the average of the embeddings of words
        #  that contain the subword in their decomposition weighted by their word frequencies.
        embs = {value: [] for value in tokenizer.get_vocab().values()}
        # Go through each word in the FastText model
        for i, word in tqdm(
                enumerate(words[:max_n_word_vectors]),
                total=max_n_word_vectors,
                disable=not verbose,
        ):
            # Tokenize the word using the target tokenizer and append the FastText word index to the list of
            # indices for each token
            # In this case it is not clear why the token is tokenized twice
            for tokenized in [
                tokenizer.encode(word, add_special_tokens=False),
                tokenizer.encode(" " + word, add_special_tokens=False), # TODO: why is this done?
            ]:
                for token_id in set(tokenized):
                    embs[token_id].append(i)

        for i in range(len(embs_matrix)):
            # If the token is not in the FastText model, the embedding is set to zero
            if len(embs[i]) == 0:
                continue
            # Weight is the relative frequency of the word in the FastText model
            # Frequency is the number of occurrences of the word in the FastText model
            weight = np.array([freqs[idx] for idx in embs[i]])
            weight = weight / weight.sum()
            # For each target token get the vectors for all words whose decomposition contains the token
            vectors = [model.get_word_vector(words[idx]) for idx in embs[i]]
            # Sources contains the ids of the words whose decomposition contains the token
            sources[tokenizer.convert_ids_to_tokens([i])[0]] = embs[i]
            embs_matrix[i] = (np.stack(vectors) * weight[:, np.newaxis]).sum(axis=0)

    return embs_matrix, sources


def train_embeddings(
    text_path: str,
    language=None,
    tokenize_fn=None,
    encoding=None,
    epochs=20,
    **kwargs,
):
    """
    Utility function to train fastText embeddings.
    https://github.com/CPJKU/wechsel/blob/395e3d446ecc1f000aaf4dea2da2003d16203f0b/wechsel/__init__.py#L128-L181

    Args:
        text_path: path to a plaintext file to train on.
        language: language to use for Punkt tokenizer.
        tokenize_fn: function to tokenize the text (instead of using the Punkt tokenizer).
        encoding: file encoding.
        epochs: number of epochs to train for.
        kwargs: extra args to pass to `fasttext.train_unsupervised`.

    Returns:
        A fasttext model trained on text from the file.
    """
    if tokenize_fn is None:
        if language is None:
            raise ValueError(
                "`language` must not be `None` if no `tokenize_fn` is passed!"
            )

        tokenize_fn = partial(nltk.word_tokenize, language=language)

    if text_path.endswith(".txt"):
        dataset = load_dataset("text", data_files=text_path, split="train")
    if text_path.endswith(".json") or text_path.endswith(".jsonl"):
        dataset = load_dataset("json", data_files=text_path, split="train")
    else:
        raise ValueError(
            f"Unsupported file format: {text_path}. Only .txt and .json(l) files are supported."
        )

    dataset = dataset.map(
        lambda row: {"text": " ".join(tokenize_fn(row["text"]))},
        num_proc=multiprocessing.cpu_count(),
    )

    out_file = tempfile.NamedTemporaryFile("w+")
    for text in dataset["text"]:
        out_file.write(text + "\n")

    return fasttext.train_unsupervised(
        out_file.name,
        dim=100,
        neg=10,
        model="cbow",
        minn=5,
        maxn=5,
        epoch=epochs,
        **kwargs,
    )


def load_embeddings(identifier: str, verbose=True, aligned=False):
    """
    Utility function to download and cache embeddings from https://fasttext.cc.
    https://github.com/CPJKU/wechsel/blob/395e3d446ecc1f000aaf4dea2da2003d16203f0b/wechsel/__init__.py#L184-L211
    I added the option to load aligned embeddings from https://fasttext.cc/docs/en/aligned-vectors.html

    Args:
        identifier: 2-letter language code or path to a fasttext model.
        verbose: Whether to print download progress.
        aligned: Whether to download aligned embeddings.

    Returns:
        fastText model loaded from https://fasttext.cc.
    """
    if os.path.exists(identifier):
        path = Path(identifier)
    else:
        logging.info(
            f"Identifier '{identifier}' does not seem to be a path (file does not exist). Interpreting as language code."
        )
        if aligned:
            path = CACHE_DIR / "pretrained_fasttext" / f"wiki.{identifier}.align.vec"

            if not path.exists():
                path = download(
                    f"https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.{identifier}.align.vec",
                    CACHE_DIR / "pretrained_fasttext" / f"wiki.{identifier}.align.vec",
                    verbose=verbose,
                )
        else:
            path = CACHE_DIR / "pretrained_fasttext" / f"cc.{identifier}.300.bin"

            if not path.exists():
                path = download(
                    f"https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{identifier}.300.bin.gz",
                    CACHE_DIR / "pretrained_fasttext" / f"cc.{identifier}.300.bin.gz",
                    verbose=verbose,
                )
                path = gunzip(path)

    return fasttext.load_model(str(path))


class WechselTokenizerTransfer(TokenizerTransfer):
    def __init__(
            self,
            source_model_name_or_path: str,
            target_tokenizer_name_or_path: str,
            target_model_path: str = None,
            bilingual_dictionary: Optional[str] = None,
            source_language_identifier: Optional[str] = None,
            target_language_identifier: Optional[str] = None,
            align_strategy: str = "bilingual_dictionary",
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
            pages = "3992--4006"
        }
        :param source_model_name_or_path:
        :param target_tokenizer_name_or_path:
        :param target_model_path:
        :param bilingual_dictionary: Path to a bilingual dictionary. The dictionary must be of the form
                ```
                english_word1 \t target_word1\n
                english_word2 \t target_word2\n
                ...
                english_wordn \t target_wordn\n
                ```
                alternatively, pass only the language name, e.g. "german", to use a bilingual dictionary
                stored as part of WECHSEL (https://github.com/CPJKU/wechsel/tree/main/dicts).
        :param source_language_identifier: Two-letter language code for downloading pretrained fasttext word embeddings if using `fasttext-wordlevel`. Defaults to None.
        :param target_language_identifier: Two-letter language code for downloading pretrained fasttext word embeddings if using `fasttext-wordlevel`. Defaults to None.
        :param align_strategy: either of "bilingual_dictionary" or `None`.
                - If `None`, embeddings are treated as already aligned.
                - If "bilingual dictionary", a bilingual dictionary must be passed
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
        self.bilingual_dictionary = bilingual_dictionary
        self.source_language_identifier = source_language_identifier
        self.target_language_identifier = target_language_identifier
        self.align_strategy = align_strategy
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
        self.transfer_method = "wechsel"

        # Adapts: https://github.com/CPJKU/wechsel/blob/395e3d446ecc1f000aaf4dea2da2003d16203f0b/wechsel/__init__.py#L350-L422
        fasttext_source_embeddings = WordEmbedding(load_embeddings(
            identifier=self.source_language_identifier,
            aligned=not self.align_strategy
        ))
        fasttext_target_embeddings = WordEmbedding(load_embeddings(
            identifier=self.target_language_identifier,
            aligned=not self.align_strategy
        ))
        min_dim = min(
            fasttext_source_embeddings.get_dimension(), fasttext_target_embeddings.get_dimension()
        )
        if fasttext_source_embeddings.get_dimension() != min_dim:
            fasttext.util.reduce_model(fasttext_source_embeddings.model, min_dim)
        if fasttext_target_embeddings.get_dimension() != min_dim:
            fasttext.util.reduce_model(fasttext_target_embeddings.model, min_dim)

        if align_strategy == "bilingual_dictionary":
            if bilingual_dictionary is None:
                raise ValueError(
                    "`bilingual_dictionary_path` must not be `None` if `align_strategy` is 'bilingual_dictionary'."
                )
            if not os.path.exists(self.bilingual_dictionary):
                bilingual_dictionary = download(
                    f"https://raw.githubusercontent.com/CPJKU/wechsel/main/dicts/data/{self.bilingual_dictionary}.txt",
                    CACHE_DIR / f"{self.bilingual_dictionary}.txt",
                )
            dictionary = []

            for line in open(bilingual_dictionary, "r"):
                line = line.strip()
                try:
                    source_word, target_word = line.split("\t")
                except ValueError:
                    source_word, target_word = line.split()
                dictionary.append((source_word, target_word))

            align_matrix = self.compute_align_matrix_from_dictionary(
                source_embeddings=fasttext_source_embeddings,
                target_embeddings=fasttext_target_embeddings,
                dictionary=dictionary
            )
            self.source_transform = lambda matrix: matrix @ align_matrix
            self.target_transform = lambda x: x
        elif align_strategy is None:
            self.source_transform = lambda x: x
            self.target_transform = lambda x: x
        else:
            raise ValueError(f"Unknown align strategy: {align_strategy}.")

        self.fasttext_source_embeddings = fasttext_source_embeddings
        self.fasttext_target_embeddings = fasttext_target_embeddings

    @override
    def save_parameters_to_dict(self):
        """
        Method to save all parameters to a dictionary for saving the model configuration
        :return:
        """
        parameters = super().save_parameters_to_dict()
        parameters.update({
            "bilingual_dictionary_path": self.bilingual_dictionary,
            "source_language_identifier": self.source_language_identifier,
            "target_language_identifier": self.target_language_identifier,
            "align_with_bilingual_dictionary": self.align_strategy,
            "use_subword_info": self.use_subword_info,
            "max_n_word_vectors": self.max_n_word_vectors,
            "neighbors": self.neighbors,
            "temperature": self.temperature,
            "auxiliary_embedding_mode": self.auxiliary_embedding_mode,
            "source_training_data_path": self.source_training_data_path,
            "target_training_data_path": self.target_training_data_path,
            "source_fasttext_model_path": self.source_fasttext_model_path,
            "target_fasttext_model_path": self.target_fasttext_model_path,
            "fasttext_model_epochs": self.fasttext_model_epochs,
            "fasttext_model_dim": self.fasttext_model_dim,
            "fasttext_model_min_count": self.fasttext_model_min_count,
            "processes": self.processes,
            "seed": self.seed,
            "device": self.device,
            "verbosity": self.verbosity
        })
        return parameters

    def load_auxiliary_embeddings(self):
        """
        Method to load fasttext embeddings for source and target languages
        https://github.com/CPJKU/wechsel/blob/395e3d446ecc1f000aaf4dea2da2003d16203f0b/wechsel/__init__.py#L376-L385
        Moved to separate function so we only load the fasttext embeddings once for the input embeddings and reuse them
        for the output embeddings
        """
        # This loads pre-trained fasttext embeddings


    @staticmethod
    def compute_align_matrix_from_dictionary(
        source_embeddings, target_embeddings, dictionary
    ):
        """
        Method to compute the alignment matrix from a bilingual dictionary using the Orthogonal Procrustes method.
        https://github.com/CPJKU/wechsel/blob/395e3d446ecc1f000aaf4dea2da2003d16203f0b/wechsel/__init__.py#L323--L348
        :param source_embeddings:
        :param target_embeddings:
        :param dictionary:
        :return:
        """
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

    def create_target_embeddings(
            self,
            source_subword_embeddings,
            target_subword_embeddings,
            source_tokenizer,
            target_tokenizer,
            source_matrix,
            neighbors=10,
            temperature=0.1,
            verbose=True,
    ):
        """
        Method to initialize the target embeddings based on the source embeddings and auxiliary subword embeddings.
        https://github.com/CPJKU/wechsel/blob/395e3d446ecc1f000aaf4dea2da2003d16203f0b/wechsel/__init__.py#L214-L311
        Minor modification to track cleverly initialized token embeddings
        :param source_subword_embeddings:
        :param target_subword_embeddings:
        :param source_tokenizer:
        :param target_tokenizer:
        :param source_matrix:
        :param neighbors:
        :param temperature:
        :param verbose:
        :return:
        """
        def get_n_closest(token_id, similarities, top_k):
            if np.asarray(target_subword_embeddings[token_id] == 0).all():
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

        target_matrix = np.zeros(
            (len(target_tokenizer), source_matrix.shape[1]), dtype=source_matrix.dtype
        )

        mean, std = (
            source_matrix.mean(0),
            source_matrix.std(0),
        )

        random_fallback_matrix = np.random.RandomState(1234).normal(
            mean, std, (len(target_tokenizer.vocab), source_matrix.shape[1])
        )

        batch_size = 1024
        n_matched = 0

        not_found = []
        sources = {}

        self.cleverly_initialized_tokens = 0

        for i in tqdm(
                range(int(math.ceil(len(target_matrix) / batch_size))), disable=not verbose
        ):
            start, end = (
                i * batch_size,
                min((i + 1) * batch_size, len(target_matrix)),
            )

            similarities = cosine_similarity(
                target_subword_embeddings[start:end], source_subword_embeddings
            )
            for token_id in range(start, end):
                closest = get_n_closest(token_id, similarities[token_id - start], neighbors)

                if closest is not None:
                    tokens, sims = zip(*closest)
                    weights = softmax(np.array(sims) / temperature, 0)

                    sources[target_tokenizer.convert_ids_to_tokens(token_id)] = (
                        tokens,
                        weights,
                        sims,
                    )

                    emb = np.zeros(target_matrix.shape[1])

                    for i, close_token in enumerate(tokens):
                        emb += source_matrix[source_vocab[close_token]] * weights[i]

                    target_matrix[token_id] = emb

                    n_matched += 1
                    self.cleverly_initialized_tokens += 1
                else:
                    target_matrix[token_id] = random_fallback_matrix[token_id]
                    not_found.append(target_tokenizer.convert_ids_to_tokens([token_id])[0])

        self.overlap_based_initialized_tokens = 0
        for token in source_tokenizer.special_tokens_map.values():
            if isinstance(token, str):
                token = [token]

            for t in token:
                if t in target_tokenizer.vocab:
                    target_matrix[target_tokenizer.vocab[t]] = source_matrix[
                        source_tokenizer.vocab[t]
                    ]
                    self.overlap_based_initialized_tokens += 1
        self.cleverly_initialized_tokens += self.overlap_based_initialized_tokens

        return target_matrix, not_found, sources

    @override
    def initialize_embeddings(self, source_embeddings, **kwargs):
        """
        Method that initializes the embeddings of a LM with a target tokenizer given a source LM and auxiliary
        FastText embeddings.
        Adapts: https://github.com/CPJKU/wechsel/blob/395e3d446ecc1f000aaf4dea2da2003d16203f0b/wechsel/__init__.py#L424-L493

        Compute subword embeddings for source language and target language -> sum of embeddings of all occurring n-grams
          subwords with no known n-grams are initialized to zero
          --> get_subword_embeddings_in_word_embedding_space() method
        Align source and target subword embeddings using alignment matrix from word embeddings
          --> apply self.source_transform and self.target_transform to source and target subword embeddings
        Compute cosine similarity between source and target subword embeddings and initialize target embeddings as
          weighted mean of k nearest source embeddings according to similarity in auxiliary embedding space

        :param source_embeddings: The embeddings of the source language (either the input embeddings or the output embeddings)
        :param kwargs: no kwargs

        :return: The initialized embedding matrix
        """
        logger.info("Compute subword embeddings for source and target languages")
        source_subword_embeddings, source_subword_to_word = get_subword_embeddings_in_word_embedding_space(
            tokenizer=self.source_tokenizer,
            model=self.fasttext_source_embeddings,
            use_subword_info=self.use_subword_info,
            max_n_word_vectors=self.max_n_word_vectors,
            verbose=self.verbosity == "debug"
        )
        target_subword_embeddings, target_subword_to_word = get_subword_embeddings_in_word_embedding_space(
            tokenizer=self.target_tokenizer,
            model=self.fasttext_target_embeddings,
            use_subword_info=self.use_subword_info,
            max_n_word_vectors=self.max_n_word_vectors,
            verbose=self.verbosity == "debug"
        )
        logger.info("Align source and target subword embeddings using alignment matrix from word embeddings")
        source_subword_embeddings = self.source_transform(source_subword_embeddings)
        target_subword_embeddings = self.target_transform(target_subword_embeddings)
        source_subword_embeddings /= (
                np.linalg.norm(source_subword_embeddings, axis=1)[:, np.newaxis] + 1e-8
        )
        target_subword_embeddings /= (
                np.linalg.norm(target_subword_embeddings, axis=1)[:, np.newaxis] + 1e-8
        )
        logger.info("Calculate target embeddings based on similarities in the auxiliary embedding space")
        target_embeddings, not_found, sources = self.create_target_embeddings(
            source_subword_embeddings,
            target_subword_embeddings,
            self.source_tokenizer,
            self.target_tokenizer,
            source_embeddings.copy(),
            neighbors=self.neighbors,
            temperature=self.temperature,
        )
        # not_found contains all tokens that could not be matched and whose embeddings were initialized randomly
        # sources contains the source tokens for each target token whose embeddings were used for initialization

        logger.info(f"Initialized {self.cleverly_initialized_tokens}/{len(self.target_tokens)} tokens using WECHSEL method.")
        return target_embeddings
