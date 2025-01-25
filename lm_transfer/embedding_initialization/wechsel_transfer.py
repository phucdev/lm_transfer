import logging
import os
import math
import tempfile
import multiprocessing
import nltk
import gc
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
from gensim.models import KeyedVectors
from lm_transfer.embedding_initialization.tokenizer_transfer import OverlapTokenizerTransfer
from lm_transfer.utils.download_utils import download, decompress_archive as gunzip
from lm_transfer.utils.utils import load_matrix


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


def load_vec(file_path: str, maxload: int = None, batch_size: int = 10000, reload_as_kv: bool = True):
    """
    Utility function to load fastText embeddings from .vec file.
    :param file_path: Path to the .vec file.
    :param maxload: Maximum number of vectors to load.
    :param batch_size: Batch size for loading vectors.
    :param reload_as_kv: Whether to export to .kv format and reload the vectors with memory mapping.
    :return: Dictionary containing the word vectors.
    """
    if reload_as_kv:
        kv_path = str(Path(file_path).with_suffix(".kv"))
        if Path(kv_path).exists():
            return KeyedVectors.load(kv_path, mmap="r")
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        # Read the header
        first_line = f.readline()
        if len(first_line.split()) == 2:
            num_vectors, vector_size = map(int, first_line.split())
        else:
            # Handle cases without a header
            num_vectors = None
            vector_size = len(first_line.strip().split()) - 1
            f.seek(0)
        # Create KeyedVectors instance
        kv = KeyedVectors(vector_size)
        vocab = []
        vectors = []
        for idx, line in tqdm(enumerate(f), desc="Loading vectors", total=num_vectors):
            tokens = line.rstrip().split(' ')
            word = tokens[0]
            vector = np.array(tokens[1:], dtype=np.float32)

            # Store vectors and vocab in batches
            vocab.append(word)
            vectors.append(vector)

            # Once the batch size is reached, add the batch of vectors to KeyedVectors
            if (idx + 1) % batch_size == 0:
                kv.add_vectors(vocab, np.array(vectors))
                vocab = []
                vectors = []

            if maxload is not None and idx + 1 >= maxload:
                break

        # Add the remaining vectors to KeyedVectors
        if vocab:
            kv.add_vectors(vocab, np.array(vectors))

    if reload_as_kv:
        # By exporting to .kv format and reloading with memory mapping, we can save memory
        kv_path = str(Path(file_path).with_suffix(".kv"))
        kv.save(kv_path)
        del kv
        gc.collect()
        kv = KeyedVectors.load(kv_path, mmap="r")

    return kv


class WordEmbedding:
    """
    Uniform interface to fastText models and gensim Word2Vec models.
    https://github.com/CPJKU/wechsel/blob/395e3d446ecc1f000aaf4dea2da2003d16203f0b/wechsel/__init__.py#L33-L75
    """

    def __init__(self, model, aligned_vectors=None):
        self.model = model
        self.aligned_vectors = aligned_vectors

        if isinstance(model, fasttext.FastText._FastText):
            if isinstance(aligned_vectors, KeyedVectors):
                self.kind = "fasttext_aligned"
                # Store the words and frequencies from original FastText model and release the model
                # to save memory
                self.words_and_freqs = self.model.get_words(include_freq=True, on_unicode_error="ignore")
                self.model = None
                del model
                gc.collect()
            else:
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
        elif self.kind == "fasttext_aligned":
            return self.words_and_freqs

    def get_dimension(self):
        if self.kind == "fasttext":
            return self.model.get_dimension()
        elif self.kind == "word2vec":
            return self.model.wv.vector_size
        elif self.kind == "fasttext_aligned":
            return self.aligned_vectors.vector_size

    def get_word_vector(self, word):
        if self.kind == "fasttext":
            return self.model.get_word_vector(word)
        elif self.kind == "word2vec":
            return self.model.wv[word]
        elif self.kind == "fasttext_aligned":
            return self.aligned_vectors[word]

    def get_word_id(self, word):
        if self.kind == "fasttext":
            return self.model.get_word_id(word)
        elif self.kind == "word2vec":
            return self.model.wv.key_to_index.get(word, -1)
        elif self.kind == "fasttext_aligned":
            return self.aligned_vectors.key_to_index.get(word, -1)


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
            # Tokenize the word both without and with a leading space to capture different tokenizations depending on
            # the word's position in text (e.g., at the beginning vs. in the middle of a sentence) and append the
            # FastText word index to the list of indices for each token
            for tokenized in [
                tokenizer.encode(word, add_special_tokens=False),
                tokenizer.encode(" " + word, add_special_tokens=False),
            ]:
                for token_id in set(tokenized):
                    embs[token_id].append(i)

        for i in range(len(embs_matrix)):
            # If the token is not in the FastText model, the embedding is set to zero -> random initialization later
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
    elif text_path.endswith(".json") or text_path.endswith(".jsonl"):
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


def load_embeddings(identifier: str, verbose=True, emb_type="crawl", cache_dir=CACHE_DIR):
    """
    Utility function to download and cache embeddings from https://fasttext.cc.
    https://github.com/CPJKU/wechsel/blob/395e3d446ecc1f000aaf4dea2da2003d16203f0b/wechsel/__init__.py#L184-L211
    I added the option to load aligned embeddings from https://fasttext.cc/docs/en/aligned-vectors.html
    and modified the function to return a `WordEmbedding` object.
    I also added a cache_dir argument to specify the directory to cache the embeddings.

    Args:
        identifier: 2-letter language code or path to a fasttext model.
        verbose: Whether to print download progress.
        emb_type: Which type of fastText embeddings to load.
        cache_dir: Directory to cache the embeddings.

    Returns:
        `WordEmbedding` object.
    """
    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)
    if emb_type == "crawl":
        if os.path.exists(identifier):
            path = Path(identifier)
        else:
            logging.info(
                f"Identifier '{identifier}' does not seem to be a path (file does not exist). Interpreting as language code."
            )
            path = cache_dir / "pretrained_fasttext" / f"cc.{identifier}.300.bin"

            if not path.exists():
                path = download(
                    f"https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{identifier}.300.bin.gz",
                    cache_dir / "pretrained_fasttext" / f"cc.{identifier}.300.bin.gz",
                    verbose=verbose,
                )
                path = gunzip(path)
        return WordEmbedding(fasttext.load_model(str(path)))
    elif emb_type == "wiki" or emb_type == "aligned":
        # In the aligned case we need to load the original model as well to obtain the frequencies
        model_path = cache_dir / "pretrained_fasttext" / f"wiki.{identifier}" / f"wiki.{identifier}.bin"
        if not model_path.exists():
            archive_path = download(
                f"https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.{identifier}.zip",
                cache_dir / "pretrained_fasttext" / f"wiki.{identifier}.zip",
                verbose=verbose,
            )
            _ = gunzip(archive_path)
        model = fasttext.load_model(str(model_path))

        if emb_type == "aligned":
            vector_path = cache_dir / "pretrained_fasttext" / f"wiki.{identifier}.align.vec"
            if not vector_path.exists():
                vector_path = download(
                    f"https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.{identifier}.align.vec",
                    cache_dir / "pretrained_fasttext" / f"wiki.{identifier}.align.vec",
                    verbose=verbose,
                )
            aligned_vectors = load_vec(str(vector_path))
            return WordEmbedding(model, aligned_vectors)
        else:
            return WordEmbedding(model)
    else:
        raise ValueError(f"Unknown embedding type: {emb_type}.")


class WechselTokenizerTransfer(OverlapTokenizerTransfer):
    def __init__(
            self,
            source_model_name_or_path: str,
            target_tokenizer_name_or_path: str,
            target_model_path: str = None,
            bilingual_dictionary: Optional[str] = None,
            source_language_identifier: Optional[str] = None,
            target_language_identifier: Optional[str] = None,
            emb_type: str = "crawl",
            align_strategy: Optional[str] = "bilingual_dictionary",
            align_matrix_path: Optional[str] = None,
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
            cache_dir: str = str(CACHE_DIR),
            leverage_overlap: bool = False,
            overwrite_with_overlap: bool = False,
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
        :param emb_type: Type of embeddings to use for the target language. Defaults to "crawl".
        :param align_strategy: either of "bilingual_dictionary" or `None`.
                - If `None`, embeddings are treated as already aligned.
                - If "bilingual dictionary", a bilingual dictionary must be passed
                    which will be used to align the embeddings using the Orthogonal Procrustes method.
                - If "align matrix", a matrix must be passed which will be used to align the embeddings.
        :param align_matrix_path: Path to alignment matrix to align the embeddings if `align_strategy` is "align matrix". Defaults to None.
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
        :param cache_dir: Directory to cache fasttext models and bilingual dictionary. Defaults to `~/.cache/wechsel`.
        :param leverage_overlap: Whether to leverage overlap between source and target tokens for initialization. Defaults to False.
        :param overwrite_with_overlap: Whether to overwrite the initialized embeddings from WECHSEL with the overlap-based initialization. Defaults to False.
        :param processes: Number of processes for parallelized workloads. Defaults to None, which uses heuristics based on available hardware.
        :param seed: Defaults to 42.
        :param device: Defaults to "cpu".
        :param verbosity: Defaults to "info".
        """
        super().__init__(source_model_name_or_path, target_tokenizer_name_or_path, target_model_path, **kwargs)
        self.bilingual_dictionary = bilingual_dictionary
        self.source_language_identifier = source_language_identifier
        self.target_language_identifier = target_language_identifier
        self.emb_type = emb_type
        self.align_strategy = align_strategy
        self.align_matrix_path = align_matrix_path
        self.use_subword_info = use_subword_info
        self.max_n_word_vectors = max_n_word_vectors
        self.neighbors = neighbors
        self.temperature = temperature
        self.auxiliary_embedding_mode = auxiliary_embedding_mode
        self.source_training_data_path = source_training_data_path
        self.target_training_data_path = target_training_data_path
        self.source_fasttext_model_path = source_fasttext_model_path
        self.target_fasttext_model_path = target_fasttext_model_path
        self.fasttext_model_epochs = fasttext_model_epochs
        self.fasttext_model_dim = fasttext_model_dim
        self.fasttext_model_min_count = fasttext_model_min_count
        self.cache_dir = cache_dir
        self.leverage_overlap = leverage_overlap
        self.overwrite_with_overlap = overwrite_with_overlap
        self.processes = processes
        self.seed = seed
        self.device = device
        self.verbosity = verbosity
        self.transfer_method = "wechsel"

        # Adapts: https://github.com/CPJKU/wechsel/blob/395e3d446ecc1f000aaf4dea2da2003d16203f0b/wechsel/__init__.py#L350-L422
        effective_emb_type = "aligned" if self.align_strategy is None else self.emb_type
        logger.info(f"Loading fastText embeddings ({effective_emb_type}) for source language ({self.source_language_identifier})...")
        fasttext_source_embeddings = load_embeddings(
            identifier=self.source_language_identifier,
            emb_type=effective_emb_type,
            cache_dir=Path(cache_dir)
        )
        logger.info(f"Loading fastText embeddings ({effective_emb_type}) for target language ({self.target_language_identifier})...")
        fasttext_target_embeddings = load_embeddings(
            identifier=self.target_language_identifier,
            emb_type=effective_emb_type,
            cache_dir=Path(cache_dir)
        )
        min_dim = min(
            fasttext_source_embeddings.get_dimension(), fasttext_target_embeddings.get_dimension()
        )
        if fasttext_source_embeddings.get_dimension() != min_dim:
            fasttext.util.reduce_model(fasttext_source_embeddings.model, min_dim)
        if fasttext_target_embeddings.get_dimension() != min_dim:
            fasttext.util.reduce_model(fasttext_target_embeddings.model, min_dim)

        # Get transformation matrix to align source embeddings with target embeddings
        if align_strategy == "bilingual_dictionary":
            if bilingual_dictionary is None:
                raise ValueError(
                    "`bilingual_dictionary_path` must not be `None` if `align_strategy` is 'bilingual_dictionary'."
                )
            if not os.path.exists(self.bilingual_dictionary):
                bilingual_dictionary = download(
                    f"https://raw.githubusercontent.com/CPJKU/wechsel/main/dicts/data/{self.bilingual_dictionary}.txt",
                    Path(self.cache_dir) / f"{self.bilingual_dictionary}.txt",
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
        elif self.align_strategy == "align matrix":
            if self.align_matrix_path is None:
                raise ValueError(
                    "`align_matrix_path` must not be `None` if `align_strategy` is 'align matrix'."
                )
            align_matrix = load_matrix(self.align_matrix_path)
            # Following Joulin et al. (2018) https://github.com/facebookresearch/fastText/blob/main/alignment/align.py#L142
            self.source_transform = lambda matrix: np.dot(matrix, align_matrix.T)
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
        if not self.leverage_overlap:
            parameters.pop("exact_match_all")
            parameters.pop("match_symbols")
            parameters.pop("fuzzy_match_all")
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
            "cache_dir": self.cache_dir,
            "leverage_overlap": self.leverage_overlap,
            "overwrite_with_overlap": self.overwrite_with_overlap,
            "processes": self.processes,
            "seed": self.seed,
            "device": self.device,
            "verbosity": self.verbosity
        })
        return parameters

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
                # If the subword embedding could not be calculated for the target token, it is set to zero
                # and its actual embedding will be initialized randomly later
                # This is the case when not using subword information and the token is not part of any word in the
                # FastText model
                return None

            best_indices = np.argpartition(similarities, -top_k)[-top_k:]
            best_tokens = source_tokenizer.convert_ids_to_tokens(best_indices)

            best = sorted(
                [
                    (token, idx, similarities[idx])
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
                    tokens, token_indices, sims = zip(*closest)
                    weights = softmax(np.array(sims) / temperature, 0)

                    sources[target_tokenizer.convert_ids_to_tokens(token_id)] = (
                        tokens,
                        token_indices,
                        weights,
                        sims,
                    )

                    emb = np.zeros(target_matrix.shape[1])

                    for j, close_token in enumerate(tokens):
                        emb += source_matrix[source_vocab[close_token]] * weights[j]

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

        num_sources = len(sources)
        assert num_sources == n_matched, f"Number of sources ({num_sources}) does not match number of matched tokens ({n_matched})."

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
        # sources contains the source tokens, used weights and similarities for each target token whose
        # embeddings were used for WECHSEL initialization

        if self.leverage_overlap:
            self.overlap_based_initialized_tokens = 0
            # Optional: Get overlapping tokens and missing tokens
            overlapping_tokens, missing_tokens = self.get_overlapping_tokens()
            overlapping_token_indices = []
            if not overlapping_tokens:
                raise ValueError("No overlapping tokens found")
            # Copy source embeddings for overlapping tokens
            for token, overlapping_token_info in tqdm(overlapping_tokens,
                                                      desc="Initialize target embeddings for overlapping tokens"):
                if token in not_found or self.overwrite_with_overlap:
                    target_token_idx = overlapping_token_info.target.id
                    source_token_idx = overlapping_token_info.source[0].id
                    target_embeddings[target_token_idx] = source_embeddings[source_token_idx]
                    overlapping_token_indices.append(target_token_idx)
                    self.overlap_based_initialized_tokens += 1
                    if token in not_found:
                        not_found.remove(token)
                    sources[token] = (
                        [overlapping_token_info.source[0].native_form],
                        [overlapping_token_info.source[0].id],
                        [1.0],  # weight
                        [1.0]   # similarity
                    )
            self.cleverly_initialized_tokens = len(self.target_tokens) - len(not_found)
        else:
            self.cleverly_initialized_tokens += self.overlap_based_initialized_tokens

        self.sources = sources

        logger.info(f"Initialized {self.cleverly_initialized_tokens}/{len(self.target_tokens)} tokens using WECHSEL method.")
        return target_embeddings
