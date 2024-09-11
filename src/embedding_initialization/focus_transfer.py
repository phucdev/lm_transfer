import logging
import torch
import entmax
import numpy as np

from fastdist import fastdist
from typing import Literal, Optional
from tqdm.asyncio import tqdm
from torch import Tensor
from .tokenizer_transfer import OverlapTokenizerTransfer
from ..training.fasttext_embs import load_target_token_embedding

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class FocusTokenizerTransfer(OverlapTokenizerTransfer):
    def __init__(
            self,
            source_model_name_or_path: str,
            target_tokenizer_name_or_path: str,
            target_model_path: str = None,
            auxiliary_embedding_mode: Literal["fasttext-tokenlevel", "fasttext-wordlevel"] = "fasttext-tokenlevel",
            target_training_data_path: Optional[str] = None,
            fasttext_model_path: Optional[str] = None,
            language_identifier: Optional[str] = None,
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
        Class for transferring embeddings from one tokenizer to another using FOCUS method by Dobler & de Melo (2023).
        Code adapted from https://github.com/konstantinjdobler/focus/blob/main/src/deepfocus/focus.py
        From the paper:
        @inproceedings{dobler-de-melo-2023-focus,
            title = "{FOCUS}: Effective Embedding Initialization for Monolingual Specialization of Multilingual Models",
            author = "Dobler, Konstantin  and
              de Melo, Gerard",
            editor = "Bouamor, Houda  and
              Pino, Juan  and
              Bali, Kalika",
            booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
            month = dec,
            year = "2023",
            address = "Singapore",
            publisher = "Association for Computational Linguistics",
            url = "https://aclanthology.org/2023.emnlp-main.829",
            doi = "10.18653/v1/2023.emnlp-main.829",
            pages = "13440--13454",
        }
        :param source_model_name_or_path:
        :param target_tokenizer_name_or_path:
        :param target_model_path:
        :param auxiliary_embedding_mode ("fasttext-tokenlevel" or "fasttext-wordlevel"): The type of auxiliary embeddings to use. Defaults to "fasttext-tokenlevel".
        :param target_training_data_path: Path to a file containing lines of text in the target language for training a fasttext model. Only necessary if using `fasttext-tokenlevel`. Defaults to None.
        :param fasttext_model_path: Path to a pretrained fasttext model for the target tokenizer. Defaults to None.
        :param language_identifier: Two-letter language code for downloading pretrained fasttext word embeddings if using `fasttext-wordlevel`. Defaults to None.
        :param fasttext_model_epochs: Number of epochs if training a custom fasttext model. Defaults to 3.
        :param fasttext_model_dim: Dimension size if training a custom fasttext model. Defaults to 100.
        :param fasttext_model_min_count: Minimum number of occurrences for a token to be included if training a custom fasttext model. Defaults to 10.
        :param processes: Number of processes for parallelized workloads. Defaults to None, which uses heuristics based on available hardware.
        :param seed: Defaults to 42.
        :param device: Defaults to "cpu".
        :param verbosity: Defaults to "info".
        """
        super().__init__(source_model_name_or_path, target_tokenizer_name_or_path, target_model_path, **kwargs)
        self.auxiliary_embedding_mode = auxiliary_embedding_mode
        self.target_training_data_path = target_training_data_path
        self.fasttext_model_path = fasttext_model_path
        self.language_identifier = language_identifier
        self.fasttext_model_epochs = fasttext_model_epochs
        self.fasttext_model_dim = fasttext_model_dim
        self.fasttext_model_min_count = fasttext_model_min_count
        self.processes = processes
        self.seed = seed
        self.device = device
        self.verbosity = verbosity
        self.fasttext_model = None

        self.transfer_method = "focus"

    def save_parameters_to_dict(self):
        """
        Method that saves the parameters of the FocusTokenizerTransfer object to a dictionary.

        :return: A dictionary containing the parameters of the FocusTokenizerTransfer object.
        """
        parameters = super().save_parameters_to_dict()
        parameters.update({
            "auxiliary_embedding_mode": self.auxiliary_embedding_mode,
            "target_training_data_path": self.target_training_data_path,
            "fasttext_model_path": self.fasttext_model_path,
            "language_identifier": self.language_identifier,
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
        if self.auxiliary_embedding_mode == "fasttext-tokenlevel":
            if not self.target_training_data_path and not self.fasttext_model_path:
                raise ValueError(
                    "You need to provide a path to training data or pretrained fasttext model for fasttext-tokenlevel auxiliary embeddings."
                )
            fasttext_model = load_target_token_embedding(
                target_tokenizer=self.target_tokenizer,
                target_training_data_path=self.target_training_data_path,
                fasttext_model_path=self.fasttext_model_path,
                epochs=self.fasttext_model_epochs,
                dim=self.fasttext_model_dim,
                min_count=self.fasttext_model_min_count,
                processes=self.processes,
            )
        elif self.auxiliary_embedding_mode == "fasttext-wordlevel":
            if not self.language_identifier:
                raise ValueError(
                    "You need to provide a language identifier (e.g. de for German) for fasttext-wordlevel auxiliary embeddings."
                )
            fasttext_model = load_target_token_embedding(
                target_tokenizer=self.target_tokenizer,
                language_identifier=self.language_identifier,
                processes=self.processes,
            )
        else:
            fasttext_model = None
        return fasttext_model

    def initialize_embeddings(self, **kwargs):
        """
        Method that initializes the embeddings of a LM with a target tokenizer given a source LM.
        Leverages overlap between the source vocabulary and the target vocabulary to directly copy source embeddings
        and uses a helper model to initialize the rest.

        :param kwargs: no kwargs

        :return: The initialized embedding matrix
        """
        logger.info("(1/3) Load or train FastText embeddings for the target tokenizer...")
        self.fasttext_model = self.load_auxiliary_embeddings()

        logger.info("(2/3) Create random fallback matrix for target embeddings and copy source embeddings for overlapping tokens...")
        # The number of copied source embeddings may be lower than the number of overlapping tokens
        # if the FastText model does not contain the token
        target_embeddings = super().initialize_embeddings(**kwargs)

        logger.info("(3/3) Initialize the rest based on the overlap and the auxiliary embeddings with the FOCUS method")
        if self.missing_tokens:
            new_tokens_lst = []
            for token, missing_token_info in self.missing_tokens:
                if self.is_very_rare_token(token):
                    # If a token is not in the fast text model, we will fall back on the random initialization
                    continue
                else:
                    auxiliary_embedding = self.fasttext_model[token]
                    missing_token_info.auxiliary_embedding = auxiliary_embedding
                    new_tokens_lst.append(missing_token_info)
            # Filter very rare overlapping tokens with no auxiliary embeddings before using them for FOCUS
            overlapping_tokens_lst = [
                overlapping_token_info for t, overlapping_token_info in self.overlapping_tokens
                if overlapping_token_info.use_for_focus
            ]

            # Convert to numpy arrays for fastdist
            new_auxiliary_embedding_matrix = np.asarray(
                [t.auxiliary_embedding.tolist() for t in new_tokens_lst], dtype="float32")
            overlapping_auxiliary_embedding_matrix = np.asarray(
                [t.auxiliary_embedding.tolist() for t in overlapping_tokens_lst], dtype="float32",
            )

            logger.debug("Computing distance matrix...")
            similarity_matrix = fastdist.cosine_matrix_to_matrix(
                new_auxiliary_embedding_matrix,
                overlapping_auxiliary_embedding_matrix,
            )

            # Not needed anymore, save memory
            del new_auxiliary_embedding_matrix
            del overlapping_auxiliary_embedding_matrix

            logger.debug("Computing new embeddings...")

            # Do `torch.stack` once outside of loop to save time
            overlapping_src_embs = [torch.from_numpy(t.source_embedding).to(self.device) for t in overlapping_tokens_lst]
            overlapping_src_embs = torch.stack(overlapping_src_embs)

            for new_token_idx in tqdm(
                    range(len(new_tokens_lst)),
                    desc="FOCUS initialization...",
                    total=len(new_tokens_lst),
            ):
                overlapping_emb_weights: Tensor = entmax.sparsemax(
                    torch.from_numpy(similarity_matrix[new_token_idx]).to(self.device))

                # performance optimization
                mask = overlapping_emb_weights > 0.0
                masked_overlapping_emb_weights = overlapping_emb_weights[mask]
                masked_overlapping_src_embs = overlapping_src_embs[mask]

                weighted_src_embs = torch.mul(masked_overlapping_src_embs, masked_overlapping_emb_weights.unsqueeze(1))
                # It's a convex combination because the weights sum up to 1
                convex_combination = torch.sum(weighted_src_embs, dim=0)

                new_token_target_vocab_idx = new_tokens_lst[new_token_idx].target.id
                # Convert to numpy array for compatibility with the rest of the code
                target_embeddings[new_token_target_vocab_idx] = convex_combination.cpu().numpy()
                self.cleverly_initialized_tokens += 1
        logger.info(f"Initialized {self.cleverly_initialized_tokens}/{len(self.target_tokens)} tokens with the FOCUS method.")
        return target_embeddings
