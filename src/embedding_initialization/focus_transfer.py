import logging
import torch
import entmax
import numpy as np
import transformers

from fastdist import fastdist
from typing import Literal, Optional, Union, Dict

from overrides import override
from tqdm.asyncio import tqdm
from torch import Tensor
from .tokenizer_transfer import OverlapTokenizerTransfer
from .overlap_utils import get_overlapping_tokens, NewToken, OverlappingToken
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
            extend_tokenizer_name_or_path: str = None,
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
        :param extend_tokenizer_name_or_path: If extending a tokenizer instead of vocabulary replacement, this should be the tokenizer that was used to extend the `source_tokenizer` (i.e. a target language specific tokenizer). The `target_tokenizer` should be the *extended* tokenizer. Defaults to None.
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

        if extend_tokenizer_name_or_path:
            extend_tokenizer = transformers.AutoTokenizer.from_pretrained(extend_tokenizer_name_or_path)
        else:
            extend_tokenizer = None

        ###########################################################
        # 1. Load auxiliary embedding model for target vocabulary #
        ###########################################################
        if auxiliary_embedding_mode == "fasttext-tokenlevel":
            if not target_training_data_path and not fasttext_model_path:
                raise ValueError(
                    "You need to provide a path to training data or pretrained fasttext model for fasttext-tokenlevel "
                    "auxiliary embeddings."
                )
            fasttext_model = load_target_token_embedding(
                target_tokenizer=extend_tokenizer or self.target_tokenizer,
                target_training_data_path=target_training_data_path,
                fasttext_model_path=fasttext_model_path,
                epochs=fasttext_model_epochs,
                dim=fasttext_model_dim,
                min_count=fasttext_model_min_count,
                processes=processes,
            )
        elif auxiliary_embedding_mode == "fasttext-wordlevel":
            if not language_identifier:
                raise ValueError(
                    "You need to provide a language identifier (e.g. de for German) for fasttext-wordlevel auxiliary "
                    "embeddings."
                )
            fasttext_model = load_target_token_embedding(
                target_tokenizer=extend_tokenizer or self.target_tokenizer,
                language_identifier=language_identifier,
                processes=processes,
            )
        else:
            fasttext_model = None
        self.fasttext_model = fasttext_model

        #################################################################
        # 2. Get overlapping tokens between source and target tokenizer #
        #################################################################
        overlapping_tokens, new_tokens = get_overlapping_tokens(
            target_tokenizer=self.target_tokenizer,
            source_tokenizer=self.source_tokenizer,
            match_symbols=self.match_symbols,
            exact_match_all=self.exact_match_all,
            fuzzy_match_all=self.fuzzy_match_all,
        )

        # Sort to ensure same order every time (especially important when executing on multiple ranks)
        sorted_overlapping_tokens = sorted(overlapping_tokens.items(), key=lambda x: x[1].target.id)
        sorted_new_tokens = sorted(new_tokens.items(), key=lambda x: x[1].target.id)
        logger.debug(f"Found {len(sorted_overlapping_tokens)} overlapping tokens.")

        ##########################################################
        # 3. Clean overlap + get auxiliary embeddings for tokens #
        ##########################################################
        # Clean overlapping tokens
        extend_tokenizer_vocab = extend_tokenizer.get_vocab() if extend_tokenizer else None
        very_rare_overlapping_tokens = []

        for token, overlapping_token_info in tqdm(
                sorted_overlapping_tokens,
                desc="Populating auxiliary embeddings for overlapping token...",
                leave=False,
        ):
            # The following code is commented out because we have to do this separately for the input and
            # output embeddings.
            # embs_lst = [self.source_embeddings[s.id] for s in overlapping_token_info.source]
            # overlapping_tokens[token].source_embedding = embs_lst[0]
            #
            # if len(embs_lst) > 1:
            #     logger.warning(
            #         f"{token} has multiple source embeddings (using first): {[s.native_form for s in overlapping_token_info.source][:min(5, len(embs_lst))]}"
            #     )

            # Filter some tokens so that they are not used for FOCUS
            if extend_tokenizer and not extend_tokenizer_vocab.get(overlapping_token_info.target.native_form):
                # if extending, we do not want to use tokens that are not in the language-specific tokenizer
                overlapping_tokens[token].use_for_focus = False
            elif self.is_very_rare_token(token, fasttext_model):
                very_rare_overlapping_tokens.append(token)
                overlapping_tokens[token].use_for_focus = False
            else:
                overlapping_tokens[token].auxiliary_embedding = fasttext_model[token]

        logger.debug(
            f"Pruned {len(very_rare_overlapping_tokens)} overlapping tokens because they do not have an auxiliary "
            f"embedding: {very_rare_overlapping_tokens}"
        )

        # Clean new tokens, mark "bad" tokens for random init
        random_init_new_tokens: list[NewToken] = []
        for token, new_token_info in tqdm(
                sorted_new_tokens,
                desc="Populating auxiliary embeddings for non-overlapping token...",
                leave=False,
        ):
            if self.is_very_rare_token(new_token_info.target.native_form, fasttext_model):
                random_init_new_tokens.append(new_token_info)
                del new_tokens[token]
            else:
                new_token_info.auxiliary_embedding = fasttext_model[token]

        logger.debug(f"Will initialize {len(random_init_new_tokens)} new tokens randomly.")
        logger.debug(f"{[t.target.native_form for t in random_init_new_tokens]}")
        self.overlapping_tokens = overlapping_tokens
        self.new_tokens = new_tokens
        self.random_init_new_tokens = random_init_new_tokens
        self.sorted_overlapping_tokens = sorted_overlapping_tokens
        self.sorted_new_tokens = sorted_new_tokens
        # We set the seed here instead of in initialize_embeddings
        # otherwise we would get the same random initialization for the input embeddings and the output embeddings
        self.gen = torch.Generator(device=self.device).manual_seed(self.seed)

    @override
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

    @staticmethod
    def focus_additional_token_initialization(
            overlapping_tokens: Dict[str, OverlappingToken],
            new_tokens: Dict[str, NewToken],
            source_embeddings: Tensor,
            target_embeddings: Tensor,
            device: Union[torch.device, str, None] = None,
    ):
        # Convert to lists to ensure same order (`.values()` might not guarantee same order every time)
        new_tokens_lst = list(new_tokens.values())
        overlapping_tokens_lst = list(overlapping_tokens.values())

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
        overlapping_src_embs = [source_embeddings[t.source[0].id] for t in overlapping_tokens_lst]
        overlapping_src_embs = torch.stack(overlapping_src_embs)

        for new_token_idx in tqdm(
                range(len(new_tokens_lst)),
                desc="FOCUS initialization...",
                total=len(new_tokens_lst),
        ):
            overlapping_emb_weights: Tensor = entmax.sparsemax(
                torch.from_numpy(similarity_matrix[new_token_idx]).to(device))

            # performance optimization
            mask = overlapping_emb_weights > 0.0
            masked_overlapping_emb_weights = overlapping_emb_weights[mask]
            masked_overlapping_src_embs = overlapping_src_embs[mask]

            weighted_src_embs = torch.mul(masked_overlapping_src_embs, masked_overlapping_emb_weights.unsqueeze(1))
            # It's a convex combination because the weights sum up to 1
            convex_combination = torch.sum(weighted_src_embs, dim=0)

            new_token_target_vocab_idx = new_tokens_lst[new_token_idx].target.id
            target_embeddings[new_token_target_vocab_idx] = convex_combination
        return target_embeddings

    @override
    def initialize_embeddings(self, source_embeddings, **kwargs):
        """
        Method that initializes the embeddings of a LM with a target tokenizer given a source LM.
        Leverages overlap between the source vocabulary and the target vocabulary to directly copy source embeddings
        and uses a helper model to initialize the rest.
        In order to keep most of the original FOCUS code we convert the source_embeddings to work with tensors
        and convert the target_embeddings to a numpy.ndarray when we are finished.

        :param source_embeddings: The source embeddings (either the input or output embeddings).
        :param kwargs: no kwargs

        :return: The initialized embedding matrix
        """
        ####################################################
        # 4. Copy source embeddings for overlapping tokens #
        ####################################################
        source_embeddings = torch.from_numpy(source_embeddings).to(self.device)
        target_embeddings = torch.zeros((len(self.target_tokenizer), source_embeddings.shape[1]), device=self.device)
        for _, overlapping_token in self.sorted_overlapping_tokens:
            # Instead of using overlapping_token.source_embedding we retrieve the embedding from source_embeddings
            # so we can initialize the input and output embeddings separately
            embs_lst = [source_embeddings[s.id] for s in overlapping_token.source]
            source_embedding = embs_lst[0]
            target_embeddings[overlapping_token.target.id] = source_embedding
        logger.info(f"Copied embeddings for {len(self.overlapping_tokens)} overlapping tokens.")

        ###########################################################
        # 5. Initialize "bad" new tokens from normal distribution #
        ###########################################################
        emb_mean = source_embeddings.mean(dim=0)
        emb_std = source_embeddings.std(dim=0)

        for ood_new_token in self.random_init_new_tokens:
            target_embeddings[ood_new_token.target.id] = torch.normal(emb_mean, emb_std, generator=self.gen)
        logger.info(
            f"Initialized {len(self.random_init_new_tokens)} new tokens from N(source_mean, source_std) because they "
            f"do not have auxiliary embeddings (this is okay if it's not too many)."
        )

        #######################################################
        # 6. Finally, initialize additional tokens with FOCUS #
        #######################################################
        overlapping_tokens_for_focus = {k: v for k, v in self.sorted_overlapping_tokens if v.use_for_focus}
        target_embeddings = self.focus_additional_token_initialization(
            overlapping_tokens_for_focus, self.new_tokens, source_embeddings, target_embeddings, device=self.device
        )
        logger.info(f"Initialized {self.cleverly_initialized_tokens}/{len(self.target_tokens)} tokens with the FOCUS method.")
        return target_embeddings.cpu().numpy()
