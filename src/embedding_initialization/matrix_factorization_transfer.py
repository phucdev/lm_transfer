import logging
import numpy as np

from overrides import override
from .tokenizer_transfer import OverlapTokenizerTransfer
from ..utils.utils import perform_factorize
from ..utils.snmf import SNMF


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class MatrixFactorizationTransfer(OverlapTokenizerTransfer):
    def __init__(
            self,
            source_model_name_or_path: str,
            target_tokenizer_name_or_path: str,
            target_model_path: str = None,
            keep_dim: int = 100,
            mf_method: str = None,
            **kwargs):
        """
        Transfer method leveraging overlap and matrix factorization.
        This is somewhat similar to OFA:
        @article{liu2023ofa,
         title={OFA: A Framework of Initializing Unseen Subword Embeddings for Efficient Large-scale Multilingual Continued Pretraining}
         author={Liu, Yihong and Lin, Peiqin and Wang, Mingyang and Sch{\"u}tze, Hinrich},
         journal={arXiv preprint arXiv:2311.08849},
         year={2023}
        }
        And:
        @inproceedings{pfeiffer-etal-2021-unks,
            title = "{UNK}s Everywhere: {A}dapting Multilingual Language Models to New Scripts",
            author = "Pfeiffer, Jonas  and
              Vuli{\'c}, Ivan  and
              Gurevych, Iryna  and
              Ruder, Sebastian",
            editor = "Moens, Marie-Francine  and
              Huang, Xuanjing  and
              Specia, Lucia  and
              Yih, Scott Wen-tau",
            booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
            month = nov,
            year = "2021",
            address = "Online and Punta Cana, Dominican Republic",
            publisher = "Association for Computational Linguistics",
            url = "https://aclanthology.org/2021.emnlp-main.800",
            doi = "10.18653/v1/2021.emnlp-main.800",
            pages = "10186--10203",
        }

        :param source_model_name_or_path:
        :param target_tokenizer_name_or_path:
        :param target_model_path:
        :param keep_dim: Number of dimensions to keep after factorization.
        :param mf_method: Method for matrix factorization ('svd' or 'snmf').
        :param kwargs:
        """
        super().__init__(source_model_name_or_path, target_tokenizer_name_or_path, target_model_path, **kwargs)
        self.keep_dim = keep_dim
        self.mf_method = mf_method

        self.transfer_method = "matrix_factorization"

    @override
    def initialize_embeddings(self, source_embeddings, **kwargs):
        """
        Initialize embeddings using matrix factorization transfer method.
        :param source_embeddings: Source embeddings (either the input embeddings or the output embeddings).
        :param kwargs:
        :return: Target embeddings
        """
        if self.mf_method == "svd":
            # SVD method as in OFA by Liu et al. (2024)
            # f is described as lower embeddings or coordinates
            # g is described as primitive embeddings
            logger.info(f"Performing SVD factorization as in OFA by Liu et al. (2024)")
            g, f = perform_factorize(source_embeddings, keep_dim=self.keep_dim)
            logger.info(f'{f.shape=}; {g.shape=}')
        elif self.mf_method == "snmf":
            # Semi Non-Negative Matrix Factorization similar to Pfeiffer et al. (2021)
            logger.info(f"Performing SNMF factorization similar to Pfeiffer et al. (2021)")
            snmf_mdl = SNMF(source_embeddings, niter=3000, num_bases=self.keep_dim)
            snmf_mdl.factorize()
            f = snmf_mdl.W
            g = snmf_mdl.H
            logger.info(f'{f.shape=}; {g.shape=}')
        else:
            # No factorization
            logger.info(f"No matrix factorization")
            f = source_embeddings
            g = np.eye(source_embeddings.shape[1])

        target_embeddings = self.initialize_random_embeddings(source_embeddings)

        # TODO Use any of the other transfer methods to initialize the target embeddings, e.g. CLP
        pass

        # Multiply target embeddings with g to get full-sized embedding matrix
        target_embeddings = np.dot(target_embeddings, g)

        target_embeddings, overlapping_token_indices = self.copy_overlapping_tokens(
            source_embeddings=source_embeddings,
            target_embeddings=target_embeddings,
            return_overlapping_token_indices=True
        )
        target_embeddings, overlapping_special_token_indices = self.copy_special_tokens(
            source_embeddings=source_embeddings,
            target_embeddings=target_embeddings,
            return_overlapping_token_indices=True
        )
        for idx in overlapping_special_token_indices:
            if idx not in overlapping_token_indices:
                overlapping_token_indices.append(idx)

        self.overlap_based_initialized_tokens = len(overlapping_token_indices)
        self.cleverly_initialized_tokens = len(overlapping_token_indices)

        return target_embeddings
