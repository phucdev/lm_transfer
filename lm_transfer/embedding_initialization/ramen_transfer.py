import json
import logging
import subprocess
import os
import numpy as np
import torch
from torch import nn

from tqdm import tqdm
from pathlib import Path
from collections import defaultdict, Counter
from overrides import override

from lm_transfer.embedding_initialization.tokenizer_transfer import TokenizerTransfer
from lm_transfer.utils.download_utils import download, decompress_archive

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class RamenTokenizerTransfer(TokenizerTransfer):
    def __init__(
            self,
            source_model_name_or_path: str,
            target_tokenizer_name_or_path: str,
            aligned_data_path: str,
            source_language_identifier: str,
            target_language_identifier: str,
            target_model_path: str = None,
            corpus: str = "OpenSubtitles",
            seed: int = 42,
            num_samples: int = None,
            **kwargs
    ):
        """
        Class for transferring embeddings from one tokenizer to another using RAMEN method by Tran (2020).
        Code adapted from https://github.com/alexa/ramen
        From the paper:
        @article{trnke2020_ramen,
               author = {{Tran}, Ke},
                title = "{From English To Foreign Languages: Transferring Pre-trained Language Models}",
              journal = {arXiv e-prints},
                 year = 2020,
                month = feb,
                  eid = {arXiv:2002.07306},
                pages = {arXiv:2002.07306},
        archivePrefix = {arXiv},
               eprint = {2002.07306},
         primaryClass = {cs.CL},
        }
        Steps in RAMEN according to: https://github.com/alexa/ramen/blob/master/code/demo.sh
        1. Obtain parallel data, e.g. from OpenSubtitles
        2. Install fast_align
        3. Prepare vocab/ train a BertWordPiece tokenizer
        4. Prepare parallel data for fast_align: tokenize the text and its translation with the respective tokenizers
            and convert it to a format that fast_align can use: <tokenized source lang text> ||| <tokenized target lang text>
        5. Run fast_align on the data (forward, reverse and create symmetric alignment)
        5. Get the translation probabilities from the alignments
        6. Initialize the target embeddings by averaging the source embeddings of the tokens in the partition based on
           the translation probabilities

        :param source_model_name_or_path:
        :param target_tokenizer_name_or_path:
        :param aligned_data_path: Path to the aligned data file (parallel data aligned with tools like fast_align)
        :param source_language_identifier: Source language identifier, e.g. en
        :param target_language_identifier: Target language identifier, e.g. vi
        :param target_model_path:
        :param corpus: Name of the corpus to download parallel data from, e.g. OpenSubtitles or CCMatrix
        :param seed: Random seed for reproducibility
        :param num_samples: Number of samples to use for calculating the alignment. If None, all samples are used.
        """
        super().__init__(source_model_name_or_path, target_tokenizer_name_or_path, target_model_path, **kwargs)
        self.aligned_data_path = aligned_data_path
        self.source_language_identifier = source_language_identifier
        self.target_language_identifier = target_language_identifier
        self.corpus = corpus
        self.num_samples = num_samples
        self.translation_probabilities = None

        self.seed = seed

    @override
    def save_parameters_to_dict(self):
        """
        Method to save the parameters of the class to a dictionary.

        :return: A dictionary containing the parameters of the class
        """
        params = super().save_parameters_to_dict()
        params.update({
            "aligned_data_path": self.aligned_data_path,
            "source_language_identifier": self.source_language_identifier,
            "target_language_identifier": self.target_language_identifier,
            "corpus": self.corpus,
            "seed": self.seed,
        })
        return params

    @staticmethod
    def get_parallel_data(source_language, target_language, data_path, corpus="OpenSubtitles"):
        """
        Method to download parallel data from OPUS for a given source and target language.
        This is my implementation.

        :param source_language: Source language identifier, e.g. en
        :param target_language: Target language identifier, e.g. vi
        :param data_path: Path to save the downloaded data
        :param corpus: Name of the corpus to download data from, e.g. OpenSubtitles or CCMatrix
        """
        data_path = Path(data_path)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        language_pair = f"{source_language}-{target_language}"
        output_path = Path(data_path).joinpath(f"{language_pair}")
        if output_path.exists():
            logger.info(f"Parallel data for {source_language} and {target_language} already exists in {data_path}. "
                        f"Skipping download")
        else:
            if corpus == "OpenSubtitles":
                base_link = f"https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/"
            elif corpus == "CCMatrix":
                base_link = f"https://object.pouta.csc.fi/OPUS-CCMatrix/v1/moses/"
            elif corpus == "NLLB":
                base_link = f"https://object.pouta.csc.fi/OPUS-NLLB/v1/moses/"
            elif corpus == "CCAligned":
                base_link = f"https://object.pouta.csc.fi/OPUS-CCAligned/v1/moses/"
            elif corpus == "WikiMatrix":
                base_link = f"https://object.pouta.csc.fi/OPUS-WikiMatrix/v1/moses/"
            else:
                # This might fail and result in a RunTime error if the base_link is different for the corpus
                base_link = f"https://object.pouta.csc.fi/OPUS-{corpus}/v1/moses/"
            download(f"{base_link}{language_pair}.txt.zip",
                     data_path.joinpath(f"{language_pair}.txt.zip"))
            decompress_archive(data_path.joinpath(f"{language_pair}.txt.zip"),
                               output_path=output_path)
            logger.info(f"Downloaded parallel data for {source_language} and {target_language} to {data_path}")

    def process_batch(self, source_batch, target_batch):
        """
        Method to process a batch of text by tokenizing it with the tokenizer.
        This is somewhat similar to
        :param source_batch:
        :param target_batch:
        :return: String buffer with the tokenized text in the format
            <tokenized source language text> ||| <tokenized target language text>
        """
        buffer = []
        src_input_ids_batch = self.source_tokenizer.batch_encode_plus(source_batch, add_special_tokens=False)["input_ids"]
        tgt_input_ids_batch = self.target_tokenizer.batch_encode_plus(target_batch, add_special_tokens=False)["input_ids"]
        for src_input_ids, tgt_input_ids in zip(src_input_ids_batch, tgt_input_ids_batch):
            src_tokenized = " ".join(self.source_tokenizer.convert_ids_to_tokens(src_input_ids))
            tgt_tokenized = " ".join(self.target_tokenizer.convert_ids_to_tokens(tgt_input_ids))
            buffer.append(f'{src_tokenized} ||| {tgt_tokenized}\n')
        return buffer

    def prepare_data_for_alignment(
            self,
            source_file_path,
            target_file_path,
            output_path,
            batch_size=10000
    ):
        """
        Method to prepare data for alignment by tokenizing the text and its translation with the respective tokenizer
        and writing it to a file in the format <tokenized source language text> ||| <tokenized target language text>
        This is somewhat similar to https://github.com/alexa/ramen/blob/master/code/utils/tokenizer.py
        but with updated code for HuggingFace tokenizers and simultaneous processing of the source and the
        target language.
        :param source_file_path:
        :param target_file_path:
        :param output_path:
        :param batch_size:
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        if Path(output_path).exists():
            logger.info(f"Tokenized parallel data already exists in {output_path}. Skipping tokenization")
        else:
            with (open(source_file_path, "r") as source_reader, open(target_file_path, "r") as target_reader,
                  open(output_path, "w") as writer):
                source_batch = []
                target_batch = []
                for idx, (src_line, tgt_line) in tqdm(enumerate(zip(source_reader, target_reader)), desc="Tokenizing parallel data"):
                    if self.num_samples and idx >= self.num_samples:
                        break
                    if source_file_path.endswith(".json") and target_file_path.endswith(".json"):
                        src_line = json.loads(src_line)["text"]
                        tgt_line = json.loads(tgt_line)["text"]
                    source_batch.append(src_line.strip())
                    target_batch.append(tgt_line.strip())
                    if len(source_batch) < batch_size:
                        continue
                    buffer = self.process_batch(source_batch, target_batch)
                    writer.writelines(buffer)
                    source_batch = []
                    target_batch = []
                if source_batch:
                    buffer = self.process_batch(source_batch, target_batch)
                    writer.writelines(buffer)

    def get_alignment(self, parallel_data_path, output_path):
        """
        Run fast_align on the data (forward, reverse and create symmetric alignment).
        This essentially runs the three commands from the demo.sh file:
        https://github.com/alexa/ramen/blob/381182c70cec592f65439f733c1994e8ed727f25/code/demo.sh#L52-L54
        It skips the alignment if the corresponding files already exist.

        :param parallel_data_path: Path to the parallel data file <tokenized source language text> ||| <tokenized target language text>
        :param output_path: Path to save the alignment files
        """
        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))

        language_pair = f"{self.source_language_identifier}-{self.target_language_identifier}"

        # Construct the paths to the fast_align and atools executables
        fast_align_path = os.path.join(current_dir, "..", "..", "tools", "fast_align", "build", "fast_align")
        atools_path = os.path.join(current_dir, "..", "..", "tools", "fast_align", "build", "atools")

        forward_path = Path(f"{output_path}/forward.{language_pair}")
        if forward_path.exists():
            logger.info(f"Forward alignment already exists in {output_path}.")
        else:
            logger.info("Running fast_align forward alignment")
            command1 = [
                fast_align_path,
                "-i", parallel_data_path,
                "-d", "-o", "-v",
                "-I", "10"
            ]
            with open(forward_path, "w") as output_file:
                subprocess.run(command1, stdout=output_file)

        reverse_path = Path(f"{output_path}/reverse.{language_pair}")
        if reverse_path.exists():
            logger.info(f"Reverse alignment already exists in {output_path}.")
        else:
            logger.info("Running fast_align reverse alignment")
            command2 = [
                fast_align_path,
                "-i", parallel_data_path,
                "-d", "-o", "-v", "-r",
                "-I", "10"
            ]
            with open(reverse_path, "w") as output_file:
                subprocess.run(command2, stdout=output_file)

        symmetric_path = Path(f"{output_path}/align.{language_pair}")
        if symmetric_path.exists():
            logger.info(f"Symmetric alignment already exists in {output_path}.")
        else:
            logger.info("Running atools for symmetric alignment")
            command3 = [
                atools_path,
                "-i", str(forward_path),
                "-j", str(reverse_path),
                "-c", "grow-diag-final-and"
            ]
            with open(symmetric_path, "w") as output_file:
                subprocess.run(command3, stdout=output_file)

    def get_prob_para(self, parallel_data_path, alignment_path, output_path=None):
        """
        Collect counts and compute translation probabilities from the alignment.
        This is the adapted code from:
        https://github.com/alexa/ramen/blob/381182c70cec592f65439f733c1994e8ed727f25/code/alignment/get_prob_para.py
        Some of the variable names were changed for clarity.
        :param parallel_data_path:
        :param alignment_path:
        :param output_path:
        :return:
        """
        language_pair = f"{self.source_language_identifier}-{self.target_language_identifier}"
        if output_path is None:
            output_path = Path(alignment_path).parent.joinpath(f"translation_probabilities_{language_pair}.pt")
        if Path(output_path).exists():
            logger.info(f"Translation probabilities already exist in {output_path}. Loading them.")
            return torch.load(output_path)
        else:
            count = defaultdict(Counter)
            with open(parallel_data_path, "r") as parallel_data_file, open(alignment_path, "r") as alignment_file:
                for line, alignment in tqdm(zip(parallel_data_file, alignment_file), desc="Collecting counts"):
                    langs = line.strip().split(" ||| ")
                    if len(langs) != 2:
                        continue
                    # An alignment looks something like this: 0-0 1-1 1-2
                    alignment = [tuple(map(int, x.split("-"))) for x in alignment.split()]
                    source_tokens = langs[0].split()
                    target_tokens = langs[1].split()

                    for (sid, tid) in alignment:
                        if sid >= len(source_tokens) or tid >= len(target_tokens):
                            continue
                        source_token, target_token = source_tokens[sid], target_tokens[tid]
                        count[target_token][source_token] += 1

            # re-normalize counts to get translation probability
            logger.info("Re-normalizing counts")
            normalized_count = defaultdict(defaultdict)
            for target_token, source_token_counts in tqdm(count.items()):
                total_source_token_count = sum(source_token_counts.values())
                for source_token, source_token_count in source_token_counts.items():
                    normalized_count[target_token][source_token] = source_token_count / total_source_token_count
            torch.save(normalized_count, output_path)
            return normalized_count

    @override
    def initialize_embeddings(self, source_embeddings, **kwargs):
        """
        Method that initializes the embeddings of a given LM with the source embeddings.

        :param source_embeddings: The source embeddings to transfer
        :param kwargs: no kwargs

        :return: The initialized embedding matrix and optionally the bias vector
        """
        if self.translation_probabilities is not None:
            logger.info("Translation probabilities already computed. Skipping alignment and translation probability computation.")
        else:
            Path(self.aligned_data_path).mkdir(parents=True, exist_ok=True)
            if self.source_language_identifier == self.target_language_identifier and Path(self.corpus).exists():
                # Instead of using parallel data we tokenize the same data with both tokenizers
                logger.info("(1/6) Preparing data...")
                language_pair = f"{self.source_language_identifier}-{self.target_language_identifier}"
                logger.info("(2/6) Tokenizing parallel data for alignment...")
                tokenized_parallel_data_path = f"{self.aligned_data_path}/tokenized_parallel_data.{language_pair}"
                self.prepare_data_for_alignment(
                    self.corpus,
                    self.corpus,
                    tokenized_parallel_data_path
                )
            else:
                logger.info("(1/6) Downloading parallel data...")
                self.get_parallel_data(
                    self.source_language_identifier,
                    self.target_language_identifier,
                    self.aligned_data_path,
                    self.corpus
                )
                language_pair = f"{self.source_language_identifier}-{self.target_language_identifier}"
                base_path = f"{self.aligned_data_path}/{language_pair}/{self.corpus}.{language_pair}"
                logger.info("(2/6) Tokenizing parallel data for alignment...")
                tokenized_parallel_data_path = f"{self.aligned_data_path}/tokenized_parallel_data.{language_pair}"
                self.prepare_data_for_alignment(
                    f"{base_path}.{self.source_language_identifier}",
                    f"{base_path}.{self.target_language_identifier}",
                    tokenized_parallel_data_path
                )
            logger.info("(3/6) Computing alignment...")
            symmetric_alignment_path = f"{self.aligned_data_path}/align.{language_pair}"
            self.get_alignment(tokenized_parallel_data_path, self.aligned_data_path)

            logger.info("(4/6) Collecting counts and computing translation probabilities...")
            self.translation_probabilities = self.get_prob_para(
                tokenized_parallel_data_path,
                symmetric_alignment_path
            )
            num_src_per_tgt = np.array([len(x) for x in self.translation_probabilities.values()]).mean()
            logger.info(f"# aligned src / tgt: {num_src_per_tgt:.5}")

        logger.info("(5/6) Create random fallback matrix and copy embeddings for overlapping special tokens...")
        # For compatibility with the original RAMEN code
        src_embs = torch.from_numpy(source_embeddings)
        src_bias = torch.from_numpy(self.source_output_bias) if self.source_output_bias is not None else None
        src_tokenizer = self.source_tokenizer
        tgt_tokenizer = self.target_tokenizer
        prob = self.translation_probabilities

        emb_dim = src_embs.size(1)
        # Original not compatible with FastTokenizers: num_tgt = tgt_tokenizer.get_vocab_size()
        num_tgt = len(tgt_tokenizer.get_vocab())

        # init with zero
        tgt_embs = src_embs.new_empty(num_tgt, emb_dim)
        if src_bias is not None:
            tgt_bias = src_bias.new_zeros(num_tgt)
        else:
            tgt_bias = None
        nn.init.normal_(tgt_embs, mean=0, std=emb_dim ** -0.5)

        # copy over embeddings of special words
        self.overlap_based_initialized_tokens = 0
        # Original does not handle different order of special tokens
        # for i in src_tokenizer.all_special_ids:
        #     tgt_embs[i] = src_embs[i]
        #     if tgt_bias is not None:
        #         tgt_bias[i] = src_bias[i]
        for i, t in enumerate(tgt_tokenizer.all_special_tokens):
            if t in src_tokenizer.all_special_tokens:
                j = src_tokenizer.convert_tokens_to_ids(t)
                tgt_embs[i] = src_embs[j]
                if tgt_bias is not None:
                    tgt_bias[i] = src_bias[j]
            self.overlap_based_initialized_tokens += 1

        self.cleverly_initialized_tokens = self.overlap_based_initialized_tokens

        logger.info("(6/6) Initializing target embeddings using translation probabilities...")
        for t, ws in prob.items():
            # Original is not compatible with FastTokenizers: if not tgt_tokenizer.token_to_id(t): continue
            if not self.target_token_to_idx[t]:
                continue

            px, ix = [], []
            for e, p in ws.items():
                # get index of the source word e
                j = src_tokenizer.convert_tokens_to_ids(e)
                ix.append(j)
                px.append(p)
            px = torch.tensor(px).to(src_embs.device)
            # get index of target word t
            # Original is not compatible with FastTokenizers: ti = tgt_tokenizer.token_to_id(t)
            ti = self.target_token_to_idx[t]
            tgt_embs[ti] = px @ src_embs[ix]
            self.cleverly_initialized_tokens += 1
            if tgt_bias is not None:
                tgt_bias[ti] = px.dot(src_bias[ix])
        logger.info(f"Initialized embeddings for {self.cleverly_initialized_tokens}/{len(self.target_tokens)} tokens "
                    f"using RAMEN method.")
        return tgt_embs, tgt_bias if tgt_bias is not None else None

    @override
    def transfer(self, **kwargs):
        """
        Method that creates a new LM model with transferred embeddings.
        :param kwargs: no kwargs

        :return: A new in_domain model
        """
        torch.manual_seed(self.seed)
        target_embeddings, target_output_bias = self.initialize_embeddings(
            source_embeddings=self.source_embeddings,
            **kwargs
        )

        target_model = self.source_model
        target_model.resize_token_embeddings(len(self.target_tokenizer))
        target_model.get_input_embeddings().weight.data = target_embeddings

        # For models with separate embedding and unembedding matrix we need to repeat the process
        # for the unembedding matrix
        if not target_model.config.tie_word_embeddings:
            target_output_embeddings, target_output_bias = self.initialize_embeddings(
                source_embeddings=self.source_output_embeddings,
                **kwargs
            )
            target_model.get_output_embeddings().weight.data = target_output_embeddings

        if target_output_bias is not None:
            if "roberta" in self.source_model_name_or_path:
                target_model.lm_head.bias = nn.Parameter(target_output_bias)
            elif "bert" in self.source_model_name_or_path:
                target_model.cls.predictions.decoder.bias = nn.Parameter(target_output_bias)
            else:
                logger.warning("Could not set output bias. Model type not recognized.")

        if self.target_model_path:
            self.target_tokenizer.save_pretrained(self.target_model_path)
            target_model.save_pretrained(self.target_model_path)
            transfer_info_path = Path(self.target_model_path) / "transfer_information.json"
            with open(transfer_info_path, "w") as f:
                information_dict = self.save_parameters_to_dict()
                f.write(json.dumps(information_dict, indent=2))
        return target_model
