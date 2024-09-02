import logging
import subprocess
import torch
import os
import numpy as np

from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from .tokenizer_transfer import RandomInitializationTokenizerTransfer
from ..utils.download_utils import download, gunzip

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class RamenTokenizerTransfer(RandomInitializationTokenizerTransfer):
    def __init__(
            self,
            source_model_name_or_path: str,
            target_tokenizer_name_or_path: str,
            target_model_path: str = None,
            aligned_data_path: str = None,
            source_language: str = None,
            target_language: str = None,
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
        Steps:
        1. Obtain parallel data, e.g. from OpenSubtitles
        2. Prepare data by tokenizing the text and its translation with the respective tokenizers
        3. Convert it to a format that fast_align can use: <tokenized source lang text> ||| <tokenized target lang text>
        4. Run fast_align on the data (forward, reverse and create symmetric alignment)
        5. Get the translation probabilities from the alignments
        6. Initialize the target embeddings by averaging the source embeddings of the tokens in the partition based on
           the translation probabilities

        :param source_model_name_or_path:
        :param target_tokenizer_name_or_path:
        :param target_model_path:
        :param aligned_data_path: Path to the aligned data file (parallel data aligned with tools like fast_align)
        """
        super().__init__(source_model_name_or_path, target_tokenizer_name_or_path, target_model_path, **kwargs)
        self.aligned_data_path = aligned_data_path
        self.source_language = source_language
        self.target_language = target_language

    @staticmethod
    def get_parallel_data(source_language, target_language, data_path, corpus="OpenSubtitles"):
        """
        Method to download parallel data from OPUS for a given source and target language.

        :param source_language: Source language identifier, e.g. en
        :param target_language: Target language identifier, e.g. de
        :param data_path: Path to save the downloaded data
        :param corpus: Name of the corpus to download data from, e.g. OpenSubtitles or CCMatrix
        """
        data_path = Path(data_path)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        if corpus == "OpenSubtitles":
            base_link = f"https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/"
        elif corpus == "CCMatrix":
            base_link = f"https://object.pouta.csc.fi/OPUS-CCMatrix/v1/mono/"
        elif corpus == "NLLB":
            base_link = f"https://object.pouta.csc.fi/OPUS-NLLB/v1/mono/"
        elif corpus == "CCAligned":
            base_link = f"https://object.pouta.csc.fi/OPUS-CCAligned/v1/mono/"
        elif corpus == "WikiMatrix":
            base_link = f"https://object.pouta.csc.fi/OPUS-WikiMatrix/v1/mono/"
        else:
            # This might fail and result in a RunTime error
            base_link = f"https://object.pouta.csc.fi/OPUS-{corpus}/v1/mono/"
        download(f"{base_link}{source_language}.txt.gz",
                 data_path.joinpath(f"{source_language}.txt.gz"))
        download(f"https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/{target_language}.txt.gz",
                    data_path.joinpath(f"{target_language}.txt.gz"))
        gunzip(data_path.joinpath(f"{source_language}.txt.gz"))
        gunzip(data_path.joinpath(f"{target_language}.txt.gz"))
        logger.info(f"Downloaded parallel data for {source_language} and {target_language} to {data_path}")

    def process_batch(self, source_batch, target_batch):
        """
        Method to process a batch of text by tokenizing it with the tokenizer.
        :param source_batch:
        :param target_batch:
        :return: String buffer with the tokenized text in the format
            <tokenized source language text> ||| <tokenized target language text>
        """
        buffer = []
        en_input_ids_batch = self.source_tokenizer.batch_encode_plus(source_batch, add_special_tokens=False)["input_ids"]
        vi_input_ids_batch = self.target_tokenizer.batch_encode_plus(target_batch, add_special_tokens=False)["input_ids"]
        for en_input_ids, vi_input_ids in zip(en_input_ids_batch, vi_input_ids_batch):
            en_tokenized = " ".join(self.source_tokenizer.convert_ids_to_tokens(en_input_ids))
            vi_tokenized = " ".join(self.target_tokenizer.convert_ids_to_tokens(vi_input_ids))
            buffer.append(f'{en_tokenized} ||| {vi_tokenized}\n')
        return buffer

    def prepare_data_for_alignment(self, source_file_path, target_file_path, output_path, batch_size=10000):
        """
        Method to prepare data for alignment by tokenizing the text with the tokenizer and writing it to a file
        in the format <tokenized source language text> ||| <tokenized target language text>
        :param source_file_path:
        :param target_file_path:
        :param output_path:
        :param batch_size:
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with (open(source_file_path, 'r') as source_reader, open(target_file_path, 'r') as target_reader,
              open(output_path, 'w') as writer):
            source_batch = []
            target_batch = []
            for idx, (en_line, vi_line) in tqdm(enumerate(zip(source_reader, target_reader)), desc="Tokenizing parallel data"):
                source_batch.append(en_line.strip())
                target_batch.append(vi_line.strip())
                if len(source_batch) < batch_size:
                    continue
                buffer = self.process_batch(source_batch, target_batch)
                writer.writelines(buffer)
                source_batch = []
                target_batch = []
            if source_batch:
                buffer = self.process_batch(source_batch, target_batch)
                writer.writelines(buffer)

    @staticmethod
    def get_alignment(parallel_data_path, output_path):
        """
        Run fast_align on the data (forward, reverse and create symmetric alignment)

        :param parallel_data_path: Path to the parallel data file <tokenized source language text> ||| <tokenized target language text>
        :param output_path: Path to save the alignment files
        """
        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the paths to the fast_align and atools executables
        fast_align_path = os.path.join(current_dir, '..', '..', 'tools', 'fast_align', 'build', 'fast_align')
        atools_path = os.path.join(current_dir, '..', '..', 'tools', 'fast_align', 'build', 'atools')

        logger.info("Running fast_align forward alignment")
        command1 = [
            fast_align_path,
            "-i", parallel_data_path,
            "-d", "-o", "-v",
            "-I", "10"
        ]
        with open(f"{output_path}/forward.source-target", "w") as output_file:
            subprocess.run(command1, stdout=output_file)

        logger.info("Running fast_align reverse alignment")
        command2 = [
            fast_align_path,
            "-i", parallel_data_path,
            "-d", "-o", "-v", "-r",
            "-I", "10"
        ]
        with open(f"{output_path}/reverse.source-target", "w") as output_file:
            subprocess.run(command2, stdout=output_file)

        logger.info("Running atools for symmetric alignment")
        command3 = [
            atools_path,
            "-i", f"{output_path}/forward.source-target",
            "-j", f"{output_path}/reverse.source-target",
            "-c", "grow-diag-final-and"
        ]
        with open(f"{output_path}/align.source-target", "w") as output_file:
            subprocess.run(command3, stdout=output_file)

    @staticmethod
    def get_prob_para(parallel_data_path, alignment_path):
        count = defaultdict(dict)
        with open(parallel_data_path, "r") as parallel_data_file, open(alignment_path, "r") as alignment_file:
            for line, alignment in tqdm(zip(parallel_data_file, alignment_file), desc="Collecting counts"):
                langs = line.strip().split(' ||| ')
                if len(langs) != 2:
                    continue
                # An alignment looks something like this: 0-0 1-1 1-2
                alignment = [tuple(map(int, x.split('-'))) for x in alignment.split()]
                source_tokens = langs[0].split()
                target_tokens = langs[1].split()

                for (sid, tid) in alignment:
                    if sid >= len(source_tokens) or tid >= len(target_tokens):
                        continue
                    source_token, target_token = source_tokens[sid], target_tokens[tid]
                    count[target_token][source_token] += 1

        # re-normalize counts to get translation probability
        logger.info('Re-normalizing counts')
        for target_token, source_token_counts in tqdm(count.items()):
            total_source_token_count = sum(source_token_counts.values())
            for source_token, source_token_count in source_token_counts.items():
                count[target_token][source_token] = source_token_count / total_source_token_count
        return count

    def initialize_embeddings(self, **kwargs):
        """
        Method that initializes the embeddings of a given LM with the source embeddings.

        :param kwargs: no kwargs

        :return: The initialized embedding matrix
        """
        target_embeddings = super().initialize_embeddings(**kwargs)

        self.get_parallel_data(self.source_language, self.target_language, self.aligned_data_path)
        self.prepare_data_for_alignment(
            f"{self.aligned_data_path}/{self.source_language}.txt",
            f"{self.aligned_data_path}/{self.target_language}.txt",
            f"{self.aligned_data_path}/tokenized_parallel_data.txt"
        )
        self.get_alignment(f"{self.aligned_data_path}/tokenized_parallel_data.txt", self.aligned_data_path)
        prob = self.get_prob_para(
            f"{self.aligned_data_path}/tokenized_parallel_data.txt",
            f"{self.aligned_data_path}/align.source-target"
        )
        num_src_per_tgt = np.array([len(x) for x in prob.values()]).mean()
        logger.info(f"# aligned src / tgt: {num_src_per_tgt:.5}")

        for t, ws in prob.items():
            if not self.target_tokenizer.token_to_id(t):
                continue

            px, ix = [], []
            for e, p in ws.items():
                # get index of the source word e
                j = self.source_tokenizer.convert_tokens_to_ids(e)
                ix.append(j)
                px.append(p)
            px = np.asarray(px)
            # get index of target word t
            ti = self.target_tokenizer.token_to_id(t)
            target_embeddings[ti] = px @ self.source_embeddings[ix]

        return target_embeddings
