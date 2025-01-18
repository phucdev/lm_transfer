import os
import argparse
import logging
import json

from time import perf_counter
from contextlib import contextmanager
from pathlib import Path

from lm_transfer.embedding_initialization.tokenizer_transfer import (
    RandomInitializationTokenizerTransfer,
)
from lm_transfer.embedding_initialization.fvt_transfer import FVTTokenizerTransfer
from lm_transfer.embedding_initialization.wechsel_transfer import WechselTokenizerTransfer
from lm_transfer.embedding_initialization.focus_transfer import FocusTokenizerTransfer
from lm_transfer.embedding_initialization.ramen_transfer import RamenTokenizerTransfer


logger = logging.getLogger(__name__)


def parse_args():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)

    parser = argparse.ArgumentParser(description="Transfer models")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for all the models"
    )
    parser.add_argument(
        "--transfer_type",
        type=str,
        required=True,
        choices=["monolingual", "multilingual"],
        help="Type of source model (monolingual, multilingual)"
    )
    parser.add_argument(
        "--source_model_name",
        type=str,
        default=None,
        help="Source model name"
    )
    parser.add_argument(
        "--target_tokenizer_name",
        type=str,
        default=None,
        help="Target tokenizer name"
    )
    parser.add_argument(
        "--source_language_identifier",
        type=str,
        default="en",
        help="Source language identifier"
    )
    parser.add_argument(
        "--target_language_identifier",
        type=str,
        default="vi",
        help="Target language identifier"
    )
    parser.add_argument(
        "--bilingual_dictionary",
        type=str,
        default=None,
        help="Bilingual dictionary file path"
    )
    parser.add_argument(
        "--align_matrix_path",
        type=str,
        default=None,
        help="Alignment matrix path"
    )
    parser.add_argument(
        "--aligned_data_path",
        type=str,
        default=None,
        help="Aligned data path"
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="OpenSubtitles",
        help="Parallel corpus for RAMEN"
    )
    parser.add_argument(
        "--target_training_data_path",
        type=str,
        default=None,
        help="Target training data path"
    )
    parser.add_argument(
        "--statistics_file",
        type=str,
        default=None,
        help="File path to store transfer statistics"
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help="Number of processes for FOCUS initialization"
    )
    parser.add_argument(
        "--fasttext_model_dim",
        type=int,
        default=300,
        help="FastText model dimension for FOCUS initialization"
    )
    parser.add_argument(
        "--freq_dict_path",
        type=str,
        default=None,
        help="Frequency dictionary path for FVT initialization"
    )
    args = parser.parse_args()
    # Set some default values for paths
    if args.bilingual_dictionary is None:
        args.bilingual_dictionary = os.path.join(parent_dir, "bilingual_dictionary/MUSE/en-vi.txt")
    if args.align_matrix_path is None:
        args.align_matrix_path = os.path.join(parent_dir, "alignment/res/cc.en-vi.vec-mat")
    if args.aligned_data_path is None:
        args.aligned_data_path = os.path.join(parent_dir, "data/parallel_data/OpenSubtitles")
    if args.target_training_data_path is None:
        args.target_training_data_path = os.path.join(parent_dir, "data/culturax_vi/train.json")
    if args.statistics_file is None:
        args.statistics_file = os.path.join(args.output_dir, "transfer_statistics.json")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    return args


@contextmanager
def measure_time() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start


def format_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"


def random_embedding_initialization(
        output_dir,
        source_model_name="FacebookAI/roberta-base",
        target_tokenizer_name="phucdev/vi-bpe-culturax-4g-sample"
):
    transfer_pipeline = RandomInitializationTokenizerTransfer(
        source_model_name,
        target_tokenizer_name,
        target_model_path=os.path.join(output_dir, "random_initialization")
    )
    transfer_pipeline.transfer()
    return transfer_pipeline.get_transfer_statistics()


def ramen_embedding_initialization(
        output_dir,
        aligned_data_path,
        source_language_identifier="en",
        target_language_identifier="vi",
        corpus="OpenSubtitles",
        source_model_name="FacebookAI/roberta-base",
        target_tokenizer_name="phucdev/vi-bpe-culturax-4g-sample"
):
    transfer_pipeline = RamenTokenizerTransfer(
        source_model_name,
        target_tokenizer_name,
        aligned_data_path=aligned_data_path,
        source_language_identifier=source_language_identifier,
        target_language_identifier=target_language_identifier,
        corpus=corpus,
        target_model_path=os.path.join(output_dir, "ramen_initialization")
    )
    transfer_pipeline.transfer()
    return transfer_pipeline.get_transfer_statistics()


def wechsel_embedding_initialization(
        output_dir,
        bilingual_dictionary,
        source_model_name="FacebookAI/roberta-base",
        target_tokenizer_name="phucdev/vi-bpe-culturax-4g-sample",
        source_language_identifier="en",
        target_language_identifier="vi"
):
    transfer_pipeline = WechselTokenizerTransfer(
        source_model_name,
        target_tokenizer_name,
        bilingual_dictionary=bilingual_dictionary,
        source_language_identifier=source_language_identifier,
        target_language_identifier=target_language_identifier,
        target_model_path=os.path.join(output_dir, "wechsel_initialization")
    )
    transfer_pipeline.transfer()
    return transfer_pipeline.get_transfer_statistics()


def wechsel_aligned_embedding_initialization(
        output_dir,
        source_model_name="FacebookAI/roberta-base",
        target_tokenizer_name="phucdev/vi-bpe-culturax-4g-sample",
        source_language_identifier="en",
        target_language_identifier="vi",
        leverage_overlap=False,
        overwrite_with_overlap=False
):
    transfer_pipeline = WechselTokenizerTransfer(
        source_model_name,
        target_tokenizer_name,
        align_strategy=None,
        use_subword_info=False,
        bilingual_dictionary=None,
        source_language_identifier=source_language_identifier,
        target_language_identifier=target_language_identifier,
        target_model_path=os.path.join(output_dir, "wechsel_aligned_initialization"),
        leverage_overlap=leverage_overlap,
        overwrite_with_overlap=overwrite_with_overlap
    )
    transfer_pipeline.transfer()
    return transfer_pipeline.get_transfer_statistics()


def wechsel_rcsls_embedding_initialization(
        output_dir,
        align_matrix_path,
        source_model_name="FacebookAI/roberta-base",
        target_tokenizer_name="phucdev/vi-bpe-culturax-4g-sample",
        source_language_identifier="en",
        target_language_identifier="vi"
):
    transfer_pipeline = WechselTokenizerTransfer(
        source_model_name,
        target_tokenizer_name,
        align_strategy="align matrix",
        align_matrix_path=align_matrix_path,
        emb_type="crawl",
        use_subword_info=True,
        bilingual_dictionary=None,
        source_language_identifier=source_language_identifier,
        target_language_identifier=target_language_identifier,
        target_model_path=os.path.join(output_dir, "wechsel_rcsls_initialization"),
    )
    transfer_pipeline.transfer()
    return transfer_pipeline.get_transfer_statistics()


def wechsel_overlap_embedding_initialization(
        output_dir,
        bilingual_dictionary,
        source_model_name="FacebookAI/roberta-base",
        target_tokenizer_name="phucdev/vi-bpe-culturax-4g-sample",
        source_language_identifier="en",
        target_language_identifier="vi"
):
    transfer_pipeline = WechselTokenizerTransfer(
        source_model_name,
        target_tokenizer_name,
        bilingual_dictionary=bilingual_dictionary,
        source_language_identifier=source_language_identifier,
        target_language_identifier=target_language_identifier,
        target_model_path=os.path.join(output_dir, "wechsel_overlap_initialization"),
        leverage_overlap=True,
        overwrite_with_overlap=True
    )
    transfer_pipeline.transfer()
    return transfer_pipeline.get_transfer_statistics()


def focus_monolingual_embedding_initialization(
        output_dir,
        target_training_data_path,
        source_model_name="FacebookAI/roberta-base",
        target_tokenizer_name="phucdev/vi-bpe-culturax-4g-sample",
        language_identifier="vi",
        processes=None,
        fasttext_model_dim=300
):
    transfer_pipeline = FocusTokenizerTransfer(
        source_model_name,
        target_tokenizer_name,
        language_identifier=language_identifier,
        target_training_data_path=target_training_data_path,
        processes=processes,
        fasttext_model_dim=fasttext_model_dim,
        target_model_path=os.path.join(output_dir, "focus_monolingual_initialization")
    )
    transfer_pipeline.transfer()
    return transfer_pipeline.get_transfer_statistics()


def fvt_embedding_initialization(
        output_dir,
        source_model_name="FacebookAI/xlm-roberta-base",
        target_tokenizer_name="phucdev/vi-spm-culturax-4g-sample",
):
    transfer_pipeline = FVTTokenizerTransfer(
        source_model_name,
        target_tokenizer_name,
        target_model_path=os.path.join(output_dir, "fvt_initialization")
    )
    transfer_pipeline.transfer()
    return transfer_pipeline.get_transfer_statistics()


def fvt_subword_length_embedding_initialization(
        output_dir,
        source_model_name="FacebookAI/xlm-roberta-base",
        target_tokenizer_name="phucdev/vi-spm-culturax-4g-sample",
):
    # Intuitively, subwords with longer length should contain more relevant information
    transfer_pipeline = FVTTokenizerTransfer(
        source_model_name,
        target_tokenizer_name,
        target_model_path=os.path.join(output_dir, "fvt_subword_length_initialization"),
        aggregation_method="subword_length_weighted"
    )
    transfer_pipeline.transfer()
    return transfer_pipeline.get_transfer_statistics()


def fvt_minimize_punctuation_embedding_initialization(
        output_dir,
        source_model_name="FacebookAI/xlm-roberta-base",
        target_tokenizer_name="phucdev/vi-spm-culturax-4g-sample",
):
    # With SPM tokenizers a lot of target tokens contain punctuation characters
    # -> we want to minimize the impact of punctuation tokens on the target embedding
    transfer_pipeline = FVTTokenizerTransfer(
        source_model_name,
        target_tokenizer_name,
        target_model_path=os.path.join(output_dir, "fvt_minimize_punctuation_initialization"),
        minimize_punctuation_weight=True
    )
    transfer_pipeline.transfer()
    return transfer_pipeline.get_transfer_statistics()


def fvt_rescale_initialization(
        output_dir,
        source_model_name="FacebookAI/xlm-roberta-base",
        target_tokenizer_name="phucdev/vi-spm-culturax-4g-sample",
):
    transfer_pipeline = FVTTokenizerTransfer(
        source_model_name,
        target_tokenizer_name,
        target_model_path=os.path.join(output_dir, "fvt_rescale_initialization"),
        rescale=True
    )
    transfer_pipeline.transfer()
    return transfer_pipeline.get_transfer_statistics()


def fvt_freq_weighted_minimize_punctuation_embedding_initialization(
        output_dir,
        source_model_name="FacebookAI/xlm-roberta-base",
        target_tokenizer_name="phucdev/vi-spm-culturax-4g-sample",
        target_training_data_path="data/culturax_vi/train.json",
        num_proc=None,
        freq_dict_path=None
):
    # Weight source token embeddings by their frequency in the training data
    # -> frequent tokens should have better quality embeddings
    transfer_pipeline = FVTTokenizerTransfer(
        source_model_name,
        target_tokenizer_name,
        target_model_path=os.path.join(output_dir, "fvt_freq_weighted_minimize_punctuation_initialization"),
        minimize_punctuation_weight=True,
        aggregation_method="freq_weighted",
        target_training_data_path=target_training_data_path,
        num_proc=num_proc,
        freq_dict_path=freq_dict_path
    )
    transfer_pipeline.transfer()
    return transfer_pipeline.get_transfer_statistics()


def focus_multilingual_embedding_initialization(
        output_dir,
        target_training_data_path,
        source_model_name="FacebookAI/xlm-roberta-base",
        target_tokenizer_name="phucdev/vi-spm-culturax-4g-sample",
        language_identifier="vi",
        fasttext_model_dim=300,
        processes=None
):
    transfer_pipeline = FocusTokenizerTransfer(
        source_model_name,
        target_tokenizer_name,
        language_identifier=language_identifier,
        target_training_data_path=target_training_data_path,
        processes=processes,
        fasttext_model_dim=fasttext_model_dim,
        target_model_path=os.path.join(output_dir, "focus_multilingual_initialization")
    )
    transfer_pipeline.transfer()
    return transfer_pipeline.get_transfer_statistics()


def main():
    args = parse_args()
    output_dir = args.output_dir
    transfer_type = args.transfer_type
    source_model_name = args.source_model_name
    target_tokenizer_name = args.target_tokenizer_name
    source_language_identifier = args.source_language_identifier
    target_language_identifier = args.target_language_identifier
    bilingual_dictionary = args.bilingual_dictionary
    align_matrix_path = args.align_matrix_path
    aligned_data_path = args.aligned_data_path
    processes = args.processes
    fasttext_model_dim = args.fasttext_model_dim
    corpus = args.corpus
    target_training_data_path = args.target_training_data_path
    freq_dict_path = args.freq_dict_path
    statistics_file = args.statistics_file

    if os.path.exists(statistics_file):
        with open(statistics_file, "r") as f:
            transfer_statistics = json.load(f)
    else:
        transfer_statistics = {}

    logger.info(f"Args: {transfer_type=}, {source_model_name=}, {target_tokenizer_name=}, {output_dir=}")
    if transfer_type == "monolingual":
        logger.info("(1/7) Random initialization")
        with measure_time() as timer:
            transfer_statistics["random_monolingual"] = random_embedding_initialization(
                output_dir=output_dir, source_model_name=source_model_name, target_tokenizer_name=target_tokenizer_name
            )
        elapsed_time = timer()
        transfer_statistics["random_monolingual"]["elapsed_time"] = elapsed_time
        logger.info(f"Elapsed time: {format_time(elapsed_time)}")

        logger.info("(2/7) RAMEN initialization")
        with measure_time() as timer:
            transfer_statistics["RAMEN"] = ramen_embedding_initialization(
                output_dir=output_dir, source_model_name=source_model_name, target_tokenizer_name=target_tokenizer_name,
                aligned_data_path=aligned_data_path, source_language_identifier=source_language_identifier,
                target_language_identifier=target_language_identifier, corpus=corpus
            )
        elapsed_time = timer()
        transfer_statistics["RAMEN"]["elapsed_time"] = elapsed_time
        logger.info(f"Elapsed time: {format_time(elapsed_time)}")

        logger.info("(3/7) WECHSEL initialization")
        with measure_time() as timer:
            transfer_statistics["WECHSEL"] = wechsel_embedding_initialization(
                output_dir=output_dir, source_model_name=source_model_name, target_tokenizer_name=target_tokenizer_name,
                bilingual_dictionary=bilingual_dictionary
            )
        elapsed_time = timer()
        transfer_statistics["WECHSEL"]["elapsed_time"] = elapsed_time
        logger.info(f"Elapsed time: {format_time(elapsed_time)}")

        logger.info("(4/7) WECHSEL+pre-aligned auxiliary embeddings initialization")
        with measure_time() as timer:
            transfer_statistics["WECHSEL+aligned"] = wechsel_aligned_embedding_initialization(
                output_dir=output_dir, source_model_name=source_model_name, target_tokenizer_name=target_tokenizer_name,
                leverage_overlap=True, overwrite_with_overlap=True
            )
        elapsed_time = timer()
        transfer_statistics["WECHSEL+aligned"]["elapsed_time"] = elapsed_time
        logger.info(f"Elapsed time: {format_time(elapsed_time)}")

        logger.info("(5/7) WECHSEL+RCSLS embeddings initialization")
        with measure_time() as timer:
            transfer_statistics["WECHSEL+rcsls"] = wechsel_rcsls_embedding_initialization(
                output_dir=output_dir, align_matrix_path=align_matrix_path,
                source_model_name=source_model_name, target_tokenizer_name=target_tokenizer_name
            )
        elapsed_time = timer()
        transfer_statistics["WECHSEL+rcsls"]["elapsed_time"] = elapsed_time
        logger.info(f"Elapsed time: {format_time(elapsed_time)}")

        logger.info("(6/7) WECHSEL+overlap initialization")
        with measure_time() as timer:
            transfer_statistics["WECHSEL+overlap"] = wechsel_overlap_embedding_initialization(
                output_dir=output_dir, source_model_name=source_model_name, target_tokenizer_name=target_tokenizer_name,
                bilingual_dictionary=bilingual_dictionary
            )
        elapsed_time = timer()
        transfer_statistics["WECHSEL+overlap"]["elapsed_time"] = elapsed_time
        logger.info(f"Elapsed time: {format_time(elapsed_time)}")

        logger.info("(7/7) FOCUS initialization")
        with measure_time() as timer:
            transfer_statistics["FOCUS_monolingual"] = focus_monolingual_embedding_initialization(
                output_dir=output_dir, source_model_name=source_model_name, target_tokenizer_name=target_tokenizer_name,
                target_training_data_path=target_training_data_path, processes=processes,
                fasttext_model_dim=fasttext_model_dim
            )
        elapsed_time = timer()
        transfer_statistics["FOCUS_monolingual"]["elapsed_time"] = elapsed_time
        logger.info(f"Elapsed time: {format_time(elapsed_time)}")
    elif transfer_type == "multilingual":
        logger.info("(1/8) Random initialization")
        with measure_time() as timer:
            transfer_statistics["random_multilingual"] = random_embedding_initialization(
                output_dir=output_dir, source_model_name=source_model_name, target_tokenizer_name=target_tokenizer_name
            )
        elapsed_time = timer()
        transfer_statistics["random_multilingual"]["elapsed_time"] = elapsed_time
        logger.info(f"Elapsed time: {format_time(elapsed_time)}")

        logger.info("(2/8) FVT initialization")
        with measure_time() as timer:
            transfer_statistics["FVT"] = fvt_embedding_initialization(
                output_dir=output_dir, source_model_name=source_model_name, target_tokenizer_name=target_tokenizer_name
            )
        elapsed_time = timer()
        transfer_statistics["FVT"]["elapsed_time"] = elapsed_time
        logger.info(f"Elapsed time: {format_time(elapsed_time)}")

        logger.info("(3/8) FVT+Subword_Length initialization")
        with measure_time() as timer:
            transfer_statistics["FVT+Subword_Length"] = fvt_subword_length_embedding_initialization(
                output_dir=output_dir, source_model_name=source_model_name, target_tokenizer_name=target_tokenizer_name
            )
        elapsed_time = timer()
        transfer_statistics["FVT+Subword_Length"]["elapsed_time"] = elapsed_time
        logger.info(f"Elapsed time: {format_time(elapsed_time)}")

        logger.info("(4/8) FVT+Minimize_Punctuation initialization")
        with measure_time() as timer:
            transfer_statistics["FVT+Minimize_Punctuation"] = fvt_minimize_punctuation_embedding_initialization(
                output_dir=output_dir, source_model_name=source_model_name, target_tokenizer_name=target_tokenizer_name
            )
        elapsed_time = timer()
        transfer_statistics["FVT+Minimize_Punctuation"]["elapsed_time"] = elapsed_time
        logger.info(f"Elapsed time: {format_time(elapsed_time)}")

        logger.info("(5/8) FVT+Rescale initialization")
        with measure_time() as timer:
            transfer_statistics["FVT+Rescale"] = fvt_rescale_initialization(
                output_dir=output_dir, source_model_name=source_model_name, target_tokenizer_name=target_tokenizer_name
            )
        elapsed_time = timer()
        transfer_statistics["FVT+Rescale"]["elapsed_time"] = elapsed_time
        logger.info(f"Elapsed time: {format_time(elapsed_time)}")

        logger.info("(6/8) FVT+Freq_Weighted+Minimize_Punctuation initialization")
        with measure_time() as timer:
            transfer_statistics["FVT+Freq_Weighted+Minimize_Punctuation"] = fvt_freq_weighted_minimize_punctuation_embedding_initialization(
                output_dir=output_dir, source_model_name=source_model_name, target_tokenizer_name=target_tokenizer_name,
                num_proc=processes, freq_dict_path=freq_dict_path
            )
        elapsed_time = timer()
        transfer_statistics["FVT+Freq_Weighted+Minimize_Punctuation"]["elapsed_time"] = elapsed_time
        logger.info(f"Elapsed time: {format_time(elapsed_time)}")

        logger.info("(7/8) FOCUS initialization")
        with measure_time() as timer:
            transfer_statistics["FOCUS_multilingual"] = focus_multilingual_embedding_initialization(
                output_dir=output_dir, source_model_name=source_model_name, target_tokenizer_name=target_tokenizer_name,
                language_identifier=target_language_identifier, target_training_data_path=target_training_data_path,
                fasttext_model_dim=fasttext_model_dim, processes=processes
            )
        elapsed_time = timer()
        transfer_statistics["FOCUS_multilingual"]["elapsed_time"] = elapsed_time
        logger.info(f"Elapsed time: {format_time(elapsed_time)}")

        logger.info("(8/8) RAMEN initialization")
        with measure_time() as timer:
            transfer_statistics["RAMEN_multilingual"] = ramen_embedding_initialization(
                output_dir=output_dir, source_model_name=source_model_name, target_tokenizer_name=target_tokenizer_name,
                aligned_data_path=aligned_data_path, source_language_identifier=source_language_identifier,
                target_language_identifier=target_language_identifier, corpus=corpus
            )
        elapsed_time = timer()
        transfer_statistics["RAMEN_multilingual"]["elapsed_time"] = elapsed_time
        logger.info(f"Elapsed time: {format_time(elapsed_time)}")
    else:
        logger.error("Invalid transfer type")
        exit(-1)
    logger.info(f"All done! Check the models in {output_dir}. "
                f"You should train these models on target language data before using them.")
    with open(statistics_file, "w") as f:
        f.write(json.dumps(transfer_statistics, indent=2))


if __name__ == "__main__":
    main()
