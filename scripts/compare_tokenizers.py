import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt

from transformers import AutoTokenizer
from lm_transfer.embedding_initialization.overlap_utils import get_overlapping_tokens


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_model", type=str, default=None, help="The model_name_or_path of the source model")
    parser.add_argument("--tgt_model", type=str, default=None,help="The model_name_or_path of the target model")
    parser.add_argument("--language", type=str, default=None, choices=["monolingual", "multilingual"],
                        help="The language of the tokenizer (monolingual or multilingual)")
    parser.add_argument("--output_path", type=str, default=None, help="The path to save the plot")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.src_model and args.tgt_model:
        src_model = args.src_model
        tgt_model = args.tgt_model
    elif args.language == "monolingual":
        src_model = "FacebookAI/roberta-base"
        tgt_model = "phucdev/vi-bpe-culturax-2048"
    elif args.language == "multilingual":
        src_model = "FacebookAI/xlm-roberta-base"
        tgt_model = "phucdev/vi-spm-culturax-2048"
    else:
        raise ValueError("Please provide either src_model and tgt_model or language")
    src_tokenizer = AutoTokenizer.from_pretrained(src_model)
    tgt_tokenizer = AutoTokenizer.from_pretrained(tgt_model)

    # Simple overlap analysis -> CLP method
    target_tokens = set(tgt_tokenizer.get_vocab().keys())
    source_tokens = set(src_tokenizer.get_vocab().keys())

    exact_overlapping_tokens = target_tokens & source_tokens
    exact_missing_tokens = target_tokens - source_tokens

    logger.info(f'Simple match (CLP): {len(exact_overlapping_tokens)=}; {len(exact_missing_tokens)=}')


    # FOCUS method of overlap analysis involves canonicalization
    canon_overlapping_tokens, canon_missing_tokens = get_overlapping_tokens(
        target_tokenizer=tgt_tokenizer,
        source_tokenizer=src_tokenizer,
        exact_match_all=True,
        match_symbols=False,
        fuzzy_match_all=False
    )
    # Sort to ensure same order every time (especially important when executing on multiple ranks)
    # Target token -> source token(s)
    canon_overlapping_tokens = sorted(canon_overlapping_tokens.items(), key=lambda x: x[1].target.id)
    canon_missing_tokens = sorted(canon_missing_tokens.items(), key=lambda x: x[1].target.id)

    logger.info(f'Canonicalization (FOCUS): {len(canon_overlapping_tokens)=}; {len(canon_missing_tokens)=}')

    # FOCUS method of overlap analysis: fuzzy matching ignoring whitespace and case
    fuzzy_overlapping_tokens, fuzzy_missing_tokens = get_overlapping_tokens(
        target_tokenizer=tgt_tokenizer,
        source_tokenizer=src_tokenizer,
        exact_match_all=True,
        match_symbols=False,
        fuzzy_match_all=True
    )
    # Sort to ensure same order every time (especially important when executing on multiple ranks)
    # Target token -> source token(s)
    fuzzy_overlapping_tokens = sorted(fuzzy_overlapping_tokens.items(), key=lambda x: x[1].target.id)
    fuzzy_missing_tokens = sorted(fuzzy_missing_tokens.items(), key=lambda x: x[1].target.id)

    logger.info(f'Fuzzy Matching (FOCUS): {len(fuzzy_overlapping_tokens)=}; {len(fuzzy_missing_tokens)=}')

    # Now create a bar plot to compare the number of overlapping tokens and missing tokens using the three matching methods
    # Create a bar plot
    # 1) Prepare data
    methods = ["Exact", "Canonical", "Fuzzy"]
    overlap_counts = [len(exact_overlapping_tokens), len(canon_overlapping_tokens), len(fuzzy_overlapping_tokens)]
    missing_counts = [len(exact_missing_tokens), len(canon_missing_tokens), len(fuzzy_missing_tokens)]

    x = np.arange(len(methods))
    width = 0.5

    # 2) Create the figure and axes
    fig, ax = plt.subplots(figsize=(7, 5))

    # 3) Plot stacked bars
    bar_overlap = ax.bar(
        x,
        overlap_counts,
        width,
        label="Overlap",
        color="#1f77b4"  # e.g. a nice blue
    )
    bar_missing = ax.bar(
        x,
        missing_counts,
        width,
        bottom=overlap_counts,
        label="Missing",
        color="#ff7f0e"  # e.g. an orange
    )

    # 4) Customize
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Number of Tokens")
    ax.set_title("Vocabulary Overlap vs. Missing Tokens")
    ax.legend()

    # Now label the bars.
    # - For Overlap (bottom bars), a label_type of 'center' or 'edge' can work.
    # - For Missing (top bars), 'center' usually places the label nicely in the middle of that stacked segment.
    ax.bar_label(bar_overlap, label_type='center', color="white")
    ax.bar_label(bar_missing, label_type='center', color="white")

    plt.tight_layout()
    plt.grid(axis="y")
    plt.show()

    if args.output_path:
        plt.savefig(args.output_path)
    plt.close()


if __name__ == "__main__":
    main()
