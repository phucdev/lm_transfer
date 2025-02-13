import argparse
import logging
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm

from transformers import AutoTokenizer
from lm_transfer.embedding_initialization.overlap_utils import get_overlapping_tokens


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default=None, help="The path to save the plot")
    parser.add_argument("--title", type=str, default=None, help="The title of the plot")
    parser.add_argument("--include_canonicalized", action="store_true", default=False, help="Include overlap of canonicalized tokens in the plot")
    parser.add_argument("--include_fuzzy", action="store_true", default=False, help="Include overlap of fuzzy matched tokens in the plot")
    parser.add_argument("--fig_size", type=int, nargs=2, default=[7, 5], help="The size of the plot")
    parser.add_argument("--sample_size", type=int, default=100, help="The number of overlapping tokens to sample")
    parser.add_argument("--output_dir", type=str, default=None, help="The path to save samples of overlapping tokens")
    parser.add_argument("--seed", type=int, default=42, help="The random seed")
    parser.add_argument("--use_serif_font", action="store_true", default=False, help="Use the Source Serif font")
    return parser.parse_args()


def get_exact_overlap(target_tokenizer, source_tokenizer):
    target_tokens = set(target_tokenizer.get_vocab().keys())
    source_tokens = set(source_tokenizer.get_vocab().keys())
    exact_overlapping_tokens = target_tokens & source_tokens
    exact_missing_tokens = target_tokens - source_tokens
    return exact_overlapping_tokens, exact_missing_tokens


def get_canonicalized_overlap(target_tokenizer, source_tokenizer):
    canon_overlapping_tokens, canon_missing_tokens = get_overlapping_tokens(
        target_tokenizer=target_tokenizer,
        source_tokenizer=source_tokenizer,
        exact_match_all=True,
        match_symbols=False,
        fuzzy_match_all=False
    )
    return canon_overlapping_tokens, canon_missing_tokens


def get_fuzzy_overlap(target_tokenizer, source_tokenizer):
    fuzzy_overlapping_tokens, fuzzy_missing_tokens = get_overlapping_tokens(
        target_tokenizer=target_tokenizer,
        source_tokenizer=source_tokenizer,
        exact_match_all=True,
        match_symbols=False,
        fuzzy_match_all=True
    )
    return fuzzy_overlapping_tokens, fuzzy_missing_tokens


def main():
    args = parse_args()

    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['legend.fontsize'] = 8
    try:
        # Path to the custom font
        if args.use_serif_font:
            font_path = "/usr/share/fonts/truetype/Source_Serif_4/static/SourceSerif4-Regular.ttf"
        else:
            font_path = "/usr/share/fonts/truetype/Source_Sans_3/static/SourceSans3-Regular.ttf"
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        logger.info(f"Setting {prop.get_name()} as font family.")
        # Set the font globally in rcParams
        mpl.rcParams['font.family'] = prop.get_name()
    except Exception as e:
        logger.warning(f"Failed to set font family: {e}. Defaulting to system font.")

    random.seed(args.seed)
    np.random.seed(args.seed)

    mono_src_model = "FacebookAI/roberta-base"
    mono_tgt_model = "phucdev/vi-bpe-culturax-2048"
    multi_src_model = "FacebookAI/xlm-roberta-base"
    multi_tgt_model = "phucdev/vi-spm-culturax-2048"

    mono_src_tokenizer = AutoTokenizer.from_pretrained(mono_src_model)
    mono_tgt_tokenizer = AutoTokenizer.from_pretrained(mono_tgt_model)
    multi_src_tokenizer = AutoTokenizer.from_pretrained(multi_src_model)
    multi_tgt_tokenizer = AutoTokenizer.from_pretrained(multi_tgt_model)

    # Calculate the overlap for each model pair and then plot the results
    models = ["RoBERTa", "XLM-RoBERTa"]

    # Get the exact overlap
    mono_exact_overlap, mono_exact_missing = get_exact_overlap(mono_tgt_tokenizer, mono_src_tokenizer)
    multi_exact_overlap, multi_exact_missing = get_exact_overlap(multi_tgt_tokenizer, multi_src_tokenizer)
    overlap_counts = [len(mono_exact_overlap), len(multi_exact_overlap)]
    missing_counts = [len(mono_exact_missing), len(multi_exact_missing)]

    if args.output_dir:
        # Sample random tokens from the overlapping tokens, decode them, and write them to Excel file for manual inspection
        for model, overlap_tokens, tokenizer in zip(models, [mono_exact_overlap, multi_exact_overlap], [mono_src_tokenizer, multi_src_tokenizer]):
            sample_tokens = np.random.choice(list(overlap_tokens), size=args.sample_size, replace=False)
            sample_decoded = [tokenizer.decode(tokenizer.convert_tokens_to_ids(token)) for token in sample_tokens]
            df = pd.DataFrame({"Token": sample_tokens, "Decoded": sample_decoded, "Category": ["" for _ in range(args.sample_size)]})
            df.to_excel(f"{args.output_dir}/{model}_exact_overlap_sample.xlsx", index=False, engine="xlsxwriter")
            logger.info(f"Saved a sample of {args.sample_size} overlapping tokens from {model} to {args.output_dir}/{model}_exact_overlap_sample.xlsx")

    # Now create a bar plot to compare the number of overlapping tokens and missing tokens
    # Create a bar plot
    x = np.arange(len(models))
    width = 0.5

    # 2) Create the figure and axes
    fig, ax = plt.subplots(figsize=args.fig_size)

    # 3) Plot stacked bars
    bar_overlap = ax.bar(
        x,
        overlap_counts,
        width,
        label="Overlapping Tokens",
        color="#1f77b4"  # e.g. a nice blue
    )
    bar_missing = ax.bar(
        x,
        missing_counts,
        width,
        bottom=overlap_counts,
        label="Non-Overlapping Tokens",
        color="#ff7f0e"  # e.g. an orange
    )

    # 4) Customize
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Number of Tokens")
    if args.title:
        ax.set_title(args.title)
    ax.legend(bbox_to_anchor=(0.5, 1.0), loc='lower center', ncol=2)

    # Now label the bars.
    # - For Overlap (bottom bars), a label_type of 'center' or 'edge' can work.
    # - For Missing (top bars), 'center' usually places the label nicely in the middle of that stacked segment.
    ax.bar_label(bar_overlap, label_type='center', color="white")
    ax.bar_label(bar_missing, label_type='center', color="white")

    plt.tight_layout()
    # plt.grid(axis="y")
    if args.output_path:
        plt.savefig(args.output_path, format="pdf", bbox_inches="tight")
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
