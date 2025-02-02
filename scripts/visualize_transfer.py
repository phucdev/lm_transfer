import argparse
import json
import logging
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import numpy as np

from tqdm import tqdm
from transformers import AutoTokenizer

from lm_transfer.utils.utils import NpEncoder


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


METHOD_MAP = {
  "random_initialization": "RAND",
  "ramen_initialization": "RAMEN",
  "ramen_overlap_initialization": "RAMEN+Overlap",
  "wechsel_initialization": "WECHSEL",
  "wechsel_overlap_initialization": "WECHSEL+Overlap",
  "wechsel_aligned_initialization": "WECHSEL+PreAligned+Overlap",
  "wechsel_rcsls_initialization": "WECHSEL+RCSLS",
  "focus_monolingual_initialization": "FOCUS",
  "fvt_initialization": "FVT",
  "fvt_minimize_punctuation_initialization": "FVT+MinPunct",
  "fvt_subword_length_initialization": "FVT+SubwordLength",
  "fvt_rescale_initialization": "FVT+Rescale",
  "fvt_freq_weighted_minimize_punctuation_initialization": "FVT+FreqWeighted+MinPunct",
  "focus_multilingual_initialization": "FOCUS",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize the transfer of a model.")
    parser.add_argument("--input_dir", type=str, default=None, help="The directory containing the models.")
    parser.add_argument("--source_model_name_or_path", type=str, default=None, help="The source model name or path.")
    parser.add_argument("--normalize", action="store_true", default=False, help="Normalize the sum of weights.")
    parser.add_argument("--use_serif_font", action="store_true", default=False, help="Use the Source Serif font.")
    parser.add_argument("--show_plot", action="store_true", default=False, help="Show the plot.")
    parser.add_argument("--top_k", type=int, default=10, help="Extract top k contributing source tokens.")

    args = parser.parse_args()
    return args


def visualize_transfer(
        input_dir,
        source_model_name_or_path,
        normalize=False,
        top_k=10,
        fig_size=(10, 6),
        show_plot=False
):
    src_emb_usage = {}
    for model_dir in tqdm(os.listdir(input_dir)):
        model_path = os.path.join(input_dir, model_dir)
        if not os.path.isdir(model_path):
            continue
        if not os.path.exists(os.path.join(model_path, "sources.json")):
            continue
        with open(os.path.join(model_path, "sources.json"), "r") as f:
            sources = json.load(f)
        src_tokenizer = AutoTokenizer.from_pretrained(source_model_name_or_path)
        tgt_tokenizer = AutoTokenizer.from_pretrained(model_path)
        src_usage_count = np.zeros(src_tokenizer.vocab_size, dtype=np.int32)
        src_weight_sums = np.zeros(src_tokenizer.vocab_size, dtype=np.float32)
        # length of src_token_indices for each target, account for tokens that were randomly initialized
        target_sources_count = [0] * (tgt_tokenizer.vocab_size - len(sources))
        is_used_direct_copy = [False] * src_tokenizer.vocab_size
        is_used_weighted = [False] * src_tokenizer.vocab_size

        num_rand_init = tgt_tokenizer.vocab_size - len(sources)
        num_direct_copy = 0
        num_weighted_mean = 0

        for target_token, source_info in sources.items():
            source_token_indices = source_info[1]
            source_weights = source_info[2]
            target_sources_count.append(len(source_token_indices))
            # Accumulate usage counts
            for idx in source_token_indices:
                src_usage_count[idx] += 1
            if len(source_token_indices) == 1:
                is_used_direct_copy[source_token_indices[0]] = True
                num_direct_copy += 1
            else:
                for idx in source_token_indices:
                    is_used_weighted[idx] = True
                num_weighted_mean += 1
            source_weights = np.array(source_weights)
            np.add.at(src_weight_sums, source_token_indices, source_weights)
        if normalize:
            total_sum = src_weight_sums.sum()
            if total_sum > 0:
                src_weight_sums /= total_sum

        # How many source tokens contributed per target token
        plt.figure(figsize=fig_size)
        plt.hist(target_sources_count, bins=50, color="blue", alpha=0.7)
        plt.xlabel("Number of source tokens used for a single target token")
        plt.ylabel("Count of target tokens")
        plt.title("Distribution of Source Token Count per Target Token")
        plt.tight_layout()
        plt.savefig(f"{model_path}/hist_target_sources_count.png")
        if show_plot:
            plt.show()
        plt.close()

        # Contributions to target embeddings per source token (source tokens that were not used are filtered out)
        plt.figure(figsize=fig_size)
        plt.hist(src_weight_sums[src_usage_count > 0], bins=50, color="blue", alpha=0.7, log=True)
        plt.xlabel("Sum of source token weights")
        plt.ylabel("Count of source tokens (Log scale)")
        plt.title("Source Token Contributions for Embedding Initialization")
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"{model_path}/hist_source_embedding_contributions.png")
        if show_plot:
            plt.show()
        plt.close()

        # Top k source tokens with the highest contribution
        top_k_contribution_idx = np.argsort(src_weight_sums)[-top_k:]
        top_k_contribution = src_weight_sums[top_k_contribution_idx]
        top_k_contribution_tokens = [src_tokenizer.convert_ids_to_tokens(int(idx)) for idx in top_k_contribution_idx]

        # Not used: tokens where both is_used_direct_copy and is_used_weighted are False
        not_used_src_tokens = np.sum(np.logical_not(np.logical_or(is_used_direct_copy, is_used_weighted)))

        # Direct copy: tokens where is_used_direct_copy is True and is_used_weighted is False
        used_only_direct_copy_src_tokens = np.sum(np.logical_and(is_used_direct_copy, np.logical_not(is_used_weighted)))

        # Used weighted: tokens where is_used_weighted is True and is_used_direct_copy is False
        used_only_weighted_src_tokens = np.sum(np.logical_and(is_used_weighted, np.logical_not(is_used_direct_copy)))

        # Used both: tokens where both is_used_direct_copy and is_used_weighted are True
        used_both_src_tokens = np.sum(np.logical_and(is_used_direct_copy, is_used_weighted))

        with open(os.path.join(model_path, "transfer_information.json"), "r") as f:
            transfer_info = json.load(f)

        method_name = METHOD_MAP.get(model_dir, model_dir)
        src_emb_usage[method_name] = {
            "top_k_contribution": list(zip(top_k_contribution_tokens, top_k_contribution)),
            "used_only_weighted_src_tokens": used_only_weighted_src_tokens,
            "used_both_src_tokens": used_both_src_tokens,
            "used_only_direct_copy_src_tokens": used_only_direct_copy_src_tokens,
            "not_used_src_tokens": not_used_src_tokens,
            "src_weight_sums": src_weight_sums,
        }
        if num_weighted_mean + num_direct_copy != transfer_info["cleverly_initialized_tokens"]:
            logger.warning(
                f"Number of tokens initialized with clever initialization does not match for {model_dir}."
                f"Expected: {num_weighted_mean + num_direct_copy}, Transfer Info: {transfer_info['cleverly_initialized_tokens']}"
            )
        if num_rand_init != tgt_tokenizer.vocab_size - transfer_info["cleverly_initialized_tokens"]:
            logger.warning(
                f"Number of tokens initialized with random initialization does not match for {model_dir}."
                f"Expected: {num_rand_init}, Transfer Info: {tgt_tokenizer.vocab_size - transfer_info['cleverly_initialized_tokens']}"
            )
        src_emb_usage[method_name]["num_direct_copy"] = num_direct_copy
        src_emb_usage[method_name]["num_weighted_mean"] = num_weighted_mean
        src_emb_usage[method_name]["num_rand_init"] = num_rand_init


    with open(os.path.join(input_dir, "src_emb_usage.json"), "w") as f:
        f.write(json.dumps(src_emb_usage, cls=NpEncoder))

    # Create a bar plot for all models that shows the percentage of source tokens used, direct copies, and not used
    method_names = list(src_emb_usage.keys())
    # I want to sort the model names in a specific order like they are listed in the METHOD_MAP
    method_names = sorted(method_names, key=lambda m: list(METHOD_MAP.values()).index(m))
    # For multilingual case only keep FOCUS and FVT as the FVT variants have the same values
    if "FVT" in method_names:
        method_names = ["FVT", "FOCUS"]

    used_only_weighted_src_tokens = np.asarray(
        [src_emb_usage[model_name]["used_only_weighted_src_tokens"] for model_name in method_names]
    )
    used_both_src_tokens = np.asarray(
        [src_emb_usage[model_name]["used_both_src_tokens"] for model_name in method_names]
    )
    used_only_direct_copy_src_tokens = np.asarray(
        [src_emb_usage[model_name]["used_only_direct_copy_src_tokens"] for model_name in method_names]
    )
    not_used_src_tokens = np.asarray(
        [src_emb_usage[model_name]["not_used_src_tokens"] for model_name in method_names]
    )

    x = np.arange(len(method_names))
    width = 0.35
    color_used_weighted = "#1f77b4"  # Blue
    color_used_direct = "#ff7f0e"  # Orange
    color_not_used = "#7f7f7f"  # Gray

    fig, ax = plt.subplots(figsize=fig_size)

    # Bottom segment: used_non_direct
    bar_used_only_weighted_src_tokens = ax.bar(
        x,
        used_only_weighted_src_tokens,
        width,
        color=color_used_weighted,
        label="Used (Weighted Mean)"
    )

    # Middle segment: direct_copy_src_tokens
    bar_used_only_direct_copy_src_tokens = ax.bar(
        x,
        used_only_direct_copy_src_tokens,
        width,
        bottom=used_only_weighted_src_tokens,
        color=color_used_direct,
        label="Used (Direct Copy)"
    )

    # Middle segment: direct_copy_src_tokens
    bar_used_both_src_tokens = ax.bar(
        x,
        used_both_src_tokens,
        width,
        bottom=used_only_weighted_src_tokens + used_only_direct_copy_src_tokens,
        color=color_used_direct,
        hatch="//",
        edgecolor=color_used_weighted,
        label="Used (Direct Copy + Weighted Mean)"
    )

    # Top segment: not_used_src_tokens
    bar_not_used = ax.bar(
        x,
        not_used_src_tokens,
        width,
        bottom=used_only_weighted_src_tokens + used_only_direct_copy_src_tokens + used_both_src_tokens,
        color=color_not_used,
        label="Not Used"
    )

    ax.set_ylabel("Number of Source Tokens")
    ax.set_title("Source Token Usage")
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=45, ha="right")
    ax.legend(
        loc="upper left",  # Align the top-left corner of the legend...
        bbox_to_anchor=(1.05, 1)  # ...to this anchor point (slightly off to the right).
    )

    plt.grid(axis='y', alpha=0.3)  # you can configure grid lines to show only horizontal lines, etc.
    plt.tight_layout()
    plt.savefig(f"{input_dir}/src_token_usage.png")
    if show_plot:
        plt.show()
    plt.close()

    # Create a bar plot for all models that shows how many target embeddings were initialized randomly, with direct copy, and with weighted mean
    randomly_init_tgt_tokens = np.asarray(
        [src_emb_usage[model_name]["num_rand_init"] for model_name in method_names]
    )
    direct_copy_tgt_tokens = np.asarray(
        [src_emb_usage[model_name]["num_direct_copy"] for model_name in method_names]
    )
    weighted_mean_tgt_tokens = np.asarray(
        [src_emb_usage[model_name]["num_weighted_mean"] for model_name in method_names]
    )

    x = np.arange(len(method_names))
    width = 0.35
    color_random = "#1f77b4"  # Blue
    color_direct = "#ff7f0e"  # Orange
    color_weighted = "#2ca02c"  # Green

    fig, ax = plt.subplots(figsize=fig_size)

    # Bottom segment: randomly_init_tgt_tokens
    bar_randomly_init_tgt_tokens = ax.bar(
        x,
        randomly_init_tgt_tokens,
        width,
        color=color_random,
        label="Random Initialization"
    )

    # Middle segment: direct_copy_tgt_tokens
    bar_direct_copy_tgt_tokens = ax.bar(
        x,
        direct_copy_tgt_tokens,
        width,
        bottom=randomly_init_tgt_tokens,
        color=color_direct,
        label="Direct Copy"
    )

    # Top segment: weighted_mean_tgt_tokens
    bar_weighted_mean_tgt_tokens = ax.bar(
        x,
        weighted_mean_tgt_tokens,
        width,
        bottom=randomly_init_tgt_tokens + direct_copy_tgt_tokens,
        color=color_weighted,
        label="Weighted Mean Initialization"
    )

    ax.set_ylabel("Number of Target Embeddings")
    ax.set_title("Target Embedding Initialization")
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=45, ha="right")
    ax.legend(
        loc="upper left",  # Align the top-left corner of the legend...
        bbox_to_anchor=(1.05, 1)  # ...to this anchor point (slightly off to the right).
    )

    plt.grid(axis='y', alpha=0.3)  # you can configure grid lines to show only horizontal lines, etc.
    plt.tight_layout()
    plt.savefig(f"{input_dir}/tgt_embedding_init.png")
    if show_plot:
        plt.show()
    plt.close()


def main():
    args = parse_args()

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

    logger.info(f"Processing models in {args.input_dir}")
    visualize_transfer(
        args.input_dir,
        args.source_model_name_or_path,
        args.normalize,
        show_plot=args.show_plot,
        top_k=args.top_k
    )


if __name__ == "__main__":
    main()
