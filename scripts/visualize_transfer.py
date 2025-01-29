import argparse
import json
import logging
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import numpy as np

from transformers import AutoTokenizer

from lm_transfer.utils.utils import NpEncoder


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    for model_dir in os.listdir(input_dir):
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
        for target_token, source_info in sources.items():
            source_token_indices = source_info[1]
            source_weights = source_info[2]
            target_sources_count.append(len(source_token_indices))
            # Accumulate usage counts
            for idx in source_token_indices:
                src_usage_count[idx] += 1
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
        plt.hist(src_weight_sums[src_usage_count > 0], bins=50, color="blue", alpha=0.7)
        plt.xlabel("Sum of source token contributions to target embeddings")
        plt.ylabel("Count of source tokens")
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

        # TODO This information can be taken from the transfer_information.json file or simply follow the
        #  calculate_emb_norms.py code. The numbers here are somehow not correct
        used_src_tokens = (src_weight_sums > 0).sum()
        direct_copy_src_tokens = (src_usage_count == 1).sum()
        not_used_src_tokens = (src_usage_count == 0).sum()

        src_emb_usage[model_dir] = {
            "top_k_contribution": list(zip(top_k_contribution_tokens, top_k_contribution)),
            "used_src_tokens": used_src_tokens,
            "direct_copy_src_tokens": direct_copy_src_tokens,
            "not_used_src_tokens": not_used_src_tokens,
            "percentage_used": used_src_tokens / src_tokenizer.vocab_size,
            "percentage_direct_copy": direct_copy_src_tokens / src_tokenizer.vocab_size,
            "percentage_not_used": not_used_src_tokens / src_tokenizer.vocab_size,
            "src_weight_sums": src_weight_sums,
        }

    with open(os.path.join(input_dir, "src_emb_usage.json"), "w") as f:
        f.write(json.dumps(src_emb_usage, cls=NpEncoder))

    # Create a bar plot for all models that shows the percentage of source tokens used, direct copies, and not used
    model_names = list(src_emb_usage.keys())
    # TODO: map to pretty model names and for FVT we can skip the variants since they are the same
    used_src_tokens = [src_emb_usage[model_name]["used_src_tokens"] for model_name in model_names]
    direct_copy_src_tokens = [src_emb_usage[model_name]["direct_copy_src_tokens"] for model_name in model_names]
    not_used_src_tokens = [src_emb_usage[model_name]["not_used_src_tokens"] for model_name in model_names]
    used_non_direct = np.array(used_src_tokens) - np.array(direct_copy_src_tokens)

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=fig_size)

    # Bottom segment: used_non_direct
    bar_used_non_direct = ax.bar(
        x,
        used_non_direct,
        width,
        label="Used (Non-Direct Copy)"
    )

    # Middle segment: direct_copy_src_tokens
    bar_direct_copy = ax.bar(
        x,
        direct_copy_src_tokens,
        width,
        bottom=used_non_direct,
        label="Used (Direct Copy)"
    )

    # Top segment: not_used_src_tokens
    bar_not_used = ax.bar(
        x,
        not_used_src_tokens,
        width,
        bottom=used_non_direct + direct_copy_src_tokens,
        label="Not Used"
    )

    ax.set_ylabel("Number of Source Tokens")
    ax.set_title("Source Token Usage")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.legend()

    plt.grid(axis='y', alpha=0.3)  # you can configure grid lines to show only horizontal lines, etc.
    plt.tight_layout()
    plt.savefig(f"{input_dir}/src_token_usage.png")
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
        logger.info(prop.get_name())
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
