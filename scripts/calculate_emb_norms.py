import argparse
import json
import logging
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import numpy as np

from pathlib import Path
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm.auto import tqdm

from lm_transfer.utils.utils import NpEncoder


logger = logging.getLogger(__name__)

MONOLINGUAL_MODEL_MAP = {
    "random_initialization": "roberta-random_init",
    "ramen_initialization": "roberta-ramen_init",
    "ramen_sparsemax_initialization": "roberta-ramen_sparsemax_init",
    "ramen_top_k_initialization": "roberta-ramen_top_k_init",
    "ramen_overlap_initialization": "roberta-ramen_overlap_init",
    "wechsel_initialization": "roberta-wechsel_init",
    "wechsel_overlap_initialization": "roberta-wechsel_overlap_init",
    "wechsel_aligned_initialization": "roberta-wechsel_aligned_overlap_init",
    "wechsel_rcsls_initialization": "roberta-wechsel_rcsls_init",
    "focus_monolingual_initialization": "roberta-focus_init",
}

MULTILINGUAL_MODEL_MAP = {
    "xlm-roberta-base": "xlm-roberta-lapt",
    "random_initialization": "xlm-roberta-random_init",
    "focus_multilingual_initialization": "xlm-roberta-focus_init",
    "fvt_initialization": "xlm-roberta-fvt_init",
    "fvt_unk_rand_initialization": "xlm-roberta-fvt_unk_rand_init",
    "fvt_minimize_punctuation_initialization": "xlm-roberta-fvt_minimize_punctuation_init",
    "fvt_freq_weighted_minimize_punctuation_initialization": "xlm-roberta-fvt_freq_weighted_minimize_punctuation_init",
    "fvt_rescale_initialization": "xlm-roberta-fvt_rescale_init",
    "fvt_subword_length_initialization": "xlm-roberta-fvt_subword_length_init",
    "xlm-roberta-zett-init": "xlm-roberta-zett_init",
}

METHOD_MAP = {
    "random_initialization": "RAND",
    "ramen_initialization": "R-RAMEN",
    "ramen_sparsemax_initialization": "R-RAMEN+Sparsemax",
    "ramen_top_k_initialization": "R-RAMEN+TopK",
    "ramen_overlap_initialization": "R-RAMEN+Overlap",
    "wechsel_initialization": "R-WECHSEL",
    "wechsel_overlap_initialization": "R-WECHSEL+Overlap",
    "wechsel_aligned_initialization": "R-WECHSEL+PreAligned+Overlap",
    "wechsel_rcsls_initialization": "R-WECHSEL+RCSLS",
    "focus_monolingual_initialization": "R-FOCUS",
    "xlm-roberta-base": "XLM-R",
    "fvt_initialization": "XLM-R-FVT",
    "fvt_unk_rand_initialization": "XLM-R-FVT+UnkRand",
    "fvt_minimize_punctuation_initialization": "XLM-R-FVT+MinPunct",
    "fvt_subword_length_initialization": "XLM-R-FVT+SubwordLength",
    "fvt_rescale_initialization": "XLM-R-FVT+Rescale",
    "fvt_freq_weighted_minimize_punctuation_initialization": "XLM-R-FVT+FreqWeighted+MinPunct",
    "focus_multilingual_initialization": "XLM-R-FOCUS",
    "xlm-roberta-zett_init": "XLM-R-ZeTT",
}

METHOD_NAMES = [
    "R-RAND", "R-RAMEN", "R-RAMEN+Overlap", "R-WECHSEL", "R-WECHSEL+Overlap", "R-WECHSEL+PreAligned+Overlap",
    "R-WECHSEL+RCSLS", "R-FOCUS", "XLM-R", "XLM-R-RAND", "XLM-R-FVT", "XLM-R-FVT+MinPunct", "XLM-R-FVT+SubwordLength",
    "XLM-R-FVT+Rescale", "XLM-R-FVT+FreqWeighted+MinPunct", "XLM-R-FOCUS", "XLM-R-ZeTT"
]


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate the norms of the embeddings of a model.")
    parser.add_argument("--input_dir_before", type=str, default=None, help="The directory containing the models before LAPT.")
    parser.add_argument("--input_dir_after", type=str, default=None, help="The directory containing the models after LAPT.")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="The model name or path.")
    parser.add_argument("--output_dir", type=str, default=None, help="The output directory.")
    parser.add_argument("--is_source_model", action="store_true",default=False, help="Whether the model is a source model.")
    parser.add_argument("--use_serif_font", action="store_true", default=False, help="Use the Source Serif font.")
    parser.add_argument("--xlim", type=float, default=None, help="The x-axis limit for the histogram.")
    parser.add_argument("--ylim", type=float, default=None, help="The y-axis limit for the histogram.")
    parser.add_argument("--log", action="store_true", default=False, help="Use log scale for the histogram.")
    parser.add_argument("--show_plot", action="store_true", default=False, help="Show the plot.")
    parser.add_argument("--fig_size", type=int, nargs=2, default=[10, 10], help="The size of the plot")
    args = parser.parse_args()
    return args

def calculate_embedding_norms(model_path):
    model = AutoModelForMaskedLM.from_pretrained(model_path)
    embeddings = model.get_input_embeddings()
    embedding_norms = embeddings.weight.norm(dim=1).detach().numpy()
    return embedding_norms


def extract_sources(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    randomly_initialized_idx = list(range(tokenizer.vocab_size))
    direct_copies_idx = []
    cleverly_initialized_idx = []
    if Path(f"{model_path}/sources.json").exists():
        with open(f"{model_path}/sources.json", "r") as f:
            sources = json.load(f)
        for target_token, source_info in sources.items():
            target_token_id = tokenizer.convert_tokens_to_ids(target_token)
            source_tokens = source_info[0]
            if len(source_tokens) == 1:
                direct_copies_idx.append(target_token_id)
            else:
                cleverly_initialized_idx.append(target_token_id)
            randomly_initialized_idx.remove(target_token_id)
    return {
        "randomly_initialized_idx": randomly_initialized_idx,
        "direct_copies_idx": direct_copies_idx,
        "cleverly_initialized_idx": cleverly_initialized_idx
    }


def plot_embedding_norms(
        model_path,
        output_dir,
        is_source_model=False,
        xlim=None,
        ylim=None,
        log=False,
        show_plot=False,
        fig_size=(10, 6)
):
    """
    Plot the distribution of the norms of the embeddings of a single model.
    :param model_path:
    :param output_dir:
    :param is_source_model:
    :param xlim:
    :param ylim:
    :param log:
    :param show_plot:
    :param fig_size:
    :return:
    """
    token_indices = extract_sources(model_path)

    embedding_norms = calculate_embedding_norms(model_path)
    with open(f"{output_dir}/embedding_norms.json", "w") as f:
        f.write(json.dumps(embedding_norms.tolist(), cls=NpEncoder))

    randomly_initialized_norms = embedding_norms[token_indices["randomly_initialized_idx"]]
    direct_copies_norms = embedding_norms[token_indices["direct_copies_idx"]]
    cleverly_initialized_norms = embedding_norms[token_indices["cleverly_initialized_idx"]]

    # Handle cases where some categories may be empty
    all_norms = [
        randomly_initialized_norms if len(randomly_initialized_norms) > 0 else np.array([]),
        direct_copies_norms if len(direct_copies_norms) > 0 else np.array([]),
        cleverly_initialized_norms if len(cleverly_initialized_norms) > 0 else np.array([]),
    ]

    # Colors and labels for the histogram
    colors = ['blue', 'green', 'orange']
    labels = ['Randomly Initialized', 'Direct Copies', 'Cleverly Initialized']

    # Filter out empty categories for plotting
    filtered_norms = [norms for norms in all_norms if len(norms) > 0]
    filtered_colors = [colors[i] for i, norms in enumerate(all_norms) if len(norms) > 0]
    filtered_labels = [labels[i] for i, norms in enumerate(all_norms) if len(norms) > 0]

    # Plot stacked histogram
    plt.figure(figsize=fig_size)
    if token_indices["direct_copies_idx"] or token_indices["cleverly_initialized_idx"]:
        plt.hist(filtered_norms, bins=50, stacked=True, color=filtered_colors, label=filtered_labels, alpha=0.7,
                 log=log)
    else:
        plt.hist(filtered_norms, bins=50, color=filtered_colors, alpha=0.7, log=log)
    plt.title("Distribution of Embedding Norms")
    plt.xlabel("Norm")
    plt.ylabel("Frequency")

    if xlim is not None:
        plt.xlim(0, xlim)   # 4.5 for monolingual, 7.5 for multilingual
    if ylim is not None:
        plt.ylim(0, ylim)   # 12500 for monolingual, 27000 for multilingual
    plt.grid()
    if not is_source_model and (token_indices["direct_copies_idx"] or token_indices["cleverly_initialized_idx"]):
        plt.legend()
    plt.tight_layout()

    plt.savefig(f"{output_dir}/embedding_norms.pdf")
    if show_plot:
        plt.show()


def plot_multi_embedding_norms(
        models_before, models_after, output_path, xlim=None, ylim=None, log=False, fig_size=(10, 6)
):
    num_models = len(models_before)
    fig, axes = plt.subplots(num_models, 2, figsize=(10, 4 * num_models))

    # Colors and labels for the histogram
    colors = ['blue', 'green', 'orange']
    labels = ['Randomly Initialized', 'Direct Copies', 'Cleverly Initialized']
    model_names = [model_name for model_name in METHOD_NAMES if model_name in models_before.keys()]

    for i, model_name in enumerate(model_names):
        norms_before = models_before[model_name]["embedding_norms"]
        norms_after = models_after[model_name]["embedding_norms"]
        token_indices = models_before[model_name]["sources"]

        randomly_initialized_norms_before = norms_before[token_indices["randomly_initialized_idx"]]
        direct_copies_norms_before = norms_before[token_indices["direct_copies_idx"]]
        cleverly_initialized_norms_before = norms_before[token_indices["cleverly_initialized_idx"]]

        # Handle cases where some categories may be empty
        all_norms_before = [
            randomly_initialized_norms_before if len(randomly_initialized_norms_before) > 0 else np.array([]),
            direct_copies_norms_before if len(direct_copies_norms_before) > 0 else np.array([]),
            cleverly_initialized_norms_before if len(cleverly_initialized_norms_before) > 0 else np.array([]),
        ]
        # Filter out empty categories for plotting
        filtered_norms_before = [norms for norms in all_norms_before if len(norms) > 0]
        filtered_colors_before = [colors[i] for i, norms in enumerate(all_norms_before) if len(norms) > 0]
        filtered_labels_before = [labels[i] for i, norms in enumerate(all_norms_before) if len(norms) > 0]

        randomly_initialized_norms_after = norms_after[token_indices["randomly_initialized_idx"]]
        direct_copies_norms_after = norms_after[token_indices["direct_copies_idx"]]
        cleverly_initialized_norms_after = norms_after[token_indices["cleverly_initialized_idx"]]
        all_norms_after = [
            randomly_initialized_norms_after if len(randomly_initialized_norms_after) > 0 else np.array([]),
            direct_copies_norms_after if len(direct_copies_norms_after) > 0 else np.array([]),
            cleverly_initialized_norms_after if len(cleverly_initialized_norms_after) > 0 else np.array([]),
        ]
        filtered_norms_after = [norms for norms in all_norms_after if len(norms) > 0]
        filtered_colors_after = [colors[i] for i, norms in enumerate(all_norms_after) if len(norms) > 0]
        filtered_labels_after = [labels[i] for i, norms in enumerate(all_norms_after) if len(norms) > 0]

        for j, (norms, color, label, title) in enumerate(
                zip(
                    [filtered_norms_before, filtered_norms_after],
                    [filtered_colors_before, filtered_colors_after],
                    [filtered_labels_before, filtered_labels_after],
                    ["Before LAPT", "After LAPT"]
                )
        ):
            ax = axes[i, j] if num_models > 1 else axes[j]
            ax.hist(norms, bins=50, color=color, label=label, alpha=0.7, log=log)
            ax.set_title(f"{model_name} - {title}")
            ax.set_xlabel("Norm")
            ax.set_ylabel("Frequency")
            if xlim:
                ax.set_xlim(0, xlim)
            if ylim:
                ax.set_ylim(0, ylim)
            ax.grid()
            if len(label) > 1:
                ax.legend()

        # Set ylim to the max y of before and after
        if ylim is None:
            max_y = max([ax.get_ylim()[1] for ax in axes[i]])
            for ax in axes[i]:
                ax.set_ylim(0, max_y)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


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

    if args.model_name_or_path is not None and args.output_dir is not None:
        # Plot the norms of the embeddings of a single model
        logger.info(f"Processing model {args.model_name_or_path}")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        plot_embedding_norms(
            args.model_name_or_path, args.output_dir, args.is_source_model, args.xlim, args.ylim, args.log, args.show_plot, args.fig_size
        )
    elif args.input_dir_before is not None and args.input_dir_after is not None and args.output_dir is not None:
        # Plot the norms of the embeddings of multiple models
        logger.info(f"Processing models in {args.input_dir_before} and {args.input_dir_after}")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        models_before = {}
        models_after = {}
        for model_name in tqdm(os.listdir(args.input_dir_before), desc="Processing models"):
            model_path_before = os.path.join(args.input_dir_before, model_name)
            mapped_model_name = MONOLINGUAL_MODEL_MAP.get(model_name, model_name) if "monolingual" in args.input_dir_before else MULTILINGUAL_MODEL_MAP.get(model_name, model_name)
            model_path_after = os.path.join(args.input_dir_after, mapped_model_name)
            if os.path.isdir(model_path_before) and os.path.isdir(model_path_after):
                logger.info(f"Processing model {model_name}")
                method_name = METHOD_MAP.get(model_name, model_name)
                if method_name == "RAND":
                    if "monolingual" in args.input_dir_before:
                        method_name = "R-RAND"
                    else:
                        method_name = "XLM-R-RAND"
                embedding_norms_before = calculate_embedding_norms(model_path_before)
                embedding_norms_after = calculate_embedding_norms(model_path_after)
                models_before[method_name] = {"embedding_norms": embedding_norms_before}
                models_after[method_name] = {"embedding_norms": embedding_norms_after}
                sources_before = extract_sources(model_path_before)
                sources_after = extract_sources(model_path_after)
                models_before[method_name]["sources"] = sources_before
                models_after[method_name]["sources"] = sources_after
            else:
                logger.warning(f"Skipping {model_name} as it is not a directory.")
        plot_multi_embedding_norms(
            models_before, models_after, os.path.join(args.output_dir, "embedding_norms.pdf"), args.xlim, args.ylim, args.log, args.fig_size
        )
    else:
        raise ValueError("Please provide either --input_dir or both --model_name_or_path and --output_dir")


if __name__ == "__main__":
    main()
