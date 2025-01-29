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
from tqdm import tqdm

from lm_transfer.utils.utils import NpEncoder


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate the norms of the embeddings of a model.")
    parser.add_argument("--input_dir", type=str, default=None, help="The directory containing the models.")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="The model name or path.")
    parser.add_argument("--output_dir", type=str, default=None, help="The output directory.")
    parser.add_argument("--is_source_model", action="store_true",default=False, help="Whether the model is a source model.")
    parser.add_argument("--use_serif_font", action="store_true", default=False, help="Use the Source Serif font.")
    parser.add_argument("--xlim", type=float, default=None, help="The x-axis limit for the histogram.")
    parser.add_argument("--ylim", type=float, default=None, help="The y-axis limit for the histogram.")
    parser.add_argument("--log", action="store_true", default=False, help="Use log scale for the histogram.")
    parser.add_argument("--show_plot", action="store_true", default=False, help="Show the plot.")
    args = parser.parse_args()
    return args


def calculate_embedding_norms(
        model_path, output_dir, is_source_model=False, xlim=None, ylim=None, log=False, show_plot=False
):
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

    model = AutoModelForMaskedLM.from_pretrained(model_path)
    embeddings = model.get_input_embeddings()

    embedding_norms = embeddings.weight.norm(dim=1).detach().numpy()
    with open(f"{output_dir}/embedding_norms.json", "w") as f:
        f.write(json.dumps(embedding_norms.tolist(), cls=NpEncoder))

    randomly_initialized_norms = embedding_norms[randomly_initialized_idx]
    direct_copies_norms = embedding_norms[direct_copies_idx]
    cleverly_initialized_norms = embedding_norms[cleverly_initialized_idx]

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
    plt.figure(figsize=(10, 6))
    if direct_copies_idx or cleverly_initialized_idx:
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
    if not is_source_model and (direct_copies_idx or cleverly_initialized_idx):
        plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/embedding_norms.png")
    if show_plot:
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

    if args.input_dir is not None:
        logger.info(f"Processing models in {args.input_dir}")
        for model_dir in tqdm(os.listdir(args.input_dir), desc="Calculating & Plotting Embedding Norms"):
            model_path = os.path.join(args.input_dir, model_dir)
            if not os.path.isdir(model_path):
                continue
            calculate_embedding_norms(
                model_path, model_path, args.is_source_model, args.xlim, args.ylim, args.log, args.show_plot
            )
    elif args.model_name_or_path is not None and args.output_dir is not None:
        logger.info(f"Processing model {args.model_name_or_path}")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        calculate_embedding_norms(
            args.model_name_or_path, args.output_dir, args.is_source_model, args.xlim, args.ylim, args.log, args.show_plot
        )
    else:
        raise ValueError("Please provide either --input_dir or both --model_name_or_path and --output_dir")


if __name__ == "__main__":
    main()
