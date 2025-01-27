import argparse
import json
import logging
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import numpy as np

from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize the transfer of a model.")
    parser.add_argument("--input_dir", type=str, default=None, help="The directory containing the models.")
    parser.add_argument("--source_model_name_or_path", type=str, default=None, help="The source model name or path.")
    parser.add_argument("--normalize", action="store_true", default=False, help="Normalize the sum of weights.")
    parser.add_argument("--use_serif_font", action="store_true", default=False, help="Use the Source Serif font.")

    args = parser.parse_args()
    return args


def visualize_transfer(model_dir, source_model_name_or_path, normalize=False):
    if Path(model_dir).joinpath("sources.json").exists():
        with open(Path(model_dir).joinpath("sources.json"), "r") as f:
            sources = json.load(f)
        src_tokenizer = AutoTokenizer.from_pretrained(source_model_name_or_path)
        src_weight_sums = np.zeros(src_tokenizer.vocab_size)
        for target_token, source_info in sources.items():
            source_token_indices = source_info[1]
            source_weights = source_info[2]
            source_weights = np.array(source_weights)
            for source_token_id, source_weight in zip(source_token_indices, source_weights):
                src_weight_sums[source_token_id] += source_weight
        if normalize:
            src_weight_sums /= src_weight_sums.sum()
        plt.figure(figsize=(10, 5))
        plt.bar(range(src_tokenizer.vocab_size), src_weight_sums)
        plt.xlabel("Source token index")
        plt.ylabel("Sum of weights")
        plt.title("Sum of weights of source tokens for transfer")
        plt.savefig(f"{model_dir}/src_embedding_weight_sums.png")
        plt.show()
    else:
        logger.warning(f"No sources.json file found in {model_dir}.")


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
    for model_dir in tqdm(os.listdir(args.input_dir), desc="Plotting source embedding contribution"):
        model_path = os.path.join(args.input_dir, model_dir)
        if not os.path.isdir(model_path):
            continue
        visualize_transfer(
            model_path, args.source_model_name_or_path, args.normalize
        )


if __name__ == "__main__":
    main()
