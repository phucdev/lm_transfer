import argparse
import json
import logging
import matplotlib.pyplot as plt
import os

from pathlib import Path
from transformers import AutoModelForMaskedLM
from tqdm import tqdm

from lm_transfer.utils.utils import NpEncoder


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate the norms of the embeddings of a model.")
    parser.add_argument("--input_dir", type=str, default=None, help="The directory containing the models.")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="The model name or path.")
    parser.add_argument("--output_dir", type=str, default=None, help="The output directory.")
    args = parser.parse_args()
    return args


def calculate_embedding_norms(model_path, output_dir):
    model = AutoModelForMaskedLM.from_pretrained(model_path)
    embeddings = model.get_input_embeddings()
    embedding_norms = embeddings.weight.norm(dim=1).detach().numpy()
    with open(f"{output_dir}/embedding_norms.json", "w") as f:
        f.write(json.dumps(embedding_norms.tolist(), cls=NpEncoder))

    # Plot the norms
    plt.figure(figsize=(10, 6))
    plt.hist(embedding_norms, bins=50, alpha=0.7, color='blue')
    plt.title("Distribution of Embedding Norms")
    plt.xlabel("Norm")
    plt.ylabel("Frequency")
    plt.grid()
    plt.savefig(f"{output_dir}/embedding_norms.png")


def main():
    args = parse_args()

    if args.input_dir is not None:
        logger.info(f"Processing models in {args.input_dir}")
        for model_dir in tqdm(os.listdir(args.input_dir), desc="Calculating & Plotting Embedding Norms"):
            model_path = os.path.join(args.input_dir, model_dir)
            if not os.path.isdir(model_path):
                continue
            calculate_embedding_norms(model_path, model_path)
    elif args.model_name_or_path is not None and args.output_dir is not None:
        logger.info(f"Processing model {args.model_name_or_path}")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        calculate_embedding_norms(args.model_name_or_path, args.output_dir)
    else:
        raise ValueError("Please provide either --input_dir or both --model_name_or_path and --output_dir")


if __name__ == "__main__":
    main()
