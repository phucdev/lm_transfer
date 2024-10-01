import argparse
import json
import numpy as np
import plotly.express as px
from datasets import load_dataset
from pathlib import Path


class NpEncoder(json.JSONEncoder):
    """
    JSON Encoder that can handle numpy types
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def parse_args():
    parser = argparse.ArgumentParser(description="Collect statistics about the length of text examples in a dataset.")
    parser.add_argument("--dataset_name_or_path", type=str, required=True,
                        help="Name of the dataset to load from the Hugging Face hub.")
    parser.add_argument("--dataset_config_name", type=str, default=None, help="Configuration name of the dataset.")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for the dataset.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory path for saving the statistics and the plots.")
    parser.add_argument("--num_proc", type=int, default=16, help="Number of processes to use for mapping.")
    parser.add_argument("--cleanup_cache_files", action="store_true", default=False,
                        help="Whether to clean up the cache files after the analysis.")
    parser.add_argument("--")

    return parser.parse_args()


def compute_length(example):
    return {
        "num_chars": len(example["text"]),
        "num_words": len(example["text"].split()),
        "num_bytes": len(example["text"].encode("utf-8")),
    }


def plot_statistics(lengths, output_file):
    fig = px.histogram(lengths, nbins=50, title='Distribution of Text Lengths')
    fig.update_layout(xaxis_title='Length', yaxis_title='Frequency')
    fig.write_image(output_file)


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load dataset
    data = load_dataset(args.dataset_name_or_path, args.dataset_config_name, split="train", cache_dir=args.cache_dir)

    # Collect statistics
    lengths = data.map(compute_length, num_proc=args.num_proc).remove_columns(["text"])

    for length_type in ["num_chars", "num_words", "num_bytes"]:
        print(f"Statistics for {length_type}:")
        np_array = np.array(lengths[length_type])
        mean_length = np.mean(np_array)
        median_length = np.median(np_array)
        max_length = np.max(np_array)
        min_length = np.min(np_array)
        print(f"Mean length: {mean_length}")
        print(f"Median length: {median_length}")
        print(f"Max length: {max_length}")
        print(f"Min length: {min_length}")

        with open(Path(args.output_dir) / f"{length_type}_statistics.json", "w") as f:
            stat_dict = {
                "mean_length": mean_length,
                "median_length": median_length,
                "max_length": max_length,
                "min_length": min_length,
            }
            f.write(json.dumps(stat_dict, indent=2, cls=NpEncoder))

        try:
            plot_statistics(
                lengths=np_array,
                output_file=Path(args.output_dir) / f"{length_type}_distribution.png"
            )
        except Exception as e:
            print("Retry plotting the data with down sampled data")
            plot_statistics(
                lengths=np.random.choice(np_array, min(1000000, len(np_array)), replace=False),
                output_file=Path(args.output_dir) / f"{length_type}_distribution.png"
            )

    total_num_bytes = np.sum(np.array(lengths["num_bytes"]))

    # Clean up cache files
    if args.cleanup_cache_files:
        print("Cleaning up cache files")
        data.cleanup_cache_files()


if __name__ == "__main__":
    main()
