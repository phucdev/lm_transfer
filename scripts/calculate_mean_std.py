import os
import json
import argparse
import numpy as np

from typing import List


TASK_TO_METRICS = {
    "xnli": ["accuracy"],
    "vihsd": ["accuracy", "macro_f1", "weighted_f1"],
    "phoner": ["accuracy", "f1"],
    "mlqa": ["f1", "exact_match"],
}


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
    parser = argparse.ArgumentParser(description="Calculate mean and standard deviation of task metrics from multiple files.")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing the files.")
    parser.add_argument("--task", type=str, required=True, help="Name of the task for selecting the metrics.")
    parser.add_argument("--output_file", type=str, default=None, help="Output file to save the results.")
    return parser.parse_args()


def calculate_mean_and_std(input_dir, metrics: List[str]):
    metrics_values = {metric: [] for metric in metrics}
    results = {}

    # Iterate over all files in the input directory
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file == "predict_results.json":
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    results = json.load(f)
                    for metric in metrics:
                        if metric in results:
                            metrics_values[metric].append(results[metric])

    for metric in metrics:
        values = metrics_values[metric]
        mean = np.mean(values)
        std = np.std(values)
        print(f"{metric}: Mean = {mean:.4f}, Std = {std:.4f}")
        results[metric] = {"mean": mean, "std": std}
    return results


def main():
    args = parse_args()
    task = args.task
    metrics = TASK_TO_METRICS[task]
    results = calculate_mean_and_std(args.input_dir, metrics)
    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write(json.dumps(results, indent=4, cls=NpEncoder))


if __name__ == "__main__":
    main()
