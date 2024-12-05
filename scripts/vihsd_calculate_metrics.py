import datasets
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report


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


def main():
    # The original run_classification.py script did not calculate the metrics for --do_predict step
    # We need to calculate the metrics for the predictions
    vihsd_test = datasets.load_dataset("phucdev/ViHSD", split="test")
    gold_labels = vihsd_test["label_id"]

    # for the predictions
    task_path = "results/extracted_results/vihsd"
    for model in os.listdir(task_path):
        model_path = os.path.join(task_path, model)
        for run in os.listdir(model_path):
            run_path = os.path.join(model_path, run)
            predictions_path = os.path.join(run_path, "predict_results.txt")
            predictions = pd.read_csv(predictions_path, delimiter="\t")
            predicted_labels = list(predictions["prediction"])
            class_report = classification_report(gold_labels, predicted_labels, output_dict=True)
            accuracy = class_report["accuracy"]
            macro_f1 = class_report["macro avg"]["f1-score"]
            weighted_f1 = class_report["weighted avg"]["f1-score"]
            results_file = os.path.join(run_path, "all_results.json")
            with open(results_file, mode="r") as f:
                print("Processing", results_file)
                results = json.load(f)
                results["test_accuracy"] = accuracy
                results["test_macro_f1"] = macro_f1
                results["test_weighted_f1"] = weighted_f1
            with open(results_file, mode="w") as f:
                try:
                    json.dump(results, f, indent=2, cls=NpEncoder)
                except Exception as e:
                    print("Error while saving the results:", e)


if __name__ == "__main__":
    main()
