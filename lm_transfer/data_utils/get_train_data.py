import json
import argparse
import sys
import datasets

from typing import Union
from tqdm.auto import tqdm
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="uonlp/CulturaX")
    parser.add_argument("--dataset_config_name", type=str, default="vi")
    parser.add_argument("--output_dir", type=str, default="data/culturax_vi")
    parser.add_argument("--subsample_size_mb", type=int, default=4096)
    parser.add_argument("--valid_percentage", type=int, default=10)
    parser.add_argument("--skip_download_and_split",  default=False, action="store_true")
    parser.add_argument("--preprocess_dataset", default=False, action="store_true")
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--preprocessing_num_workers", type=int, default=None)
    parser.add_argument("--only_keep_text", default=False, action="store_true")
    parser.add_argument("--quality_warning_filter", default=False, action="store_true",
                        help="Filter out documents with quality warnings from OSCAR")
    parser.add_argument("--oscar_2301_filter", default=False, action="store_true",
                        help="Only use OSCAR 23.01 documents from CulturaX")
    args = parser.parse_args()
    return args


def skip_entry(entry, args):
    if "oscar" in args.dataset_name and args.quality_warning_filter and entry["meta"]["quality_warnings"]:
        return True
    if args.dataset_name == "uonlp/CulturaX" and args.oscar_2301_filter and entry["source"] != "OSCAR-2301":
        return True
    return False


def get_size(obj):
    return sys.getsizeof(obj)


def create_split(file_path, dataset_iter, subsample_size_mb, args):
    with open(file_path, "w") as f:
        size = 0
        bar = tqdm(total=subsample_size_mb)

        while size < subsample_size_mb:
            entry = next(dataset_iter)

            if skip_entry(entry, args):
                continue

            entry_size = len(entry["text"].encode("utf-8"))
            # entry_size = get_size(entry["text"])
            size += entry_size

            bar.update(entry_size)

            if args.only_keep_text:
                entry = {
                    "text": entry["text"],
                }
            f.write(f"{json.dumps(entry, ensure_ascii=False)}\n")


def prepare_dataset():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    if not args.skip_download_and_split:
        # Adapted from https://github.com/CPJKU/wechsel/blob/main/legacy/prepare.py
        subsample_size = 1024 * 1024 * args.subsample_size_mb  # in bytes

        dataset = load_dataset(
            args.dataset_name, args.dataset_config_name, split="train", streaming=True
        )
        dataset_iter = iter(dataset)

        create_split(output_dir / "train.json", dataset_iter, subsample_size, args)
        create_split(output_dir / "valid.json", dataset_iter, subsample_size * args.valid_percentage / 100, args)

    if args.preprocess_dataset:
        preprocess_dataset(
            train_file=output_dir / "train.json",
            validation_file=output_dir / "valid.json",
            save_preprocessed_dataset_path=output_dir / "preprocessed",
            model_name_or_path=args.model_name_or_path,
            block_size=args.block_size,
            preprocessing_num_workers=args.preprocessing_num_workers
        )


def preprocess_dataset(train_file: Union[str, Path], validation_file: Union[str, Path],
                       save_preprocessed_dataset_path: Union[str, Path],
                       model_name_or_path: str, block_size: int, preprocessing_num_workers: int):
    train_file = str(train_file)
    validation_file = str(validation_file)
    save_preprocessed_dataset_path = str(save_preprocessed_dataset_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if block_size is None:
        block_size = tokenizer.model_max_length
    data_files = {"train": train_file, "validation": validation_file}
    extension = (
        train_file.split(".")[-1]
        if train_file is not None
        else validation_file.split(".")[-1]
    )
    if extension == "txt":
        extension = "text"
    dataset = datasets.load_dataset(
        extension,
        data_files=data_files
    )

    def preprocess_function(examples):
        return tokenizer([x for x in examples["text"]])

    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=preprocessing_num_workers,
        remove_columns=dataset["train"].column_names,
        desc="Running tokenizer on dataset",
    )

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of block_size.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        num_proc=preprocessing_num_workers,
        desc=f"Grouping texts in chunks of {block_size}"
    )
    lm_dataset.save_to_disk(save_preprocessed_dataset_path)


if __name__ == "__main__":
    prepare_dataset()
