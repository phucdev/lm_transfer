import datasets
import logging
import argparse
import requests
import time

from pathlib import Path
from transformers import AutoTokenizer
from rich.logging import RichHandler

logger = logging.getLogger(__name__)
logger.addHandler(RichHandler())
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a tokenizer")
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        help="Name of the dataset to train the tokenizer on",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        help="Name of the dataset config to use",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory to cache the dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory to save the tokenizer",
        required=True,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Batch size to use for training the tokenizer",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=50000,
        help="Size of the vocabulary to use",
    )
    parser.add_argument(
        "--original_tokenizer",
        type=str,
        help="Name of the original tokenizer to use in order to use the same tokenization algorithm and special tokens",
        required=True,
    )
    arguments = parser.parse_args()
    return arguments


def batch_iterator(data, batch_size=1000, max_retries=3, retry_delay=15):
    # This function is used to iterate over the dataset when streaming
    batch = []
    idx = 0
    retries = 0

    while retries <= max_retries:
        try:
            for idx, example in enumerate(data):
                batch.append(example["text"])
                if len(batch) == batch_size:
                    yield batch
                    batch = []
            break  # Exit the loop if successful
        except requests.exceptions.ConnectionError:
            retries += 1
            if retries > max_retries:
                raise
            time.sleep(retry_delay)
            # Continue from the last successful index
            for c_idx, example in enumerate(data):
                if c_idx < idx:
                    continue
                batch.append(example["text"])
                if len(batch) == batch_size:
                    yield batch
                    batch = []
    if batch:  # yield last batch
        yield batch


def get_training_corpus(data, batch_size=1000):
    for start_idx in range(0, len(data), batch_size):
        samples = data[start_idx:start_idx+batch_size]
        yield samples["text"]


def main():
    args = parse_args()
    logger.info(f"Training tokenizer on {args.dataset_name_or_path} with vocab size {args.vocab_size}")
    # Local dataset
    if Path(args.dataset_name_or_path).exists():
        data = datasets.load_dataset("json", data_files=args.dataset_name_or_path, split="train")
        training_corpus = get_training_corpus(data)
    else:
        # Load dataset from the Hugging Face hub
        if args.cache_dir:
            data = datasets.load_dataset(
                args.dataset_name_or_path,
                args.dataset_config_name,
                split="train",
                cache_dir=args.cache_dir
            )
            training_corpus = batch_iterator(data)
        else:
            # Stream the dataset (default)
            data = datasets.load_dataset(
                args.dataset_name_or_path, args.dataset_config_name, split="train", streaming=True
            )
            training_corpus = get_training_corpus(data)

    old_tokenizer = AutoTokenizer.from_pretrained(args.original_tokenizer)
    assert old_tokenizer.is_fast, "This script only works with fast tokenizers"
    tokenizer = old_tokenizer.train_new_from_iterator(
        training_corpus,
        vocab_size=args.vocab_size,
        show_progress=True,
    )
    logger.info(f"Saving tokenizer to {args.output_dir}")
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
