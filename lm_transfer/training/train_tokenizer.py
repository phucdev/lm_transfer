import datasets
import logging
import argparse
import requests
import time
import numpy as np

from pathlib import Path
from tokenizers import pre_tokenizers, normalizers
from datasets import IterableDataset
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
        "--max_num_space_separated_tokens",
        type=int,
        default=None,
        help="Filter out long texts from the dataset (per default: > 32K space separated tokens)",
    )
    parser.add_argument(
        "--max_num_bytes",
        type=int,
        default=None,
        help="Filter out long texts from the dataset (length measured in bytes)",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=4,
        help="Number of processes to use when filtering out long texts from the dataset",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        default=False,
        help="Stream the dataset from the Hugging Face hub",
    )
    parser.add_argument(
        "--original_tokenizer",
        type=str,
        help="Name of the original tokenizer to use in order to use the same tokenization algorithm and special tokens",
        required=True,
    )
    arguments = parser.parse_args()
    return arguments


def streaming_batch_iterator(
        data: IterableDataset,
        batch_size: int = 1000,
        max_retries: int = 3,
        retry_delay: int = 15,
        max_num_space_separated_tokens: int = None,
        max_num_bytes: int = None
):
    # This function is used to iterate over the dataset when streaming
    batch = []
    retries = 0
    state_dict = None

    while retries <= max_retries:
        try:
            if state_dict:
                data.load_state_dict(state_dict)
            for idx, example in enumerate(data):
                if (max_num_space_separated_tokens and
                        0 < max_num_space_separated_tokens < len(example["text"].split())):
                    continue
                elif max_num_bytes and sentence_length_in_bytes(example["text"]) > max_num_bytes:
                    continue
                batch.append(example["text"])
                state_dict = data.state_dict()
                if len(batch) == batch_size:
                    yield batch
                    batch = []
            break  # Exit the loop if successful
        except requests.exceptions.ConnectionError:
            retries += 1
            if retries > max_retries:
                logger.error(f"Failed to connect to the server after {max_retries} retries.")
                raise
            logger.warning(f"Connection error. Retrying in {retry_delay}s [{retries}/{max_retries}]")
            time.sleep(retry_delay)
    if batch:  # yield last batch
        yield batch


def batch_iterator(data, batch_size=1000):
    for start_idx in range(0, len(data), batch_size):
        samples = data[start_idx:start_idx + batch_size]
        yield samples["text"]


def sentence_length_in_bytes(sentence):
    # Encode the sentence into bytes using UTF-8 encoding
    byte_representation = sentence.encode('utf-8')
    # Return the length of the byte array
    return len(byte_representation)


def get_num_words(data):
    def compute_length(example):
        return {"num_words": len(example["text"].split())}

    data_with_lengths = data.map(compute_length)
    num_words = np.array(data_with_lengths["num_words"])
    num_words.sort()
    return list(num_words)


def main():
    args = parse_args()
    logger.info(f"Training tokenizer on {args.dataset_name_or_path} with vocab size {args.vocab_size}")
    if args.stream:
        # Stream the dataset for training the tokenizer
        logger.info(f"Streaming {args.dataset_name_or_path} from the Hugging Face hub")
        data = datasets.load_dataset(
            args.dataset_name_or_path, args.dataset_config_name, split="train", streaming=True
        )
        training_corpus = streaming_batch_iterator(
            data,
            batch_size=args.batch_size,
            max_num_space_separated_tokens=args.max_num_space_separated_tokens,
            max_num_bytes=args.max_num_bytes,
        )
    else:
        # Use local dataset/ load the entire dataset from the Hugging Face hub
        if Path(args.dataset_name_or_path).exists():
            logger.info(f"Loading dataset from local file: {args.dataset_name_or_path}")
            data = datasets.load_dataset("json", data_files=args.dataset_name_or_path, split="train")
        else:
            logger.info(f"Loading {args.dataset_name_or_path} dataset from the Hugging Face hub")
            data = datasets.load_dataset(
                args.dataset_name_or_path,
                args.dataset_config_name,
                split="train",
                cache_dir=args.cache_dir
            )
        original_size = len(data)
        if args.max_num_space_separated_tokens and args.max_num_space_separated_tokens > 0:
            data = data.filter(lambda x: len(x["text"].split()) <= args.max_num_space_separated_tokens,
                               num_proc=args.num_proc)
            logger.info(
                f"Filtered out long texts with >{args.max_num_space_separated_tokens} space separated tokens. "
                f"Original dataset size: {original_size}. New dataset size: {len(data)}")
        elif args.max_num_bytes and args.max_num_bytes > 0:
            data = data.filter(lambda x: sentence_length_in_bytes(x["text"]) <= args.max_num_bytes,
                               num_proc=args.num_proc)
            logger.info(
                f"Filtered out long texts with >{args.max_num_bytes} bytes. "
                f"Original dataset size: {original_size}. New dataset size: {len(data)}")

        training_corpus = batch_iterator(data, batch_size=args.batch_size)

    old_tokenizer = AutoTokenizer.from_pretrained(args.original_tokenizer)
    assert old_tokenizer.is_fast, "This script only works with fast tokenizers"

    logger.info(f"Training new tokenizer based on {args.original_tokenizer}")
    tokenizer = old_tokenizer.train_new_from_iterator(
        training_corpus,
        vocab_size=args.vocab_size,
        show_progress=True,
    )
    logger.info(f"Saving tokenizer to {args.output_dir}")
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
