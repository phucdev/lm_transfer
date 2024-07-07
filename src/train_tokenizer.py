import datasets
import logging
import argparse

from transformers import AutoTokenizer
from rich.logging import RichHandler

logger = logging.getLogger(__name__)
logger.addHandler(RichHandler())
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a tokenizer")
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Name of the dataset to train the tokenizer on",
        required=True,
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        help="Name of the dataset config to use",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory to save the tokenizer",
        required=True,
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        help="Size of the vocabulary to use",
        required=True,
    )
    parser.add_argument(
        "--original_tokenizer",
        type=str,
        help="Name of the original tokenizer to use in order to use the same tokenization algorithm and special tokens",
        required=True,
    )
    arguments = parser.parse_args()
    return arguments


def main():
    args = parse_args()
    logger.info(f"Training tokenizer on {args.dataset_name} with vocab size {args.vocab_size}")
    # TODO: maybe merge multiple datasets, e.g. OSCAR, Wikipedia, Social Media Text Corpus from ViSoBERT, Binhvq news
    data = datasets.load_dataset(
        args.dataset_name, args.dataset_config_name, split="train", streaming=True
    )

    def batch_iterator(batch_size=1000):
        batch = []
        for example in data:
            batch.append(example["text"])
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:  # yield last batch
            yield batch

    old_tokenizer = AutoTokenizer.from_pretrained(args.original_tokenizer)
    assert old_tokenizer.is_fast, "This script only works with fast tokenizers"
    tokenizer = old_tokenizer.train_new_from_iterator(
        batch_iterator(),
        vocab_size=args.vocab_size,
        show_progress=True,
    )
    logger.info(f"Saving tokenizer to {args.output_dir}")
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
