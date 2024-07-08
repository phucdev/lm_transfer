import datasets
import logging
import argparse
import requests
import time

from transformers import AutoTokenizer
from rich.logging import RichHandler

logger = logging.getLogger(__name__)
logger.addHandler(RichHandler())
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a tokenizer")
    parser.add_argument(
        "--use_combined_dataset",
        action="store_true",
        default=False,
        help="Whether to use a combined dataset of BKAI news corpus, ViSoBERT social media text corpus, and Vietnamese"
             "portion of OSCAR 2301, mC4 and Wikipedia",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Name of the dataset to train the tokenizer on",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        help="Name of the dataset config to use",
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


def main():
    args = parse_args()
    if args.use_combined_dataset:
        logger.info("Using a combined dataset of BKAI news corpus, ViSoBERT social media text corpus, and Vietnamese"
                    "portion of CulturaX and Wikipedia")
        bkai_news = datasets.load_dataset("bkai-foundation-models/BKAINewsCorpus", split="train", streaming=True)
        visobert = datasets.load_dataset("phucdev/ViSoBERT", split="train", streaming=True)
        culturax = datasets.load_dataset("uonlp/CulturaX", "vi", split="train", streaming=True)
        wikipedia = datasets.load_dataset("wikimedia/wikipedia", "20231101.vi", split="train", streaming=True)
        data = [bkai_news, visobert, culturax, wikipedia]
    else:
        logger.info(f"Training tokenizer on {args.dataset_name} with vocab size {args.vocab_size}")
        data = [datasets.load_dataset(
            args.dataset_name, args.dataset_config_name, split="train", streaming=True
        )]

    def batch_iterator(batch_size=1000):
        batch = []
        for dataset in data:
            idx = 0
            try:
                for idx, example in enumerate(dataset):
                    batch.append(example["text"])
                    if len(batch) == batch_size:
                        yield batch
                        batch = []
            except requests.exceptions.ConnectionError:
                time.sleep(15)
                for c_idx, example in enumerate(dataset):
                    if c_idx < idx:
                        continue
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
