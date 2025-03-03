import argparse
import datasets
import json
import random
import os
import logging
import matplotlib.pyplot as plt

from transformers import AutoTokenizer
from collections import Counter
from pathlib import Path
from tqdm import tqdm


METHOD_MAP = {
  "random_initialization": "RAND",
  "ramen_initialization": "RAMEN",
  "ramen_overlap_initialization": "RAMEN+Overlap",
  "wechsel_initialization": "WECHSEL",
  "wechsel_overlap_initialization": "WECHSEL+Overlap",
  "wechsel_aligned_initialization": "WECHSEL+PreAligned+Overlap",
  "wechsel_rcsls_initialization": "WECHSEL+RCSLS",
  "focus_monolingual_initialization": "FOCUS",
  "fvt_initialization": "FVT",
  "fvt_minimize_punctuation_initialization": "FVT+MinPunct",
  "fvt_subword_length_initialization": "FVT+SubwordLength",
  "fvt_rescale_initialization": "FVT+Rescale",
  "fvt_freq_weighted_minimize_punctuation_initialization": "FVT+FreqWeighted+MinPunct",
  "focus_multilingual_initialization": "FOCUS",
}

BPE_TOKEN_SAMPLE = {
    "high": ['Ġgá»Ĺ', 'ĠcÆ°á»Ŀi', 'ĠÄĳá»«ng'],  # gỗ (wood), cười (to laugh), đừng (don't)
    "medium": ['ĠÄĳiÃªn', 'ĠxÄĥng', 'Ġtre'],    # điên (crazy), xăng (gasoline), tre (bamboo)
    "low": ['Ġxá»Ľi', 'ĠgÃłi', 'Ġpháº©y']    # xới (to stir), gài (to fasten), phẩy (to flick/ comma)
}
SPM_TOKEN_SAMPLE = {
    "high": ['▁đỏ', '▁hành', '▁cũ'],    # đỏ (red), hành (onion/ to torture), cũ (old)
    "medium": ['▁bám', '▁vầy', '▁núi'],   # bám (to cling), vầy (like this/ to swirl), núi (mountain)
    "low": ['▁thòi', '▁câm', '▁nhện']   # thòi (to protrude), câm (mute), nhện (spider)
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--num_tokens", type=int, default=10)
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--source_tokenizer", type=str)
    parser.add_argument("--target_tokenizer", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--num_proc", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dictionary_path", type=str, default=None)
    parser.add_argument("--sample_new_tokens", action="store_true", default=False, help="Whether to sample new tokens instead of using the pre-sampled ones")
    parser.add_argument("--show_plot", action="store_true", default=False, help="Whether to show the token frequency histogram")
    return parser.parse_args()


def sample_from_bucket(bucket_tokens, overlap_tokens, n=3):
    """
       bucket_tokens: set of tokens that belong to this frequency bucket
       overlap_tokens: set of tokens that are in the overlap
       n: how many tokens to sample in total

       We try to ensure at least 1 token from overlap if possible.
    """
    # Separate overlap vs. non-overlap within this bucket
    overlap_in_bucket = bucket_tokens.intersection(overlap_tokens)
    non_overlap_in_bucket = bucket_tokens.difference(overlap_in_bucket)

    chosen = []

    # 1) Try to pick exactly 1 overlap token (if available)
    if len(overlap_in_bucket) > 0:
        chosen_overlap = random.sample(overlap_in_bucket, 1)
    else:
        chosen_overlap = []

    chosen.extend(chosen_overlap)

    # 2) Pick the remaining from the non-overlap
    needed = n - len(chosen)

    # If we can pick from non-overlap
    if len(non_overlap_in_bucket) >= needed:
        chosen_non_overlap = random.sample(non_overlap_in_bucket, needed)
        chosen.extend(chosen_non_overlap)
    else:
        # If there's not enough non-overlap, just fill from the entire bucket
        # This scenario might be rare, but let's handle it gracefully.
        # We'll remove the ones we've already chosen from the bucket to avoid duplicates.
        remaining_bucket = bucket_tokens.difference(chosen)

        if len(remaining_bucket) >= needed:
            chosen_rest = random.sample(remaining_bucket, needed)
            chosen.extend(chosen_rest)
        else:
            # fallback if the bucket is extremely small
            chosen.extend(list(remaining_bucket))

    return chosen


def get_vietnamese_word_list(file_path):
    with open(file_path, "r") as f:
        word_list = []
        for line in f.readlines():
            words = line.strip().split("\t")
            viet_word = words[-1].lower()
            if viet_word not in word_list:
                word_list.append(viet_word)
    return word_list


def get_token_frequency(data, tokenizer, num_proc):
    # Count frequency of tokens in the tokenizer vocabulary in the data
    target_tokenizer_vocab = tokenizer.get_vocab()

    def tokenize_fn(examples):
        return tokenizer(examples["text"])

    data = data.map(tokenize_fn, batched=True, num_proc=num_proc)
    # Count frequencies
    token_frequencies = Counter()
    for example in tqdm(data, desc="Counting token frequencies"):
        token_frequencies.update(example['input_ids'])
    id_to_token = {v: k for k, v in target_tokenizer_vocab.items()}
    token_frequencies = {id_to_token[token]: freq for token, freq in token_frequencies.items()}
    return token_frequencies


def filter_frequency(token_frequencies, target_tokenizer, dictionary_path):
    # Filter out special tokens and punctuation
    special_tokens = target_tokenizer.all_special_tokens
    punctuation = [".", ",", "!", "?", ":", ";", "(", ")", "[", "]", "{", "}"]
    filtered_token_frequencies = {token: freq for token, freq in token_frequencies.items() if
                                  token not in special_tokens and token not in punctuation}

    # Only keep Vietnamese tokens: use a list as a filter
    vietnamese_word_list = " ".join(get_vietnamese_word_list(dictionary_path))
    filtered = {}
    for token, freq in filtered_token_frequencies.items():
        decoded_token = target_tokenizer.decode(target_tokenizer.convert_tokens_to_ids(token)).lower()
        if decoded_token in vietnamese_word_list:
            filtered[token] = freq
    return filtered


def get_token_sample(token_frequencies, sample_size):
    # Sort the token frequencies highest to lowest
    sorted_token_freqs = sorted(token_frequencies.items(), key=lambda x: x[1], reverse=True)
    # partition the tokenizer vocabulary into three buckets: high-freq (top 10%), mid-freq (next 40%),
    # low-freq (remaining 50%)
    high_freq = sorted_token_freqs[:int(len(sorted_token_freqs) * 0.1)]
    mid_freq = sorted_token_freqs[int(len(sorted_token_freqs) * 0.1):int(len(sorted_token_freqs) * 0.5)]
    low_freq = sorted_token_freqs[int(len(sorted_token_freqs) * 0.5):]

    # Randomly sample 3 tokens from each bucket
    high_sample = [tok[0] for tok in random.sample(high_freq, sample_size)]
    mid_sample = [tok[0] for tok in random.sample(mid_freq, sample_size)]
    low_sample = [tok[0] for tok in random.sample(low_freq, sample_size)]
    return {
        "high": high_sample,
        "mid": mid_sample,
        "low": low_sample
    }


def main():
    args = parse_args()

    random.seed(args.seed)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # Load the dataset, count frequency of tokens in the tokenizer vocabulary
    data = datasets.load_dataset("json", data_files={"train": args.data_path}, split="train")
    target_tokenizer = AutoTokenizer.from_pretrained(args.target_tokenizer)
    source_tokenizer = AutoTokenizer.from_pretrained(args.source_tokenizer)

    # Calculate the overlap between the source and target tokenizers
    # target_tokens = set(target_tokenizer.get_vocab().keys())
    # source_tokens = set(source_tokenizer.get_vocab().keys())
    # overlapping_tokens = target_tokens & source_tokens
    # missing_tokens = target_tokens - source_tokens

    if Path(args.output_dir).joinpath("token_frequencies.json").exists():
        logger.info(f"Loading token frequencies from: {Path(args.output_dir).joinpath('token_frequencies.json')}")
        with open(Path(args.output_dir).joinpath("token_frequencies.json"), "r") as f:
            token_frequencies = json.load(f)
    else:
        logger.info(f"Calculating token frequencies on {args.data_path=} using {args.target_tokenizer=}")
        token_frequencies = get_token_frequency(data, target_tokenizer, args.num_proc)
        with open(Path(args.output_dir).joinpath("token_frequencies.json"), "w") as f:
            f.write(json.dumps(token_frequencies))

    if args.show_plot:
        # Plot the token frequencies with a histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.hist(list(token_frequencies.values()), bins=100)
        ax.set_title("Token Frequencies")
        plt.tight_layout()
        plt.show()

    filtered = filter_frequency(token_frequencies, target_tokenizer, args.dictionary_path)

    logger.info("Sampling tokens for each frequency bucket")
    if args.sample_new_tokens:
        token_sample = get_token_sample(filtered, args.num_tokens)
    elif args.target_tokenizer == "phucdev/vi-bpe-culturax-4g-sample":
        token_sample = BPE_TOKEN_SAMPLE
    elif args.target_tokenizer == "phucdev/vi-spm-culturax-4g-sample":
        token_sample = SPM_TOKEN_SAMPLE
    else:
        token_sample = get_token_sample(filtered, args.num_tokens)

    # Then go through each model directory containing a sources.json file
    # Create a table that shows the token, its frequency bucket, the method of its embedding initialization,
    # and the top source tokens and their weights
    table = []
    for bucket, tokens in token_sample.items():
        for token in tokens:
            table.append({
                "token": token,
                "decoded_token": target_tokenizer.decode(target_tokenizer.convert_tokens_to_ids(token)),
                "frequency_bucket": bucket,
                "sources": {}
            })

    for model_dir in tqdm(os.listdir(args.input_dir), desc="Processing model directories"):
        model_path = os.path.join(args.input_dir, model_dir)
        if not os.path.isdir(model_path):
            continue
        if not os.path.exists(os.path.join(model_path, "sources.json")):
            continue
        with open(os.path.join(model_path, "sources.json"), "r") as f:
            sources = json.load(f)
        for entry in table:
            token = entry["token"]
            method = METHOD_MAP.get(model_dir, model_dir)
            if token in sources:
                source_token_info = sources[token]
                src_tokens = source_token_info[0]
                src_token_ids = source_token_info[1]
                src_token_weights = source_token_info[2]
                token_sources = []
                # Sort the source tokens by their weight and keep only the top 5
                src_token_weights, src_tokens, src_token_ids = zip(*sorted(zip(src_token_weights, src_tokens, src_token_ids), reverse=True))
                idx = 0
                for src_token_id, weight in zip(src_token_ids, src_token_weights):
                    decoded_src_token = source_tokenizer.decode(src_token_id)
                    token_sources.append((decoded_src_token, weight))
                    idx += 1
                    if idx >= 5:
                        break
                entry["sources"][method] = {
                    "top_source_tokens": token_sources,
                    "num_source_tokens": len(src_tokens)
                }
            else:
                entry["sources"][method] = {
                    "top_source_tokens": [],
                    "num_source_tokens": 0
                }

    # Order sources methods by the order in the METHOD_MAP
    for entry in table:
        entry_sources = [(k, v) for k, v in entry["sources"].items()]
        entry_sources.sort(key=lambda x: list(METHOD_MAP.values()).index(x[0]))
        entry["sources"] = {k: v for k, v in entry_sources}

    with open(os.path.join(args.output_dir, "sample_tokens.json"), "w") as f:
        f.write(json.dumps(table, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
