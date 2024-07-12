import transformers
import argparse

from tqdm import tqdm


def process_batch(en_batch, vi_batch, source_tokenizer, target_tokenizer):
    buffer = []
    en_input_ids_batch = source_tokenizer.batch_encode_plus(en_batch, add_special_tokens=False)["input_ids"]
    vi_input_ids_batch = target_tokenizer.batch_encode_plus(vi_batch, add_special_tokens=False)["input_ids"]
    for en_input_ids, vi_input_ids in zip(en_input_ids_batch, vi_input_ids_batch):
        en_tokenized = " ".join(source_tokenizer.convert_ids_to_tokens(en_input_ids))
        vi_tokenized = " ".join(target_tokenizer.convert_ids_to_tokens(vi_input_ids))
        buffer.append(f'{en_tokenized} ||| {vi_tokenized}\n')
    return buffer


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare parallel data for fast_align.")
    parser.add_argument(
        "--sample",
        action="store_true",
        default=False,
        help="Use sample data."
    )
    parser.add_argument(

        "--bilingual_dictionary",
        action="store_true",
        default=False,
        help="Use bilingual dictionary."
    )
    parser.add_argument(

        "--source_tokenizer",
        type=str,
        default="FacebookAI/xlm-roberta-base",
        help="Source tokenizer."
    )
    parser.add_argument(
        "--target_tokenizer",
        type=str,
        default="tokenizers/vi-spm-oscar",
        help="Target tokenizer."
    )
    parser.add_argument(
        "--dict_file",
        type=str,
        default="bilingual_dictionary/PaulDenisowski/english-vietnamese.txt",
        help="Bilingual dictionary file."
    )
    parser.add_argument(
        "--out_file",
        type=str,
        required=True,
        help="Output file.")
    parser.add_argument(
        "--en_file",
        type=str,
        default="parallel_data/full/OpenSubtitles.en-vi.en",
        help="English file."
    )
    parser.add_argument(
        "--vi_file",
        type=str,
        default="parallel_data/full/OpenSubtitles.en-vi.vi",
        help="Vietnamese file."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sample = args.sample
    bilingual_dictionary = args.bilingual_dictionary

    source_tokenizer = transformers.AutoTokenizer.from_pretrained(args.source_tokenizer)
    target_tokenizer = transformers.AutoTokenizer.from_pretrained(args.target_tokenizer)

    out_file = args.out_file

    if bilingual_dictionary:
        dict_file = args.dict_file
        with open(dict_file, 'r') as reader, open(out_file, 'w') as writer:
            en_batch = []
            vi_batch = []
            for line in tqdm(reader, desc="Processing"):
                split_line = line.strip().split("\t")
                if len(split_line) != 2:
                    continue
                en, vi = split_line[0], split_line[1]
                en_batch.append(en)
                vi_batch.append(vi)
                if len(en_batch) < 10000:
                    continue
                buffer = process_batch(en_batch, vi_batch, source_tokenizer, target_tokenizer)
                writer.writelines(buffer)
                en_batch = []
                vi_batch = []
            if en_batch:
                buffer = process_batch(en_batch, vi_batch, source_tokenizer, target_tokenizer)
                writer.writelines(buffer)
    else:
        en_file = args.en_file
        vi_file = args.vi_file
        if sample:
            lines = 1000
        else:
            lines = 3505276     # Number of lines in OpenSubtitles.en-vi.en, change if you use a different dataset

        with open(en_file, 'r') as en_reader, open(vi_file, 'r') as vi_reader, open(out_file, 'w') as writer:
            en_batch = []
            vi_batch = []
            for idx, (en_line, vi_line) in tqdm(enumerate(zip(en_reader, vi_reader)), desc="Processing", total=lines):
                if idx >= lines:
                    break
                en_batch.append(en_line.strip())
                vi_batch.append(vi_line.strip())
                if len(en_batch) < 10000:
                    continue
                buffer = process_batch(en_batch, vi_batch, source_tokenizer, target_tokenizer)
                writer.writelines(buffer)
                en_batch = []
                vi_batch = []
            if en_batch:
                buffer = process_batch(en_batch, vi_batch, source_tokenizer, target_tokenizer)
                writer.writelines(buffer)


if __name__ == '__main__':
    main()
