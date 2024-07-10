import transformers

from tqdm import tqdm


def main():
    sample = False
    if sample:
        en_file = "parallel_data/sample.en"
        vi_file = "parallel_data/sample.vi"
        out_file = "parallel_data/sample-sample-parallel-en-vi.txt"
        lines = 1000
    else:
        en_file = "parallel_data/OpenSubtitles.en-vi.en"
        vi_file = "parallel_data/OpenSubtitles.en-vi.vi"
        out_file = "parallel_data/parallel2-en-vi.txt"
        lines = 3505276
    source_tokenizer = transformers.AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
    target_tokenizer = transformers.AutoTokenizer.from_pretrained("tokenizers/vi-spm-oscar")

    with open(en_file, 'r') as en_reader, open(vi_file, 'r') as vi_reader, open(out_file, 'w') as writer:
        en_batch = []
        vi_batch = []
        for en_line, vi_line in tqdm(zip(en_reader, vi_reader), desc="Processing", total=lines):
            en_batch.append(en_line.strip())
            vi_batch.append(vi_line.strip())
            if len(en_batch) < 10000:
                continue
            en_tokenized = [" ".join(source_tokenizer.tokenize(en_line)) for en_line in en_batch]
            vi_tokenized = [" ".join(target_tokenizer.tokenize(vi_line)) for vi_line in vi_batch]
            buffer = []
            for en_tokens, vi_tokens in zip(en_tokenized, vi_tokenized):
                buffer.append(f'{en_tokens} ||| {vi_tokens}')
            writer.writelines(buffer)
            en_batch = []
            vi_batch = []
        if en_batch:
            en_tokenized = [" ".join(source_tokenizer.tokenize(en_line)) for en_line in en_batch]
            vi_tokenized = [" ".join(target_tokenizer.tokenize(vi_line)) for vi_line in vi_batch]
            buffer = []
            for en_tokens, vi_tokens in zip(en_tokenized, vi_tokenized):
                buffer.append(f'{en_tokens} ||| {vi_tokens}')
            writer.writelines(buffer)


if __name__ == '__main__':
    main()
