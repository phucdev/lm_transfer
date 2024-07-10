import transformers
import csv
import re

from focus import get_overlapping_tokens


def main():
    # source_model = transformers.AutoModel.from_pretrained("FacebookAI/xlm-roberta-base")
    source_tokenizer = transformers.AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
    normalized_source_tokens = [re.sub('^(##|Ġ|▁)', '', t.lower()) for t in source_tokenizer.get_vocab().keys()]

    # target_model = transformers.AutoModel.from_pretrained("FPTAI/vibert-base-cased")
    target_tokenizer = transformers.AutoTokenizer.from_pretrained("FPTAI/vibert-base-cased")

    overlapping_tokens, missing_tokens = get_overlapping_tokens(target_tokenizer, source_tokenizer,
                                                                match_symbols=False,
                                                                exact_match_all=False,
                                                                fuzzy_match_all=True)
    print(f"Overlapping tokens: {len(overlapping_tokens)}")
    print(f"Missing tokens: {len(missing_tokens)}")

    # Load bilingual dictionary, create a mapping from Vietnamese to English
    dict_pairs = {}
    with open("bilingual_dictionary/Wiktionary/english-vietnamese.txt") as f:
        csv_reader = csv.reader(f, delimiter='\t')
        for row in csv_reader:
            if row[1] in dict_pairs:
                dict_pairs[row[1]].append(row[0])
            else:
                dict_pairs[row[1]] = [row[0]]

    found_missing_tokens = []
    for t, _ in missing_tokens.items():
        if t in dict_pairs:
            the_key = t
        elif re.sub('^(##|Ġ|▁)', '', t.lower()) in dict_pairs:
            the_key = re.sub('^(##|Ġ|▁)', '', t.lower())
        else:
            continue
        en_translations = dict_pairs[the_key]
        for en_t in en_translations:
            if en_t in normalized_source_tokens or any(re.sub('^(##|Ġ|▁)', '', token.lower()) in normalized_source_tokens for token in source_tokenizer.tokenize(en_t)):
                found_missing_tokens.append(t)
                break

    print(f"Found missing tokens: {len(found_missing_tokens)}")


if __name__ == '__main__':
    main()
