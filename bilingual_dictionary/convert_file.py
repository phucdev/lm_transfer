import logging
import argparse

import regex as re
from rich.progress import track
from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.WARNING, format=FORMAT, datefmt="[%X]", handlers=[RichHandler(show_path=False)]
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def remove_enclosed_text(s):
    # Regex pattern to match text in parentheses, brackets, or braces
    # It handles nested structures as well
    pattern = r'\([^()]*\)|\[[^\[\]]*\]|\{[^{}]*\}'
    # Use re.sub() to replace these patterns with an empty string
    result = re.sub(pattern, '', s)
    # Check if there are any nested structures left
    if re.search(pattern, result):
        return remove_enclosed_text(result)  # Recurse if nested structures found
    return result


def create_bilingual_mapping(file_path, out_file, skip_multi_words=False):
    source_regex = re.Regex(r"^(?P<word>[^{]+) \{(?P<pos>[^}]+)\}(?: \((?P<gloss>.*)\))?$")
    mapping = {}
    with open(file_path, "r", encoding="utf8") as in_file, open(out_file, "w", encoding="utf8") as out_file:
        for line in track(in_file, description="Mapping..."):
            line = line.strip().replace(" :: ", "\t")
            if not line:
                continue
            # Example: "January {proper noun} (first month of the Gregorian calendar) \t tháng giêng, tháng một"
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            source, target = parts
            source = source.strip()
            match = source_regex.search(source)
            if match is None:
                logger.warning(f"Failed to match: {source}")
                continue
            word = match.group("word")
            # pos = match.group("pos")
            # gloss = match.group("gloss")
            translations = remove_enclosed_text(target).strip().split(", ")
            for translation in translations:
                if skip_multi_words and len(word.split()) > 1:
                    continue
                if word not in mapping:
                    mapping[word] = [translation]
                else:
                    if translation in mapping[word]:
                        continue
                    else:
                        mapping[word].append(translation)
                if not translation.strip() or not word.strip():
                    continue
                out_file.write(f"{word}\t{translation}\n")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Create a bilingual mapping from a raw bilingual dictionary.")
    parser.add_argument("--in_file", type=str, help="Path to the raw bilingual dictionary.")
    parser.add_argument("--out_file", type=str, help="Path to the output file.")
    parser.add_argument("--skip_multi_words", action="store_true", default=False,
                        help="Skip multi-word entries.")
    return parser.parse_args()


def main():
    args = parse_arguments()
    in_file = args.in_file
    out_file = args.out_file
    skip_multi_words = args.skip_multi_words
    create_bilingual_mapping(in_file, out_file, skip_multi_words)


if __name__ == "__main__":
    main()
