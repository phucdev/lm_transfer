import re
import logging

from rich.logging import RichHandler
from tqdm import tqdm


logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)


def remove_formatting_characters(text):
    # Regex pattern to remove various invisible/formatting characters
    # \u200B is Zero Width Space, \u200C is Zero Width Non-Joiner, etc.
    pattern = r'[\ufeff\u200B\u200C\u200D\u2060]+'
    return re.sub(pattern, '', text)


def main():
    input_file = "vnedict.txt"
    output_file = "english-vietnamese.txt"

    pattern = r'\(.*?\)'    # Remove anything enclosed by parentheses
    counter = 0
    with (open(input_file, mode="r", encoding="utf8") as in_file,
          open(output_file, mode="w", encoding="utf8") as out_file):
        for idx, line in tqdm(enumerate(in_file), desc="Processing dictionary file"):
            if line.startswith("#"):
                continue
            line = remove_formatting_characters(line).strip()
            if not line:
                continue
            splitlines = line.split(" : ")
            if len(splitlines) != 2:
                logger.warning(f"Skipping line: {line}")
                counter += 1
                continue
            vietnamese, english = splitlines
            vietnamese = vietnamese.strip()

            english = re.sub(pattern, '', english)
            english_translations = re.split('; |, ', english)
            for translation in english_translations:
                translation = translation.strip()
                out_file.write(f"{translation}\t{vietnamese}\n")
    logger.info(f"Skipped {counter} lines")


if __name__ == '__main__':
    main()
