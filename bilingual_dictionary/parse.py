from xml.etree import ElementTree
from tqdm.auto import tqdm
import regex as re
from pathlib import Path
import argparse
from contextlib import ExitStack


def parse_wiktionary_dump(dump_path, languages, out_directory):
    def tag(name):
        return "{http://www.mediawiki.org/xml/export-0.10/}" + name

    heading_regex = re.Regex(r"^==([^=]+?)==$", re.MULTILINE)
    subheading_regex = re.Regex(r"^===([^=]+?)===$", re.MULTILINE)
    translation_regex = re.Regex(r"(\[\[.+?\]\]\s?)+")

    with ExitStack() as stack:
        bidict_files = {
            lang: stack.enter_context(open(out_directory / f"{lang.replace('-', '_').lower()}.txt", "w"))
            for lang in languages
        }
        bar = tqdm()

        for _, node in ElementTree.iterparse(dump_path):
            if node.tag != tag("page"):
                continue

            title = node.find(tag("title")).text
            text = node.find(tag("revision")).find(tag("text")).text

            # skip meta pages, very crude but seems to work
            # and there are not many conceivable false positives
            meta_pages = ["Wiktionary:", "Template:", "Appendix:", "User:", "Help:", "MediaWiki:", "Thesaurus:"]
            if any(meta_page in title for meta_page in meta_pages):
                continue

            language_headings = list(heading_regex.finditer(text))

            for i, heading in enumerate(language_headings):
                lang = heading.group(1).strip()
                start = heading.span()[1]
                end = (
                    language_headings[i + 1].span()[0]
                    if i + 1 < len(language_headings)
                    else len(text)
                )

                if lang.lower() not in languages:
                    continue

                subheadings = list(subheading_regex.finditer(text[start:end]))

                for match in translation_regex.finditer(text[start:end]):
                    closest_subheading = ""
                    for subheading in subheadings[::-1]:
                        if subheading.span()[1] < match.span()[0]:
                            closest_subheading = subheading.group(1)
                            break

                    if closest_subheading.startswith(
                        "Etymology"
                    ) or closest_subheading.startswith("Pronunciation"):
                        continue

                    word = match.group(0).replace("[[", "").replace("]]", "").strip()

                    if ":" in word or "|" in word:
                        continue

                    bidict_files[lang.lower()].write(f"{word}\t{title}\n")
                    break

            bar.update(1)
            node.clear()


def main(args):
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(exist_ok=True, parents=True)
    languages = args.languages.split("|")
    language_set = set(languages)

    parse_wiktionary_dump(
        args.dump_path, language_set, out_dir
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_path", type=str)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--languages", type=str, help="Pipe-separated list of languages")
    arguments = parser.parse_args()
    main(arguments)
