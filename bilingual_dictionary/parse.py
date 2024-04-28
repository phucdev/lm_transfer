from lxml import etree as ElementTree
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
    sub_subheading_regex = re.Regex(r"^====([^=]+?)====$", re.MULTILINE)
    translation_regex = re.Regex(r"\*\s*(?P<language>[^:]+):\s*(?P<translation>.+)", re.MULTILINE)
    translation_entry_regex = re.Regex(r"{{tt\+?\|[^|]+?\|([^}|]+)")

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
            # skip meta pages
            meta_pages = ["Wiktionary:", "Template:", "Appendix:", "User:", "Help:", "MediaWiki:", "Thesaurus:",
                          "Category:", "Thread:", "User talk:"]
            if any(meta_page in title for meta_page in meta_pages):
                continue
            if "/translations" in title:
                print("Found a translation subpage")
                title = title.replace("/translations", "")
            text = node.find(tag("revision")).find(tag("text")).text

            if text is None:
                continue
            language_headings = list(heading_regex.finditer(text))

            # find English section
            start = -1
            end = -1
            for i, heading in enumerate(language_headings):
                lang = heading.group(1).strip()
                if "english" == lang.lower():
                    start = heading.span()[1]
                    end = (
                        language_headings[i + 1].span()[0]
                        if i + 1 < len(language_headings)
                        else len(text)
                    )
                    break

            if start == -1 or end == -1:
                continue
            else:
                english_section = text[start:end]
                subheadings = list(subheading_regex.finditer(english_section))
                # Etymology, Pronunciation, followed by parts of speech, e.g. Noun, Verb, etc.
                # Within each part of speech subsection, there may be a translations sub-subsection
                for i, subheading in enumerate(subheadings):
                    if "Etymology" in subheading.group(1) or "Pronunciation" in subheading.group(1):
                        continue

                    pos_start = subheading.span()[1]
                    pos_end = (
                        subheadings[i + 1].span()[0]
                        if i + 1 < len(subheadings)
                        else len(text)
                    )
                    pos_section = english_section[pos_start:pos_end]
                    pos = subheading.group(1).strip().lower()
                    sub_subheadings = list(sub_subheading_regex.finditer(pos_section))
                    trans_start = -1
                    trans_end = -1
                    for j, sub_subheading in enumerate(sub_subheadings):
                        if "translations" == sub_subheading.group(1).strip().lower():
                            trans_start = sub_subheading.span()[1]
                            trans_end = (
                                sub_subheadings[j + 1].span()[0]
                                if j + 1 < len(sub_subheadings)
                                else len(text)
                            )
                            break
                    translation_section = pos_section[trans_start:trans_end]
                    definition_match = re.search(r"\{\{trans-top\|(.*?)\}\}", translation_section)
                    if definition_match:
                        definition = definition_match.group(1)
                    else:
                        definition = None
                    translation_matches = list(translation_regex.finditer(translation_section))
                    for match in translation_matches:
                        # Examples:
                        # '* German: {{tt+|de|Wörterbuch|n}}, {{tt+|de|Diktionär|n|m}} {{q|archaic}}'
                        # '* Vietnamese: {{tt+|vi|từ điển}} ({{tt|vi|詞典}})'
                        match_dict = match.groupdict()
                        trans_lang = match_dict["language"].strip()
                        if trans_lang.lower() not in languages:
                            continue
                        translations = translation_entry_regex.findall(match_dict["translation"])
                        if translations:
                            translations = [word.replace("[[", "").replace("]]", "") for word in translations]
                            translation_entry = f"{title} {{{pos}}} ({definition})\t{', '.join(translations)}"
                            bidict_files[trans_lang.lower()].write(translation_entry + "\n")
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
    parser.add_argument("--dump_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--languages", type=str, help="Pipe-separated list of languages", required=True)
    arguments = parser.parse_args()
    main(arguments)
