import argparse
import regex as re
from lxml import etree as ElementTree
from tqdm.auto import tqdm
from pathlib import Path
from contextlib import ExitStack


def parse_wiktionary_dump(dump_path, languages, out_directory):
    def tag(name):
        return "{http://www.mediawiki.org/xml/export-0.10/}" + name

    heading_regex = re.Regex(r"^==([^=]+?)==$", re.MULTILINE)   # ==English==
    subheading_regex = re.Regex(r"^===([^=]+?)===$", re.MULTILINE)  # ===Noun===
    sub_subheading_regex = re.Regex(r"^====([^=]+?)====$", re.MULTILINE)    # ====Translations====
    # Translations example: color (Noun)
    # {{trans-top|hue as opposed to achromatic colors}}
    # * German: (die) {{tt+|de|Sonne|f}}
    # * Vietnamese: {{tt+|vi|Mặt Trời}}
    # {{trans-bottom}}
    #
    # Other top markers:
    # {{trans-see|associative array}}
    # {{trans-top-also|id=Q525|the star around which the Earth revolves|Sun}}
    #
    # The following regex ignores checktrans-top markers, matches trans-top, trans-top-also, trans-see markers,
    # ignores sense ids, and extracts the first gloss <definition> and translations <translations>.
    # https://en.wiktionary.org/wiki/Wiktionary:Translations
    # https://en.wiktionary.org/wiki/Template:trans-top
    definition_regex = re.Regex(
        r"(?<!{{checktrans-top}}\n){{trans-(?:top|top-also)\|?(?:id=[^|]*\|)?(?:[^|]*\|)?(?P<definition>[^}|]*)}}\s*(?P<translations>.*?)(?={{trans-bottom|\Z)",
        re.DOTALL
    )
    # TODO: we ignore trans-see for now, handling trans-see would require additional processing steps
    #  involving a lookup of the associated word and its translations
    translation_regex = re.Regex(r"\*\s*(?P<language>[^:]+):\s*(?P<translation>.+)", re.MULTILINE)
    # https://en.wiktionary.org/wiki/Template:tt
    translation_entry_regex = re.Regex(r"{{t{1,2}\+?\|(?P<lang_code>[^|]+?)\|(?P<translation>[^}|]+)")

    meta_pages = ["Wiktionary:", "Template:", "Appendix:", "User:", "Help:", "MediaWiki:", "Thesaurus:",
                  "Category:", "Thread:", "User talk:"]
    ignore_subheadings = ["Etymology", "Pronunciation", "Alternative forms", "See also", "Further reading",
                          "References", "Anagrams"]

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

            if any(meta_page in title for meta_page in meta_pages):
                continue
            if "/translations" in title:
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
                # Within each part of speech subsection, there may be a translations sub-subsection
                for i, subheading in enumerate(subheadings):
                    if subheading.group(1) in ignore_subheadings:
                        # Skip subheadings that we don't care about, e.g. "Etymology", "Pronunciation", etc.
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
                        # Find the translations sub-subsection
                        if "translations" == sub_subheading.group(1).strip().lower():
                            trans_start = sub_subheading.span()[1]
                            trans_end = (
                                sub_subheadings[j + 1].span()[0]
                                if j + 1 < len(sub_subheadings)
                                else len(text)
                            )
                            break
                    translation_section = pos_section[trans_start:trans_end]

                    definition_matches = list(definition_regex.finditer(translation_section))
                    for sense in definition_matches:
                        sense_dict = sense.groupdict()
                        definition = sense_dict["definition"].strip()
                        translations = sense_dict["translations"].strip()
                        translation_matches = list(translation_regex.finditer(translations))
                        for match in translation_matches:
                            # Examples:
                            # '* German: {{tt+|de|Wörterbuch|n}}, {{tt+|de|Diktionär|n|m}} {{q|archaic}}'
                            # '* Vietnamese: {{tt+|vi|từ điển}} ({{tt|vi|詞典}})'
                            match_dict = match.groupdict()
                            trans_lang = match_dict["language"].strip()
                            if trans_lang.lower() not in languages:
                                continue
                            translation_matches = list(translation_entry_regex.finditer(match_dict["translation"]))
                            translation = []
                            for translation_match in translation_matches:
                                # lang_code = translation_match.group("lang_code")
                                translation.append(
                                    translation_match.group("translation").replace("[[", "").replace("]]", "")
                                )
                            translation_entry = f"{title} {{{pos}}} ({definition})\t{', '.join(translation)}"
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
