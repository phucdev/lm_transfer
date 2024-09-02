import argparse
import logging
import regex as re
from lxml import etree as ElementTree
from pathlib import Path
from contextlib import ExitStack
from rich.logging import RichHandler
from rich.progress import Progress, track, TimeElapsedColumn, TextColumn


FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.WARNING, format=FORMAT, datefmt="[%X]", handlers=[RichHandler(show_path=False)]
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_wiktionary_dump(dump_path, languages, out_directory):
    def tag(name):
        return "{http://www.mediawiki.org/xml/export-0.10/}" + name

    # ==English==
    heading_regex = re.Regex(r"^==([^=]+?)==$", re.MULTILINE)
    # ===Noun===, ===Etymology 1===
    subheading_regex = re.Regex(r"^===([^=]+?)===$", re.MULTILINE)
    # ====Article===, ====Translations====
    sub_subheading_regex = re.Regex(r"^====([^=]+?)====$", re.MULTILINE)
    # =====Translations=====
    sub_sub_subheading_regex = re.Regex(r"^=====([^=]+?)=====$", re.MULTILINE)
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
    # The following regex ignores checktrans-top markers, matches trans-top, trans-top-also markers,
    # ignores sense ids, and extracts the first gloss <gloss> and translations <translations>.
    # https://en.wiktionary.org/wiki/Wiktionary:Translations
    # https://en.wiktionary.org/wiki/Template:trans-top
    gloss_regex = re.Regex(
        r"(?<!{{checktrans-top}}\n){{trans-(?:top|top-also)\|?(?:id=[^|]*\|)?(?:[^|]*\|)?(?P<gloss>[^}|]*)}}\s*("
        r"?P<translations>.*?)(?={{trans-bottom|\Z)",
        re.DOTALL
    )
    # TODO: we ignore trans-see for now, handling trans-see would require additional processing steps
    #  involving a lookup of the associated word and its translations and adjusting the regex to recognize trans-see
    translation_regex = re.Regex(r"\*\s*(?P<language>[^:]+):\s*(?P<translation>.+)", re.MULTILINE)
    # https://en.wiktionary.org/wiki/Template:tt
    # Example: {{tt+|vi|tự điển}} ({{tt+|vi|字典}}) {{q|dated, character dictionary}}
    translation_entry_regex = re.Regex(
        r"{{t{1,2}\+?\|(?P<lang_code>[^|]+?)\|(?P<translation>[^}|]+?)(?:\|[^}]+?)?}}"  # Capture translation
        r"(?: \(({{[^}]+}})\))?[^,]*?"  # Ignore group
        r"(?: {{q\|(?P<qualifier>[^}]+?)}})?"  # Capture optional qualifier
    )
    meta_pages = ["Wiktionary:", "Template:", "Appendix:", "User:", "Help:", "MediaWiki:", "Thesaurus:",
                  "Category:", "Thread:", "User talk:"]
    ignore_subheadings = ["Etymology", "Pronunciation", "Alternative forms", "See also", "Further reading",
                          "References", "Anagrams"]

    with ExitStack() as stack:
        bidict_files = {
            lang: stack.enter_context(
                open(out_directory / f"{lang.replace('-', '_').lower()}.txt", "w", encoding="utf8")
            )
            for lang in languages
        }
        with Progress(
                *Progress.get_default_columns(),
                TextColumn("Completed: [bold green]{task.completed} pages"),
                TimeElapsedColumn()
        ) as progress:
            task = progress.add_task("Parsing...", total=None)

            for _, node in ElementTree.iterparse(dump_path):
                if node.tag != tag("page"):
                    continue

                title = node.find(tag("title")).text
                if title is None:
                    continue
                # skip meta pages
                if any(meta_page in title for meta_page in meta_pages):
                    node.clear()
                    continue
                if "/translations" in title:
                    title = title.replace("/translations", "")
                text = node.find(tag("revision")).find(tag("text")).text

                if text is None:
                    node.clear()
                    continue
                language_headings = list(heading_regex.finditer(text))

                # Find English section
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
                    # No english section, skip
                    node.clear()
                    continue
                else:
                    english_section = text[start:end]
                    subheadings = list(subheading_regex.finditer(english_section))
                    # Within each part of speech subsection, there may be a translations sub-subsection
                    for i, subheading in enumerate(subheadings):
                        if subheading.group(1) in ignore_subheadings:
                            # Skip subheadings that we don't care about, e.g. "Etymology", "Pronunciation", etc.
                            continue

                        subsection_start = subheading.span()[1]
                        subsection_end = (
                            subheadings[i + 1].span()[0]
                            if i + 1 < len(subheadings)
                            else len(english_section)
                        )
                        subsection = english_section[subsection_start:subsection_end]
                        # This subsection is either a part of speech or an etymology subsection that contains
                        # part of speech sub-subsections
                        pos_sections = []   # (pos, pos_section, level)
                        if "Etymology" in subheading.group(1):
                            sub_subheadings = list(sub_subheading_regex.finditer(subsection))
                            for j, sub_subheading in enumerate(sub_subheadings):
                                if sub_subheading.group(1) in ignore_subheadings:
                                    continue
                                pos_start = sub_subheading.span()[1]
                                pos_end = (
                                    sub_subheadings[j + 1].span()[0]
                                    if j + 1 < len(sub_subheadings)
                                    else len(subsection)
                                )
                                pos = sub_subheading.group(1).strip().lower()
                                pos_section = subsection[pos_start:pos_end]
                                if sub_subheading_regex.match(pos_section):
                                    logger.debug(
                                        f"Check extracted PoS section. This should not happen if the regex is correct: "
                                        f"{pos_section}"
                                    )
                                pos_sections.append((pos, pos_section, 5))
                        else:
                            pos = subheading.group(1).strip().lower()
                            pos_sections.append((pos, subsection, 4))

                        for pos, pos_section, level in pos_sections:
                            if level == 4:
                                # Match ====Translations====
                                sub_subheadings = list(sub_subheading_regex.finditer(pos_section))
                            else:
                                # Match =====Translations=====
                                sub_subheadings = list(sub_sub_subheading_regex.finditer(pos_section))
                            trans_start = -1
                            trans_end = -1
                            for j, sub_subheading in enumerate(sub_subheadings):
                                # Find the translations sub-subsection
                                if "translations" == sub_subheading.group(1).strip().lower():
                                    trans_start = sub_subheading.span()[1]
                                    trans_end = (
                                        sub_subheadings[j + 1].span()[0]
                                        if j + 1 < len(sub_subheadings)
                                        else len(pos_section)
                                    )
                                    break
                            translation_section = pos_section[trans_start:trans_end]
                            gloss_matches = list(gloss_regex.finditer(translation_section))
                            for sense in gloss_matches:
                                sense_dict = sense.groupdict()
                                gloss = sense_dict["gloss"].strip()
                                translations = sense_dict["translations"].strip()
                                translation_matches = list(translation_regex.finditer(translations))
                                for translation_match in translation_matches:
                                    # Examples:
                                    # '* German: {{tt+|de|Wörterbuch|n}}, {{tt+|de|Diktionär|n|m}} {{q|archaic}}'
                                    # '* Vietnamese: {{tt+|vi|từ điển}} ({{tt|vi|詞典}})'
                                    translation_match_dict = translation_match.groupdict()
                                    trans_lang = translation_match_dict["language"].strip()
                                    if trans_lang.lower() not in languages:
                                        continue
                                    translation_entry_matches = list(
                                        translation_entry_regex.finditer(translation_match_dict["translation"])
                                    )
                                    translation_options = []
                                    for translation_entry_match in translation_entry_matches:
                                        # lang_code = translation_match.group("lang_code")
                                        qualifier = translation_entry_match.group("qualifier")
                                        if qualifier and ("dated" in qualifier or "archaic" in qualifier):
                                            continue
                                        # Consider this:
                                        # banana	(trái) chuối, (quả) chuối
                                        # banana	(cây) chuối
                                        # Either remove the parenthesis or remove the parenthesis and the word in it
                                        # or even add all combinations, i.e. chuối, trái chuối, quả chuối, cây chuối
                                        # Also check this: rain cats and dogs ('to rain very heavily')
                                        # * Vietnamese: {{t|vi|([[trời]]) [[mưa]] [[xối xả]]}}
                                        raw = translation_entry_match.group(
                                            "translation"
                                        ).replace("[[", "").replace("]]", "")
                                        pattern = re.Regex(r"\((?P<optional>[^\)]+)\)\s*(?P<main>\w+)")
                                        match = pattern.search(raw)
                                        if match:
                                            optional_word = match.group("optional")
                                            main_translation = match.group("main")
                                            translation_options.append(main_translation)
                                            if optional_word:
                                                combined_word = f"{optional_word} {main_translation}"
                                                if combined_word not in translation_options:
                                                    translation_options.append(combined_word)
                                        elif raw not in translation_options:
                                            translation_options.append(raw)
                                    if len(translation_options) == 0:
                                        logger.debug(f"No valid translation")
                                    else:
                                        translation_entry = (f"{title} {{{pos}}} ({gloss})\t"
                                                             f"{', '.join(translation_options)}")
                                        bidict_files[trans_lang.lower()].write(translation_entry + "\n")
                progress.update(task, advance=1)
                node.clear()


def sort_lines(file_path):
    # Works because the files are relatively small
    # For huge files: https://medium.com/@hemansnation/python-secret-to-effortlessly-sorting-large-datasets-6e298becab31
    # The article describes a method to break the file into chunks, sort those and then merge the sorted files
    with open(file_path, "r", encoding="utf8") as file:
        lines = file.readlines()
    lines_sorted = sorted(lines, key=lambda line: line.lower())
    with open(file_path, "w", encoding="utf8") as file:
        file.writelines(lines_sorted)


def create_bilingual_mapping(file_path, out_file):
    source_regex = re.Regex(r"^(?P<word>[^{]+) \{(?P<pos>[^}]+)\}(?: \((?P<gloss>.*)\))?$")
    mapping = {}
    with open(file_path, "r", encoding="utf8") as in_file, open(out_file, "w", encoding="utf8") as out_file:
        for line in track(in_file, description="Mapping..."):
            line = line.strip()
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
            translations = target.strip().split(", ")
            for translation in translations:
                if word not in mapping:
                    mapping[word] = [translation]
                else:
                    if translation in mapping[word]:
                        continue
                    else:
                        mapping[word].append(translation)
                out_file.write(f"{word}\t{translation}\n")


def main(args):
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(exist_ok=True, parents=True)
    languages = args.languages.split("|")
    logger.info(f"Wiktionary dump: {args.dump_path}, output directory: {out_dir}, languages: {languages}")
    language_set = []
    lang_files = []
    parse_languages = []
    for lang in languages:
        lang_file = out_dir / f"{lang.replace('-', '_').lower()}.txt"
        language_set.append(lang)
        lang_files.append(lang_file)
        if args.override or not lang_file.exists():
            parse_languages.append(lang)
    if not parse_languages:
        logger.info("All files already exist. Skip parsing...")
    else:
        logger.info(f"Parsing dump at {args.dump_path} for {parse_languages}")
        parse_wiktionary_dump(
            args.dump_path, parse_languages, out_dir
        )

    for lang, lang_file in zip(language_set, lang_files):
        if args.sort:
            sort_lines(lang_file)
            logger.info(f"Sorted {lang_file}")
        if args.create_mapping:
            out_file = out_dir / f"english-{lang.lower()}.txt"
            if args.override or not out_file.exists():
                logger.info(f"Creating bilingual mapping for {lang}")
                create_bilingual_mapping(
                    lang_file, out_dir / f"english-{lang.lower()}.txt"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--languages", type=str, help="Pipe-separated list of languages", required=True)
    parser.add_argument("--override", action="store_true", default=False, help="Override existing files")
    parser.add_argument("--sort", action="store_true", default=False, help="Sort the output files")
    parser.add_argument(
        "--create_mapping", action="store_true", default=False, help="Create bilingual mapping"
    )
    arguments = parser.parse_args()
    main(arguments)
