## Bilingual dictionary for English-Vietnamese

### Existing bilingual dictionaries

- [MUSE by Facebook Research](https://github.com/facebookresearch/MUSE) (created using an internal translation tool): 
  - Bilingual dictionary: https://dl.fbaipublicfiles.com/arrival/dictionaries/en-vi.txt
  - Word embeddings: https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.vi.vec
- Matthias Buchmeier's bilingual dictionaries (created from wiktionary dump): https://en.wiktionary.org/wiki/User:Matthias_Buchmeier
- [Open DSL Dictionary Project](https://github.com/open-dsl-dict/wiktionary-dict) (based on Matthias Buchmeier): https://github.com/open-dsl-dict/wiktionary-dict/blob/master/dsl/oneway/en-vi-enwiktionary.dsl.dz
- [WECHSEL](https://github.com/CPJKU/wechsel) (created from wiktionary dump): https://github.com/CPJKU/wechsel/tree/main/dicts/data/vietnamese.txt
  - The quality of the dictionary is not good. There is a lot of the old Vietnamese script (Chữ Nôm) and is not relevant to our use case.

### Creating a new bilingual dictionary
There are multiple options to create a more up-to-date bilingual dictionary.

#### Use parse script in this repository
We can use the `parse.py` script in this repository to create a bilingual dictionary from the translations sections of
the [wiktionary database dump](https://dumps.wikimedia.org/backup-index.html).
This script is a rough python implementation of the AWK parsing script from Matthias Buchmeier.
```bash
python parse.py \
  --out_dir bilingual_dictionary/Wiktionary \
  --dump_path bilingual_dictionary/enwiktionary-20240420-pages-articles.xml \
  --languages "german|vietnamese" \
  --sort \
  --create_mapping
```

This will create: 
- a bilingual dictionary for each language, which looks like:
  ```
  accurate {adjective} (exact or careful conformity to truth)	genau, präzise, exakt
  ```
- a bilingual mapping for each language with no extra information, which looks like:
  ```
  accurate genau
  ```
  
Issues:
- The script does not handle `{{trans-see}}` translation markers that point to another page. This would require a 
  processing all pages once and creating a lookup table in order to resolve these references.

#### Using AWK scripts from Matthias Buchmeier
We can use scripts from Matthias Buchmeier to create a bilingual dictionary from the translations sections of 
the [wiktionary database dump](https://dumps.wikimedia.org/backup-index.html).
You can find the scripts here:
- https://en.wiktionary.org/wiki/User:Matthias_Buchmeier/trans-en-es.awk
- https://en.wiktionary.org/wiki/User:Matthias_Buchmeier/ding2dictd

Steps:
1. Change into this directory
    ```bash
    cd bilingual_dictionary
    ```
2. Download the dump for English Wiktionary and put it in this directory: https://dumps.wikimedia.org/enwiktionary/20240420/enwiktionary-20240420-pages-articles.xml.bz2 
3. We use the script from Matthias Buchmeier to create a bilingual dictionary for English-Vietnamese (https://en.wiktionary.org/wiki/User:Matthias_Buchmeier/trans-en-es.awk)
    ```bash
    bzcat enwiktionary-20240420-pages-articles.xml.bz2|gawk -v LANG=Vietnamese -v ISO=vi -v REMOVE_WIKILINKS="y" -f trans-en-es.awk|sort -s -d -k 1,1 -t"{">output.txt
    ```
4. In order to convert dictionaries from ding-format to dictd-format, we can use https://en.wiktionary.org/wiki/User:Matthias_Buchmeier/ding2dictd
    ```bash
    bash ./ding2dictd.sh output.txt en-vi
    ```
5. Check the resulting `en-vi.dict` file. This can serve as the base for our bilingual dictionary.

#### Using the parse script from WECHSEL
It is also possible to use the `parse.py` script from https://github.com/CPJKU/wechsel/blob/main/dicts/parse.py to 
generate a bilingual dictionary from the translations sections of the wiktionary database dump.

However, the script does not work properly and the generated dictionary is low quality.
For Vietnamese, the script uses the old Vietnamese script (Chữ Nôm), which is not used anymore and is not
relevant to our use case. We are only interested in modern Vietnamese.
You would have to adjust the following things:
- Check that there is an English section in the entry (there are non-English entries in the dump)
- Actually try to parse the Translations sections. The script currently tries to find a translation within any section,
  using the following regex (excluding cases where the closest subheading is "Pronunciation" or "Etymology"):
  ```python
  translation_regex = re.Regex(r"(\[\[.+?\]\]\s?)+")
  ```
