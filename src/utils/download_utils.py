"""
Code adapted from https://github.com/konstantinjdobler/focus/blob/main/src/deepfocus/download_utils.py
From the paper:
@inproceedings{dobler-de-melo-2023-focus,
    title = "{FOCUS}: Effective Embedding Initialization for Monolingual Specialization of Multilingual Models",
    author = "Dobler, Konstantin  and
      de Melo, Gerard",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.829",
    doi = "10.18653/v1/2023.emnlp-main.829",
    pages = "13440--13454",
}
"""


# see https://stackoverflow.com/a/63831344

import functools
import shutil
import requests
import gzip

from pathlib import Path
from tqdm.auto import tqdm


def gunzip(path):
    path = Path(path)

    assert path.suffix == ".gz"

    new_path = path.with_suffix("")

    with gzip.open(path, "rb") as f_in:
        with open(new_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    path.unlink()
    return new_path


def download(url, path, verbose=False):
    r = requests.get(url, stream=True, allow_redirects=True)
    if r.status_code != 200:
        r.raise_for_status()
        raise RuntimeError(f"Request to {url} returned status code {r.status_code}")
    file_size = int(r.headers.get("Content-Length", 0))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    r.raw.read = functools.partial(
        r.raw.read, decode_content=True
    )  # Decompress if needed

    with tqdm.wrapattr(
        r.raw,
        "read",
        total=file_size,
        disable=not verbose,
        desc=f"Downloading {url}",
    ) as r_raw:
        with path.open("wb") as f:
            shutil.copyfileobj(r_raw, f)

    return path
