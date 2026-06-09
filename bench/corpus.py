"""Deterministic text corpus for the benchmark.

Uses AG News (short news titles + descriptions) so every cell embeds the exact
same real text. The corpus is downloaded once and cached to data/corpus_ag_news.txt
as one text per line; generate it on the laptop and scp the cache to the Pi so both
machines read byte-identical input.
"""
from __future__ import annotations

import csv
import io
import urllib.request
from pathlib import Path

# Long-standing AG News CSV mirror: rows are [class, title, description], no header.
AG_NEWS_URL = (
    "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/"
    "master/data/ag_news_csv/train.csv"
)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE = DATA_DIR / "corpus_ag_news.txt"


def _download() -> list[str]:
    with urllib.request.urlopen(AG_NEWS_URL, timeout=120) as resp:
        raw = resp.read().decode("utf-8")
    texts: list[str] = []
    for row in csv.reader(io.StringIO(raw)):
        if len(row) < 3:
            continue
        title, desc = row[1].strip(), row[2].strip()
        text = f"{title}. {desc}"
        # collapse to a single line so the cache stays one-text-per-line
        text = " ".join(text.split())
        if text:
            texts.append(text)
    return texts


def build_cache(max_rows: int = 40_000) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    texts = _download()[:max_rows]
    CACHE.write_text("\n".join(texts) + "\n", encoding="utf-8")


def _all_lines() -> list[str]:
    if not CACHE.exists():
        build_cache()
    return CACHE.read_text(encoding="utf-8").splitlines()


def load_corpus(n: int) -> list[str]:
    """First `n` documents — these get indexed."""
    lines = _all_lines()
    if len(lines) < n:
        raise ValueError(f"corpus cache has {len(lines)} rows, need {n}; raise max_rows")
    return lines[:n]


def load_queries(n_docs: int, n_queries: int) -> list[str]:
    """`n_queries` texts disjoint from the indexed docs (the slice after them)."""
    lines = _all_lines()
    qs = lines[n_docs : n_docs + n_queries]
    if len(qs) < n_queries:
        raise ValueError(
            f"need {n_queries} query rows after {n_docs} docs; cache too small"
        )
    return qs
