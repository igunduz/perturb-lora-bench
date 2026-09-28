"""Match dataset gene symbols to scGPT's vocabulary through HGNC renames."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HGNC_URL = ("https://storage.googleapis.com/public-download-files/hgnc/archive/archive/"
            "monthly/tsv/hgnc_complete_set_2026-02-06.txt")
HGNC_PATH = Path("data/hgnc_complete_set_2026-02-06.txt")


def download_hgnc(path: Path = HGNC_PATH, url: str = HGNC_URL) -> Path:
    """Download the pinned HGNC snapshot if it is not there yet."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def _split(field) -> list[str]:
    if not isinstance(field, str) or not field:
        return []
    return [s for s in field.strip('"').split("|") if s]


class HGNC:
    """Current symbol lookup by Ensembl ID or previous symbol, and previous symbols of a current one."""

    def __init__(self, path: Path = HGNC_PATH):
        t = pd.read_csv(path, sep="\t", dtype=str, usecols=["symbol", "prev_symbol", "ensembl_gene_id"])
        self.by_ensembl = dict(zip(t.ensembl_gene_id.dropna(), t.symbol[t.ensembl_gene_id.notna()]))
        self.prev_of = {s: _split(p) for s, p in zip(t.symbol, t.prev_symbol)}
        self.current_of: dict[str, str] = {}
        for s, prevs in self.prev_of.items():
            for p in prevs:
                self.current_of.setdefault(p, s)

    def candidates(self, symbol: str, ensembl_id: str | None = None) -> list[str]:
        """Symbol itself, then its current HGNC symbol, then that symbol's previous names."""
        current = self.by_ensembl.get(ensembl_id) or self.current_of.get(symbol) or symbol
        out = [symbol, current, *self.prev_of.get(current, [])]
        return list(dict.fromkeys(out))


def vocab_names(genes, gene_ids, vocab: dict[str, int], hgnc: HGNC | None) -> np.ndarray:
    """For each dataset gene, the vocabulary symbol to use ('' if none); each vocab entry used at most once."""
    genes = [str(g) for g in genes]
    ids = [None] * len(genes) if gene_ids is None else [str(i) for i in gene_ids]
    names = np.array([g if g in vocab else "" for g in genes], dtype=object)
    if hgnc is None:
        return names
    used = {n for n in names if n}
    for i, (g, e) in enumerate(zip(genes, ids)):
        if names[i]:
            continue
        for c in hgnc.candidates(g, e):
            if c in vocab and c not in used:
                names[i] = c
                used.add(c)
                break
    return names
