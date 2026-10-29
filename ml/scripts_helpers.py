"""Shared helpers for reading the CDEK receipt dataset (D6).

D6 columns used: item_texts, seller_text (and prices). The exact storage of `item_texts`
(list / numpy array / JSON string / newline-joined text) is handled defensively because the
schema was not verified on the live files - run 01_download.py and read raw_schemas.json.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple


def as_list(value) -> List[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):  # numpy array
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    s = str(value).strip()
    if s.startswith("["):
        try:
            return [str(x) for x in json.loads(s)]
        except ValueError:
            pass
    return [part for part in re.split(r"\r?\n", s) if part.strip()]


def iter_d6_items(parquet_files: Iterable[Path]) -> Iterator[Tuple[str, str]]:
    """Yield (receipt_id, item_text) for every item of every receipt."""
    import pandas as pd

    for f in parquet_files:
        split = f.stem.split("-")[0]
        df = pd.read_parquet(f, columns=None)
        if "item_texts" not in df.columns:
            raise SystemExit(f"{f.name}: no 'item_texts' column; columns = {list(df.columns)}")
        for i, row in enumerate(df.itertuples(index=False)):
            for text in as_list(getattr(row, "item_texts")):
                yield f"{split}_{i}", text
