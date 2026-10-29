"""Tiny helpers: write parquet when pyarrow exists, otherwise CSV next to it.

The training environment (Colab) has pyarrow; some sandboxes do not. Every script reads/writes
tables through these helpers so a missing parquet engine degrades to CSV instead of crashing.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _has_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        try:
            import fastparquet  # noqa: F401
            return True
        except ImportError:
            return False


def write_table(df: pd.DataFrame, path: str | Path) -> Path:
    """Write `path` (.parquet preferred). Returns the path actually written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet" and not _has_parquet():
        path = path.with_suffix(".csv")
        print(f"[tableio] pyarrow not installed -> writing CSV instead: {path}")
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False, encoding="utf-8")
    return path


def read_table(path: str | Path) -> pd.DataFrame:
    """Read .parquet or .csv; if the .parquet is missing but a .csv twin exists, use it."""
    path = Path(path)
    if not path.exists() and path.suffix == ".parquet" and path.with_suffix(".csv").exists():
        path = path.with_suffix(".csv")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, keep_default_na=False, encoding="utf-8")
