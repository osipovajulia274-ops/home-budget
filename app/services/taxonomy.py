"""The category tree from data/taxonomy.yaml (13 categories, 48 subcategories)."""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List

import yaml

from app.config import DATA_DIR

UNKNOWN = "other.unknown"
TRANSFERS = "other.transfers"


@lru_cache
def _raw() -> dict:
    with open(DATA_DIR / "taxonomy.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache
def category_titles() -> Dict[str, str]:
    return {c["id"]: c["title"] for c in _raw()["categories"]}


@lru_cache
def subcategory_titles() -> Dict[str, str]:
    return {s["id"]: s["title"] for c in _raw()["categories"] for s in c["subcategories"]}


def all_titles() -> Dict[str, str]:
    """Category and subcategory titles in one mapping (for goal texts)."""
    return {**category_titles(), **subcategory_titles()}


def is_valid_sub(sub: str) -> bool:
    return sub in subcategory_titles()


def category_of(sub: str) -> str:
    return sub.split(".", 1)[0]


def is_spending(sub: str) -> bool:
    """Transfers between own accounts are not spending."""
    return sub != TRANSFERS


def tree() -> List[dict]:
    return [{"id": c["id"], "title": c["title"],
             "subcategories": [{"id": s["id"], "title": s["title"]} for s in c["subcategories"]]}
            for c in _raw()["categories"]]
