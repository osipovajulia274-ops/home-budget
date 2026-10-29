"""Expand data/seed_dictionary.yaml into clean (name, subcategory, profile, group) rows."""
from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[1]
SEED_PATH = ROOT / "data" / "seed_dictionary.yaml"
TAXONOMY_PATH = ROOT / "data" / "taxonomy.yaml"

_PLACEHOLDER = re.compile(r"\{(\w+)\}")


def load_taxonomy(path: Path = TAXONOMY_PATH) -> Dict[str, dict]:
    """Return {subcategory_id: {"category": ..., "title": ...}} in file order."""
    with open(path, encoding="utf-8") as f:
        tax = yaml.safe_load(f)
    out: Dict[str, dict] = {}
    for cat in tax["categories"]:
        for sub in cat["subcategories"]:
            out[sub["id"]] = {"category": cat["id"], "title": sub["title"], "category_title": cat["title"]}
    return out


def load_seed(path: Path = SEED_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_items(raw: str) -> List[str]:
    return [x.strip() for x in raw.split("|") if x.strip()]


def template_key(template: str) -> str:
    """Group id for leakage-free splits: all fills of one template stay together."""
    return re.sub(r"\s+", " ", _PLACEHOLDER.sub("", template)).strip().lower()


def expand_template(template: str, fills: Dict[str, List[str]], rng: random.Random, n_fills: int) -> List[str]:
    names = _PLACEHOLDER.findall(template)
    if not names:
        return [template]
    out: List[str] = []
    for _ in range(n_fills * 3):  # a few retries to get distinct fills
        text = template
        for name in names:
            text = text.replace("{" + name + "}", rng.choice(fills[name]), 1)
        if text not in out:
            out.append(text)
        if len(out) >= n_fills:
            break
    return out


def generate_seed_clean(rng: random.Random, n_fills: int = 2) -> List[Tuple[str, str, str, str]]:
    """Return rows (text_clean, subcategory, profile, group)."""
    seed = load_seed()
    fills = seed["fills"]
    rows: List[Tuple[str, str, str, str]] = []
    for sub, spec in seed["subcategories"].items():
        for template in split_items(spec["items"]):
            group = f"seed:{sub}:{template_key(template)}"
            for text in expand_template(template, fills, rng, n_fills):
                rows.append((text, sub, spec.get("profile", "goods"), group))
    return rows
