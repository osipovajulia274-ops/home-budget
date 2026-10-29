import json
import random
import re
from pathlib import Path

import pandas as pd
import yaml

from app.services.normalize import normalize
from ml.dataset_builder import build_dataset, check_no_overlap, resolve_label_conflicts
from ml.receipt_noise import noisy_variant, noisy_variants
from ml.seed_generator import (
    SEED_PATH, expand_template, generate_seed_clean, load_seed, load_taxonomy, split_items, template_key,
)

ROOT = Path(__file__).resolve().parents[1]
TAX = load_taxonomy()


def test_taxonomy_shape():
    assert len(TAX) == 48
    assert len({v["category"] for v in TAX.values()}) == 13
    assert all(sid.split(".")[0] == v["category"] for sid, v in TAX.items())


def test_yaml_off_key_not_boolean():
    # YAML 1.1 reads bare `off:` as False - the dataset uses the string "off" as a source name.
    cfg = yaml.safe_load(open(ROOT / "data" / "off_category_map.yaml", encoding="utf-8"))
    assert "food" in cfg and False not in cfg


def test_seed_covers_all_subcategories_with_enough_templates():
    seed = load_seed()
    assert set(seed["subcategories"]) == set(TAX)
    for sub, spec in seed["subcategories"].items():
        assert len(split_items(spec["items"])) >= 30, sub
        assert spec["profile"] in ("goods", "meal", "service")


def test_all_placeholders_have_fills():
    seed = load_seed()
    used = set()
    for spec in seed["subcategories"].values():
        for t in split_items(spec["items"]):
            used |= set(re.findall(r"\{(\w+)\}", t))
    assert used <= set(seed["fills"]), used - set(seed["fills"])


def test_expand_template_and_group_key():
    rng = random.Random(1)
    fills = load_seed()["fills"]
    out = expand_template("Молоко {fat} {ml}", fills, rng, 3)
    assert 1 <= len(out) <= 3 and all("{" not in o for o in out)
    assert template_key("Молоко {fat} {ml}") == "молоко"


def test_noise_is_reproducible_and_distinct():
    a = noisy_variants("Молоко пастеризованное 3,2% 900мл", 5, random.Random(5), "goods")
    b = noisy_variants("Молоко пастеризованное 3,2% 900мл", 5, random.Random(5), "goods")
    assert a == b
    assert len(set(a)) == len(a) and "Молоко пастеризованное 3,2% 900мл" not in a


def test_noise_never_returns_empty():
    rng = random.Random(0)
    for profile in ("goods", "meal", "service"):
        for _ in range(200):
            assert noisy_variant("Тако", rng, profile).strip()


def test_resolve_label_conflicts():
    df = pd.DataFrame({
        "text_clean": ["Мороженое", "мороженое", "Хлеб", "Хлеб", "Хлеб", "Масло", "масло"],
        "subcategory": ["food.dairy", "food.sweets_snacks", "food.bakery", "food.bakery", "food.bakery",
                        "food.dairy", "food.grocery"],
        "source": "seed", "profile": "goods", "group": "g",
    })
    out = resolve_label_conflicts(df)
    assert set(out["norm"]) == {"хлеб"}  # 50/50 conflicts dropped, unanimous kept


def _small_clean(n_per_sub=40):
    rows = []
    for sub in ["food.dairy", "food.bakery", "transport.taxi"]:
        for i in range(n_per_sub):
            name = f"{sub.split('.')[1]} товар {'абвгдежзик'[i % 10]}{'лмнопрстуф'[i // 10 % 10]}"
            rows.append((name, sub, "goods", f"g:{sub}:{i}", "seed"))
    return pd.DataFrame(rows, columns=["text_clean", "subcategory", "profile", "group", "source"])


def test_build_dataset_no_leakage_and_columns():
    df = build_dataset(_small_clean(), TAX, seed=1, target_train_per_sub=100)
    assert set(df["split"]) == {"train", "val", "test_synth"}
    assert sum(check_no_overlap(df).values()) == 0
    assert (df.groupby("subcategory").apply(lambda g: (g.split == "train").sum()) >= 100).all()
    assert {"text", "text_clean", "subcategory", "category", "source", "split"} <= set(df.columns)
    assert (df["category"] == df["subcategory"].str.split(".").str[0]).all()


def test_build_dataset_is_deterministic():
    a = build_dataset(_small_clean(), TAX, seed=3, target_train_per_sub=60)
    b = build_dataset(_small_clean(), TAX, seed=3, target_train_per_sub=60)
    pd.testing.assert_frame_equal(a, b)


def test_seed_only_full_build_meets_stage1_thresholds():
    rows = generate_seed_clean(random.Random(42), 2)
    clean = pd.DataFrame(rows, columns=["text_clean", "subcategory", "profile", "group"])
    clean["source"] = "seed"
    df = build_dataset(clean, TAX, seed=42)
    train = df[df.split == "train"].groupby("subcategory").size()
    assert len(train) == 48 and train.min() >= 300
    assert sum(check_no_overlap(df).values()) == 0


def test_sample_receipts_are_internally_consistent():
    exp = json.load(open(ROOT / "data" / "samples" / "expected.json", encoding="utf-8"))
    assert len(exp) >= 5
    for name, e in exp.items():
        assert (ROOT / "data" / "samples" / name).exists()
        assert abs(sum(i["amount"] for i in e["items"]) - e["total"]) <= 0.011, name
        txt = (ROOT / "data" / "samples" / name).read_text(encoding="utf-8")
        assert re.search(r"ИТОГ\s+" + f"{e['total']:.2f}", txt), name


def test_no_real_inn_in_samples():
    for p in (ROOT / "data" / "samples").glob("*.txt"):
        inns = re.findall(r"ИНН\s*(\d+)", p.read_text(encoding="utf-8"))
        assert all(set(i) == {"0"} for i in inns), p.name
