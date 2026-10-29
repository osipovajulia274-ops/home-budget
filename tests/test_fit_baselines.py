import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from ml import eval_utils as eu
from ml.seed_generator import generate_seed_clean
from ml.train_classifier import fit_baselines


def _data():
    labels = eu.load_labels()
    l2i = {s: i for i, s in enumerate(labels)}
    rows = pd.DataFrame(generate_seed_clean(random.Random(0), 2), columns=["text", "subcategory", "profile", "group"])
    bucket = rows["group"].map(lambda g: sum(map(ord, g)) % 10)
    parts = {"train": rows[bucket < 8].reset_index(drop=True), "test_synth": rows[bucket == 8].reset_index(drop=True)}
    real = rows[bucket == 9].reset_index(drop=True).head(200).copy()
    real["receipt_id"] = ["test_%d" % (i // 5) if i % 2 else "train_%d" % (i // 5) for i in range(len(real))]
    real["d6_split"] = real["receipt_id"].str.split("_").str[0]
    y = {k: v["subcategory"].map(l2i).to_numpy() for k, v in parts.items()}
    y_real = real["subcategory"].map(l2i).to_numpy()
    return labels, parts, y, real, y_real


def _fake_probs(y, n, seed):
    rng = np.random.default_rng(seed)
    p = rng.random((len(y), n)) * 0.2
    p[np.arange(len(y)), y] += 1.0
    return p / p.sum(1, keepdims=True)


def test_fit_baselines_end_to_end(tmp_path):
    labels, parts, y, real, y_real = _data()
    weights = eu.class_weights(y["train"], len(labels))
    args = SimpleNamespace(baseline_iter=40, baseline_features=3000)
    out = fit_baselines(parts, y, labels, weights, args, real, y_real, _fake_probs(y_real, len(labels), 1),
                        _fake_probs(y["test_synth"], len(labels), 2), out_dir=tmp_path)
    assert set(out) == {"baseline_full", "baseline", "ensemble"}
    b = out["baseline"]
    assert b["numpy_vs_sklearn_argmax_agreement"] >= 0.99
    assert b["features"] <= 3000 and b["size_mb"] < 10
    assert (tmp_path / "baseline.npz").exists() and (tmp_path / "baseline_vocab.json").exists()
    for k in ("all_items", "holdout_receipts", "dev_receipts", "without_dictionary_overlap"):
        assert k in b["real_test_views"], k
    assert out["ensemble"]["real_test"]["acc_cat"] >= 0.5
    choice, why = eu.choose_production(out["ensemble"]["real_test"], b["real_test"], out["ensemble"]["real_test"])
    assert choice in {"ensemble_cnn_baseline", "char_cnn", "tfidf_logreg_baseline"} and "acc_cat" in why
