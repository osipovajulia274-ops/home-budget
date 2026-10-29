"""The numpy inference of the MLP must equal a dense reference computed from scikit-learn's TF-IDF features."""
import json

import numpy as np
import pytest

pytest.importorskip("sklearn")

from app.services.normalize import normalize
from app.services.mlp_classifier import MlpTfidfClassifier

TEXTS = ["МОЛОКО ПАСТ. 3,2% 900МЛ", "Хлеб Бородинский нарезной", "Яндекс Такси поездка", "Кофе капучино 300мл",
         "Корм для кошек 85г", "Абонемент фитнес", "Сыр Российский 200г", "Вода питьевая 1.5л"]
N_LABELS, HIDDEN = 5, 8


def fitted(tmp_path):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from ml.export_mlp import export_mlp

    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1, sublinear_tf=True, max_features=300,
                          preprocessor=normalize, lowercase=False)
    vec.fit(TEXTS)
    rng = np.random.default_rng(0)
    F = len(vec.vocabulary_)
    w1 = rng.normal(0, 0.3, (F, HIDDEN)).astype(np.float32)
    b1 = rng.normal(0, 0.1, HIDDEN).astype(np.float32)
    w2 = rng.normal(0, 0.5, (HIDDEN, N_LABELS)).astype(np.float32)
    b2 = rng.normal(0, 0.1, N_LABELS).astype(np.float32)
    info = export_mlp(vec, w1, b1, w2, b2, N_LABELS, tmp_path)
    return vec, (w1.astype(np.float16).astype(np.float32), b1, w2, b2), info


def reference(vec, params, texts):
    w1, b1, w2, b2 = params
    x = vec.transform(texts).toarray().astype(np.float32)
    logits = np.maximum(x @ w1 + b1, 0) @ w2 + b2
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def test_numpy_inference_matches_dense_reference(tmp_path):
    vec, params, info = fitted(tmp_path)
    clf = MlpTfidfClassifier.load(tmp_path)
    p = clf.predict_proba(TEXTS + ["совсем неизвестный текст qqq"])
    ref = reference(vec, params, TEXTS + ["совсем неизвестный текст qqq"])
    assert p.shape == (len(TEXTS) + 1, N_LABELS)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5)
    assert np.abs(p - ref).max() < 1e-4
    assert info["hidden"] == HIDDEN and info["features"] == len(vec.vocabulary_)


def test_text_without_known_ngrams_still_gives_a_distribution(tmp_path):
    fitted(tmp_path)
    p = MlpTfidfClassifier.load(tmp_path).predict_proba(["☃☃"])
    assert np.isfinite(p).all() and abs(p.sum() - 1.0) < 1e-5


def test_categorizer_can_use_the_mlp(tmp_path):
    from app.services import taxonomy as tx
    from app.services.categorizer import Categorizer

    fitted(tmp_path)
    labels = list(tx.all_subcategory_ids())[:N_LABELS] if hasattr(tx, "all_subcategory_ids") else None
    if labels is None:
        labels = [s["id"] for c in tx.tree() for s in c["subcategories"]][:N_LABELS]
    (tmp_path / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
    (tmp_path / "model_choice.json").write_text(json.dumps(
        {"classifier": "mlp", "threshold": 0.0, "unknown": "other.unknown"}), encoding="utf-8")
    c = Categorizer.load(tmp_path)
    preds = c.categorize(["Хлеб Бородинский нарезной", "Перевод Ивану"])
    assert preds[0].source == "model" and preds[0].subcategory in labels
    assert preds[1].subcategory == "other.transfers" and preds[1].source == "rule"
    assert c.kind == "mlp"


def test_mlp_choice_without_files_is_rejected():
    from app.services.categorizer import Categorizer

    with pytest.raises(ValueError):
        Categorizer(["food.dairy"], {"classifier": "mlp"})
