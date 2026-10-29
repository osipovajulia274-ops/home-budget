"""Export the fitted scikit-learn TF-IDF + LogisticRegression baseline to numpy files (see
app/services/linear_classifier.py). Only the export touches scikit-learn objects."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def export_baseline(vec, clf, n_labels: int, out_dir: str | Path, coef_dtype=np.float16) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    vocab = [None] * len(vec.vocabulary_)
    for g, i in vec.vocabulary_.items():
        vocab[i] = g
    (out / "baseline_vocab.json").write_text(json.dumps(
        {"vocab": vocab, "n_labels": int(n_labels), "ngram_range": list(vec.ngram_range)}, ensure_ascii=False,
        separators=(",", ":")), encoding="utf-8")
    np.savez_compressed(out / "baseline.npz", idf=vec.idf_.astype(np.float32),
                        coef=np.ascontiguousarray(clf.coef_.T).astype(coef_dtype),
                        intercept=clf.intercept_.astype(np.float32), classes=clf.classes_.astype(np.int32))
    size = sum((out / n).stat().st_size for n in ("baseline_vocab.json", "baseline.npz")) / 1e6
    return {"features": len(vocab), "size_mb": round(size, 3), "coef_dtype": np.dtype(coef_dtype).name}
