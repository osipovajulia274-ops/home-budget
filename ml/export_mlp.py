"""Export the trained MLP weights (numpy arrays) next to the TF-IDF vocabulary; see app/services/mlp_classifier.py."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def export_mlp(vec, w1, b1, w2, b2, n_labels: int, out_dir, w1_dtype=np.float16) -> dict:
    """`vec` is the fitted scikit-learn TfidfVectorizer; w1 [F, H], b1 [H], w2 [H, C], b2 [C]."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    vocab = [None] * len(vec.vocabulary_)
    for g, i in vec.vocabulary_.items():
        vocab[i] = g
    w1, w2 = np.asarray(w1), np.asarray(w2)
    if w1.shape[0] != len(vocab) or w2.shape[0] != w1.shape[1]:
        raise ValueError("weight shapes do not match the vocabulary")
    (out / "mlp_vocab.json").write_text(json.dumps(
        {"vocab": vocab, "n_labels": int(n_labels), "ngram_range": list(vec.ngram_range), "hidden": int(w1.shape[1])},
        ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    np.savez_compressed(out / "mlp.npz", idf=vec.idf_.astype(np.float32), w1=np.ascontiguousarray(w1).astype(w1_dtype),
                        b1=np.asarray(b1, dtype=np.float32), w2=w2.astype(np.float32), b2=np.asarray(b2, dtype=np.float32))
    size = sum((out / n).stat().st_size for n in ("mlp_vocab.json", "mlp.npz")) / 1e6
    return {"features": len(vocab), "hidden": int(w1.shape[1]), "size_mb": round(size, 3),
            "w1_dtype": np.dtype(w1_dtype).name}
