"""Inference for the exported TF-IDF (character 2-5-grams) + logistic regression classifier, numpy only.

Production does not import scikit-learn: `ml/export_baseline.py` writes the fitted model as
    baseline_vocab.json   n-grams in feature order and the settings
    baseline.npz          idf [F] float32, coef [F, C] float16 (feature-major), intercept [C] float32, classes [C] int32
and this module reproduces scikit-learn's `TfidfVectorizer(analyzer="char", ngram_range=(2, 5), sublinear_tf=True,
preprocessor=normalize, lowercase=False)` followed by `LogisticRegression.predict_proba` (multinomial softmax).
Texts go through the same `normalize()` as everywhere else.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from app.services.normalize import normalize

_WHITE = re.compile(r"\s\s+")        # scikit-learn's _white_spaces pattern


def char_ngrams(text: str, lo: int = 2, hi: int = 5) -> List[str]:
    """Same n-grams as scikit-learn's `analyzer="char"`."""
    text = _WHITE.sub(" ", text)
    n_chars = len(text)
    out: List[str] = []
    for n in range(lo, min(hi + 1, n_chars + 1)):
        for i in range(n_chars - n + 1):
            out.append(text[i:i + n])
    return out


class LinearTfidfClassifier:
    def __init__(self, vocab: Sequence[str], idf: np.ndarray, coef: np.ndarray, intercept: np.ndarray,
                 classes: np.ndarray, n_labels: int, ngram_range=(2, 5)):
        self.index: Dict[str, int] = {g: i for i, g in enumerate(vocab)}
        self.idf = np.asarray(idf, dtype=np.float32)
        self.coef = np.asarray(coef)                       # [F, C], float16 or float32
        self.intercept = np.asarray(intercept, dtype=np.float32)
        self.classes = np.asarray(classes, dtype=np.int64)
        self.n_labels = int(n_labels)
        self.ngram_range = tuple(ngram_range)

    @classmethod
    def load(cls, directory: str | Path) -> "LinearTfidfClassifier":
        d = Path(directory)
        meta = json.loads((d / "baseline_vocab.json").read_text(encoding="utf-8"))
        z = np.load(d / "baseline.npz")
        return cls(meta["vocab"], z["idf"], z["coef"], z["intercept"], z["classes"], meta["n_labels"],
                   meta.get("ngram_range", (2, 5)))

    def _features(self, text: str):
        counts: Dict[int, int] = {}
        for g in char_ngrams(normalize(text), *self.ngram_range):
            j = self.index.get(g)
            if j is not None:
                counts[j] = counts.get(j, 0) + 1
        if not counts:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
        idx = np.fromiter(counts.keys(), dtype=np.int64, count=len(counts))
        tf = np.fromiter(counts.values(), dtype=np.float32, count=len(counts))
        val = (1.0 + np.log(tf)) * self.idf[idx]
        norm = float(np.sqrt((val * val).sum()))
        return idx, (val / norm if norm > 0 else val)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """[N, n_labels] probabilities; labels that were absent from training get 0."""
        out = np.zeros((len(texts), self.n_labels), dtype=np.float32)
        for r, t in enumerate(texts):
            idx, val = self._features(t)
            logits = self.intercept.astype(np.float32).copy()
            if idx.size:
                logits += val @ self.coef[idx].astype(np.float32)
            logits -= logits.max()
            e = np.exp(logits)
            out[r, self.classes] = e / e.sum()
        return out
