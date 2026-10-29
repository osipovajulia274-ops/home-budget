"""Inference for the exported TF-IDF + one-hidden-layer neural network (MLP), numpy only.

The network is trained in PyTorch (ml/train_mlp.py) on the SAME features as the logistic-regression baseline:
character 2-5-gram TF-IDF (sublinear tf, l2 norm). `ml/export_mlp.py` writes
    mlp_vocab.json   n-grams in feature order and the settings
    mlp.npz          idf [F] f32, w1 [F, H] f16, b1 [H] f32, w2 [H, C] f32, b2 [C] f32
and this module computes  softmax(relu(x @ w1 + b1) @ w2 + b2)  for a sparse x. Only the rows of w1 that belong to the
n-grams of the text are read, so the 10 MB weight matrix is never expanded in memory.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from app.services.linear_classifier import char_ngrams
from app.services.normalize import normalize


class MlpTfidfClassifier:
    def __init__(self, vocab: Sequence[str], idf, w1, b1, w2, b2, n_labels: int, ngram_range=(2, 5)):
        self.index: Dict[str, int] = {g: i for i, g in enumerate(vocab)}
        self.idf = np.asarray(idf, dtype=np.float32)
        self.w1 = np.asarray(w1)                       # [F, H], float16 or float32
        self.b1 = np.asarray(b1, dtype=np.float32)
        self.w2 = np.asarray(w2, dtype=np.float32)     # [H, C]
        self.b2 = np.asarray(b2, dtype=np.float32)
        self.n_labels = int(n_labels)
        self.ngram_range = tuple(ngram_range)
        if self.w2.shape[1] != self.n_labels:
            raise ValueError("w2 does not match the number of labels")

    @classmethod
    def load(cls, directory: str | Path) -> "MlpTfidfClassifier":
        d = Path(directory)
        meta = json.loads((d / "mlp_vocab.json").read_text(encoding="utf-8"))
        z = np.load(d / "mlp.npz")
        return cls(meta["vocab"], z["idf"], z["w1"], z["b1"], z["w2"], z["b2"], meta["n_labels"],
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
        out = np.zeros((len(texts), self.n_labels), dtype=np.float32)
        for r, t in enumerate(texts):
            idx, val = self._features(t)
            h = self.b1.copy()
            if idx.size:
                h += val @ self.w1[idx].astype(np.float32)
            logits = np.maximum(h, 0.0) @ self.w2 + self.b2
            logits -= logits.max()
            e = np.exp(logits)
            out[r] = e / e.sum()
        return out
