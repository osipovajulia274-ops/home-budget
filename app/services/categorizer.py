"""Runtime categoriser (guide 8.5), adapted to the model chosen at the end of Stage 2.

`app/ml_artifacts/model_choice.json` says which classifier ships:
  tfidf_logreg_baseline  numpy-only TF-IDF + logistic regression (what the Stage 2 rule picked)
  char_cnn               ONNX char-CNN (onnxruntime is imported only in this case)
  ensemble               mean of both probability vectors

Priority for every item: (1) the user's own earlier correction for the same normalised title,
(2) transfer rule, (3) the model; probability < threshold -> `other.unknown`.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np

from app.config import ARTIFACTS_DIR
from app.services import taxonomy as tx
from app.services.linear_classifier import LinearTfidfClassifier
from app.services.normalize import encode_batch, normalize

TRANSFER_TITLE_RE = re.compile(r"перевод|между своими|на свою карту", re.I)
LOW_CONFIDENCE = 0.7       # the preview highlights anything below this


@dataclass
class Prediction:
    subcategory: str
    category: str
    confidence: float
    source: str                                # model | user | rule
    alternatives: List[Tuple[str, float]] = field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        return self.subcategory == tx.UNKNOWN or self.confidence < LOW_CONFIDENCE


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


class Categorizer:
    def __init__(self, labels: Sequence[str], choice: dict, baseline: Optional[LinearTfidfClassifier] = None,
                 onnx_path: Optional[Path] = None):
        self.labels = list(labels)
        self.kind: str = choice.get("classifier", "tfidf_logreg_baseline")
        self.threshold: float = float(choice.get("threshold", 0.5))
        self.unknown: str = choice.get("unknown", tx.UNKNOWN)
        self.baseline = baseline
        self._onnx_path = onnx_path
        self._session = None                  # created lazily: onnxruntime is only needed for cnn / ensemble
        if self.kind != "char_cnn" and baseline is None:
            raise ValueError("baseline model files are required for this classifier choice")

    @classmethod
    def load(cls, directory: str | Path = ARTIFACTS_DIR) -> "Categorizer":
        d = Path(directory)
        labels = json.loads((d / "labels.json").read_text(encoding="utf-8"))
        choice = json.loads((d / "model_choice.json").read_text(encoding="utf-8"))
        kind = choice.get("classifier", "tfidf_logreg_baseline")
        baseline = LinearTfidfClassifier.load(d) if kind in ("tfidf_logreg_baseline", "ensemble") else None
        onnx = d / "classifier.onnx" if kind in ("char_cnn", "ensemble") else None
        return cls(labels, choice, baseline, onnx)

    # ---- model ----
    def _cnn_proba(self, texts: Sequence[str]) -> np.ndarray:
        if self._session is None:
            import onnxruntime as ort                 # lazy: not imported (and not in memory) for the baseline

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            self._session = ort.InferenceSession(str(self._onnx_path), opts, providers=["CPUExecutionProvider"])
        logits = self._session.run(None, {self._session.get_inputs()[0].name: encode_batch(texts)})[0]
        return _softmax(np.asarray(logits, dtype=np.float32))

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, len(self.labels)), dtype=np.float32)
        if self.kind == "char_cnn":
            return self._cnn_proba(texts)
        assert self.baseline is not None
        p = self.baseline.predict_proba(list(texts))
        if self.kind == "ensemble":
            p = (p + self._cnn_proba(texts)) / 2.0
        return p

    # ---- public API ----
    def categorize(self, titles: Sequence[str], feedback: Optional[Mapping[str, str]] = None) -> List[Prediction]:
        """One call for the whole receipt. `feedback` maps normalised title -> the user's subcategory."""
        feedback = feedback or {}
        out: List[Optional[Prediction]] = [None] * len(titles)
        todo: List[int] = []
        for i, t in enumerate(titles):
            key = normalize(t)
            fb = feedback.get(key)
            if fb and tx.is_valid_sub(fb):
                out[i] = Prediction(fb, tx.category_of(fb), 1.0, "user")
            elif TRANSFER_TITLE_RE.search(t or ""):
                out[i] = Prediction(tx.TRANSFERS, tx.category_of(tx.TRANSFERS), 1.0, "rule")
            else:
                todo.append(i)
        if todo:
            proba = self.predict_proba([titles[i] for i in todo])
            for row, i in zip(proba, todo):
                order = np.argsort(-row)[:4]
                top = int(order[0])
                conf = float(row[top])
                alts = [(self.labels[int(j)], round(float(row[int(j)]), 4)) for j in order[1:4] if row[int(j)] > 0.01]
                sub = self.labels[top] if conf >= self.threshold else self.unknown
                if conf < self.threshold:
                    alts = [(self.labels[top], round(conf, 4))] + alts[:2]
                out[i] = Prediction(sub, tx.category_of(sub), round(conf, 4), "model", alts)
        return out  # type: ignore[return-value]


_instance: Optional[Categorizer] = None


def get_categorizer() -> Categorizer:
    """Loaded once per process (a single uvicorn worker on Render)."""
    global _instance
    if _instance is None:
        _instance = Categorizer.load()
    return _instance
