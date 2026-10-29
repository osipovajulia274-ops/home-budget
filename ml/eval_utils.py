"""Metrics, data loading and report helpers for the classifier (numpy/sklearn/pandas; no torch)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from app.services.normalize import normalize
from ml.seed_generator import load_taxonomy
from ml.tableio import read_table

UNKNOWN = "other.unknown"
THRESHOLD = 0.5   # guide section 5: max(softmax) < 0.5 -> other.unknown


def load_labels() -> List[str]:
    """Subcategory ids in taxonomy order = the order of the model outputs (labels.json)."""
    return list(load_taxonomy().keys())


def category_of(sub: str) -> str:
    return sub.split(".", 1)[0]


def category_names(labels: Sequence[str]) -> List[str]:
    seen: List[str] = []
    for s in labels:
        c = category_of(s)
        if c not in seen:
            seen.append(c)
    return seen


def label_to_category_index(labels: Sequence[str]) -> np.ndarray:
    cats = category_names(labels)
    return np.array([cats.index(category_of(s)) for s in labels], dtype=np.int64)


def load_dataset(path: str | Path) -> pd.DataFrame:
    df = read_table(path)
    need = {"text", "text_clean", "subcategory", "split"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"{path}: missing columns {sorted(missing)}")
    df["text"] = df["text"].astype(str)
    return df


def load_real_test(path: str | Path, labels: Sequence[str]) -> Optional[pd.DataFrame]:
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p, keep_default_na=False, encoding="utf-8-sig")
    df = df[df["subcategory"].isin(set(labels))].copy()
    df["text"] = df["text"].astype(str)
    if "receipt_id" not in df.columns:
        df["receipt_id"] = ""
    if "d6_split" not in df.columns:        # the part of D6 is the prefix of the receipt id ("test_17")
        part = df["receipt_id"].astype(str).str.split("_").str[0]
        if part.isin(["train", "val", "test"]).all():
            df["d6_split"] = part
    return df.reset_index(drop=True)


def class_weights(y: np.ndarray, n_classes: int, clip: float = 10.0) -> np.ndarray:
    """Inverse class frequency, spread (max/min) clipped to `clip`, scaled to mean 1 over the samples."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = counts[counts > 0].min() if (counts > 0).any() else 1.0
    w = counts.sum() / (n_classes * counts)
    w = np.minimum(w / w.min(), clip)           # min weight 1, at most clip
    sample_mean = float((w * np.bincount(y, minlength=n_classes)).sum() / max(len(y), 1))
    return (w / sample_mean).astype(np.float32)


def split_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str]) -> Dict[str, float]:
    """Subcategory and category metrics. Macro averages run over the classes that occur in y_true."""
    from sklearn.metrics import f1_score

    cat_idx = label_to_category_index(labels)
    ct, cp = cat_idx[y_true], cat_idx[y_pred]
    return {
        "n": int(len(y_true)),
        "n_classes_sub": int(len(np.unique(y_true))),
        "n_classes_cat": int(len(np.unique(ct))),
        "acc_sub": float((y_true == y_pred).mean()),
        "macro_f1_sub": float(f1_score(y_true, y_pred, labels=np.unique(y_true), average="macro", zero_division=0)),
        "acc_cat": float((ct == cp).mean()),
        "macro_f1_cat": float(f1_score(ct, cp, labels=np.unique(ct), average="macro", zero_division=0)),
    }


_DISH_UNITS = re.compile(r"\b(порц|шт|г|мл|л)\b\.?")


def dish_key(text: str) -> str:
    """Name of a menu line without numbers, units and portion marks (to compare with dictionary templates)."""
    t = _DISH_UNITS.sub("", normalize(re.sub(r"\{\w+\}", "", text)))
    return re.sub(r"\s+", " ", re.sub(r"[0-9,./\-]+", " ", t)).strip()


def seed_dictionary_keys(subcategory: str = "eating_out.cafe") -> set:
    """dish_key of every seed template of one subcategory (used to check how literally the test overlaps)."""
    from ml.seed_generator import load_seed, split_items

    return {dish_key(t) for t in split_items(load_seed()["subcategories"][subcategory]["items"])}


def real_test_views(real: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str],
                    dictionary_keys: Optional[set] = None) -> Dict[str, dict]:
    """The same metrics on the whole real test and on stricter views:
    - unique_names: repeated normalised names count once;
    - without_cafe: without eating_out.cafe (~45% of the set);
    - holdout_receipts / dev_receipts: receipts with d6_split == "test" versus the rest. The error analysis and
      the dish dictionary were worked out on the dev receipts, the holdout is what to quote;
    - without_dictionary_overlap: without the lines whose name is literally in the cafe seed dictionary."""
    norm = real["text"].map(normalize).to_numpy()
    first = ~pd.Series(norm).duplicated().to_numpy()
    cafe = np.array([labels[i] == "eating_out.cafe" for i in y_true])
    views = {
        "all_items": split_metrics(y_true, y_pred, labels),
        "unique_names": split_metrics(y_true[first], y_pred[first], labels),
        "without_cafe": split_metrics(y_true[~cafe], y_pred[~cafe], labels),
    }
    if "d6_split" in real.columns:
        hold = (real["d6_split"].astype(str) == "test").to_numpy()
        for name, m in (("holdout_receipts", hold), ("dev_receipts", ~hold)):
            if m.any():
                views[name] = split_metrics(y_true[m], y_pred[m], labels)
    if dictionary_keys:
        overlap = real["text"].map(dish_key).isin(dictionary_keys).to_numpy() & cafe
        if (~overlap).any():
            views["without_dictionary_overlap"] = split_metrics(y_true[~overlap], y_pred[~overlap], labels)
            views["without_dictionary_overlap"]["excluded_items"] = int(overlap.sum())
    return views


def choose_production(cnn: Dict[str, float], base: Dict[str, float], ensemble: Optional[Dict[str, float]] = None,
                      margin: float = 0.01) -> Tuple[str, str]:
    """Rule fixed on 2026-10-28, after the first two real runs and before the third (see PROGRESS.md):
    1. the ensemble (mean of the network and the baseline) is used only if it beats the baseline by more than
       `margin` on BOTH accuracy of categories and macro-F1 of categories on the whole real test;
    2. otherwise the network is used unless it is worse than the baseline by more than `margin` on either metric
       (a tie goes to the network: smaller file, ONNX-ready);
    3. otherwise the baseline (guide section 0: the network must not lose to the baseline on real data)."""
    def d(a, b):
        return a["acc_cat"] - b["acc_cat"], a["macro_f1_cat"] - b["macro_f1_cat"]

    da, df = d(cnn, base)
    txt = (f"real_test acc_cat / macro-F1 cat: cnn {cnn['acc_cat']:.3f} / {cnn['macro_f1_cat']:.3f}, "
           f"baseline {base['acc_cat']:.3f} / {base['macro_f1_cat']:.3f} (cnn - baseline {da:+.3f} / {df:+.3f})")
    if ensemble:
        ea, ef = d(ensemble, base)
        txt += f", ensemble {ensemble['acc_cat']:.3f} / {ensemble['macro_f1_cat']:.3f} (vs baseline {ea:+.3f} / {ef:+.3f})"
        if ea > margin and ef > margin:
            return "ensemble_cnn_baseline", txt + f"; margin {margin}"
    if da >= -margin and df >= -margin:
        return "char_cnn", txt + f"; margin {margin}"
    return "tfidf_logreg_baseline", txt + f"; margin {margin}"


def per_class_report(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str]) -> Dict[str, dict]:
    from sklearn.metrics import precision_recall_fscore_support

    present = np.unique(np.concatenate([y_true, y_pred]))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=present, zero_division=0)
    return {labels[c]: {"precision": round(float(p[i]), 3), "recall": round(float(r[i]), 3),
                        "f1": round(float(f[i]), 3), "support": int(s[i])} for i, c in enumerate(present)}


def confusion_by_category(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str]) -> np.ndarray:
    cat_idx = label_to_category_index(labels)
    n = len(category_names(labels))
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (cat_idx[y_true], cat_idx[y_pred]), 1)
    return cm


def plot_confusion(cm: np.ndarray, names: Sequence[str], path: str | Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=60, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted category")
    ax.set_ylabel("true category")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, label="share of the true class")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def apply_threshold(probs: np.ndarray, labels: Sequence[str], thr: float = THRESHOLD) -> np.ndarray:
    """Predictions with the production rule: max probability < thr -> other.unknown."""
    pred = probs.argmax(axis=1)
    if UNKNOWN in labels:
        pred = np.where(probs.max(axis=1) < thr, list(labels).index(UNKNOWN), pred)
    return pred


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def write_errors_md(path: str | Path, real: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray,
                    probs: np.ndarray, labels: Sequence[str], model_name: str, k: int = 10) -> None:
    """The k most confident mistakes on the real test, one per true class first (variety), plus the most
    frequent confusion pairs. The explanations are filled in by a human after reading the examples."""
    wrong = np.where(y_true != y_pred)[0]
    order = wrong[np.argsort(-probs[wrong, y_pred[wrong]])]
    picked: List[int] = []
    seen_true = set()
    for i in order:
        if y_true[i] in seen_true:
            continue
        picked.append(int(i)); seen_true.add(y_true[i])
        if len(picked) == k:
            break
    for i in order:
        if len(picked) >= k:
            break
        if int(i) not in picked:
            picked.append(int(i))
    pairs = pd.Series([(labels[t], labels[p]) for t, p in zip(y_true[wrong], y_pred[wrong])]).value_counts().head(10)
    lines = [f"# Typical mistakes of {model_name} on the real test (D6)", "",
             f"Mistakes: {len(wrong)} of {len(y_true)} items. The labels were assigned by Claude and only partly "
             "re-checked, so a fraction of the 'mistakes' may be label errors.", "",
             "## Most confident mistakes (one per true class)", "",
             "| text | true | predicted | p(predicted) |", "|---|---|---|---|"]
    for i in picked:
        lines.append(f"| {real['text'].iloc[i]} | {labels[y_true[i]]} | {labels[y_pred[i]]} | {probs[i, y_pred[i]]:.2f} |")
    lines += ["", "## Most frequent confusions (true -> predicted)", "", "| true | predicted | count |", "|---|---|---|"]
    for (t, p), c in pairs.items():
        lines.append(f"| {t} | {p} | {c} |")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
