"""Stage 2: train the char-CNN classifier and the TF-IDF baseline, evaluate on test_synth and on the REAL test.

  python -m ml.train_classifier                 # full run (Colab GPU or CPU)
  python -m ml.train_classifier --smoke         # 1 epoch on a small subset: checks that everything runs

Writes:
  ml/models_out/classifier.pt                   state dict of the best epoch (by val macro-F1)
  ml/models_out/labels.json                     subcategory ids in output order
  ml/reports/metrics_classifier.json            metrics (guide 14.2 format + extra views)
  ml/reports/confusion_matrix.png               13 categories, real test
  ml/reports/errors_examples.md                 the most confident mistakes on the real test

The real test (D6, labelled by hand / by Claude) is NEVER used for training, early stopping or class weights.
Only validation macro-F1 chooses the epoch.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.normalize import MAX_LEN, VOCAB_SIZE, encode_batch, normalize, vocab_dict  # noqa: E402
from ml import eval_utils as eu  # noqa: E402

PROCESSED = ROOT / "ml" / "data" / "processed"
REPORTS = ROOT / "ml" / "reports"
MODELS_OUT = ROOT / "ml" / "models_out"


def set_seed(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def predict_logits(model, X: np.ndarray, device, batch: int = 1024) -> np.ndarray:
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).to(device)
            out.append(model(xb).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 0), dtype=np.float32)


def train_cnn(Xtr, ytr, Xva, yva, labels, args, device):
    import torch
    import torch.nn.functional as F
    from ml.models import CharCNN, count_params

    n = len(labels)
    model = CharCNN(n).to(device)
    weights = eu.class_weights(ytr, n, clip=10.0)
    w_t = torch.from_numpy(weights).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    X_t = torch.from_numpy(Xtr).to(device)
    y_t = torch.from_numpy(ytr).to(device)
    rng = np.random.default_rng(args.seed)

    history, best_f1, best_state, best_epoch, bad = [], -1.0, None, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.from_numpy(rng.permutation(len(Xtr))).to(device)
        total, seen, t0 = 0.0, 0, time.time()
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad()
            loss = F.cross_entropy(model(X_t[idx]), y_t[idx], weight=w_t)
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
            seen += len(idx)
        val_pred = predict_logits(model, Xva, device).argmax(axis=1)
        m = eu.split_metrics(yva, val_pred, labels)
        history.append({"epoch": epoch, "train_loss": total / max(seen, 1), "val_macro_f1_sub": m["macro_f1_sub"],
                        "val_acc_cat": m["acc_cat"], "seconds": round(time.time() - t0, 1)})
        print(f"[cnn] epoch {epoch:2d}  loss {total / max(seen, 1):.4f}  val macro-F1(sub) {m['macro_f1_sub']:.4f}  "
              f"val acc(cat) {m['acc_cat']:.4f}  {history[-1]['seconds']}s")
        if m["macro_f1_sub"] > best_f1 + 1e-4:
            best_f1, best_epoch, bad = m["macro_f1_sub"], epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[cnn] early stopping (no improvement for {args.patience} epochs)")
                break
    model.load_state_dict(best_state)
    return model, weights, history, best_epoch, count_params(model)


def train_baseline(train_texts, ytr, labels, weights, max_iter: int, max_features: int = 200_000):
    """TF-IDF over character 2-5-grams + logistic regression, trained from scratch, comparison only."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=max_features,
                          preprocessor=normalize, lowercase=False)
    Xs = vec.fit_transform(train_texts)
    clf = LogisticRegression(C=10.0, max_iter=max_iter, class_weight={int(c): float(weights[c]) for c in np.unique(ytr)})
    clf.fit(Xs, ytr)
    return vec, clf


def baseline_probs(vec, clf, texts, n_classes: int) -> np.ndarray:
    p = clf.predict_proba(vec.transform(texts))
    full = np.zeros((p.shape[0], n_classes), dtype=np.float64)
    full[:, clf.classes_] = p
    return full


def fit_baselines(parts, y, labels, weights, args, real, y_real, probs_real, p_syn, out_dir=None) -> dict:
    """The full TF-IDF baseline (200k features, comparison only), the lean one that is exported to numpy files
    and evaluated through the production code, and the ensemble (mean of network and lean baseline)."""
    from app.services.linear_classifier import LinearTfidfClassifier
    from ml.export_baseline import export_baseline

    out: dict = {}
    out_dir = out_dir or MODELS_OUT
    keys = eu.seed_dictionary_keys()
    t0 = time.time()
    vec, clf = train_baseline(parts["train"]["text"].tolist(), y["train"], labels, weights, args.baseline_iter)
    bp_syn = baseline_probs(vec, clf, parts["test_synth"]["text"].tolist(), len(labels))
    out["baseline_full"] = {"name": "tfidf_char_logreg_200k", "fit_seconds": round(time.time() - t0, 1),
                            "test_synth": eu.split_metrics(y["test_synth"], bp_syn.argmax(1), labels)}
    if probs_real is not None:
        bpf = baseline_probs(vec, clf, real["text"].tolist(), len(labels))
        out["baseline_full"]["real_test_views"] = eu.real_test_views(real, y_real, bpf.argmax(1), labels, keys)
    print(f"[baseline] full fitted in {time.time() - t0:.0f}s")

    t0 = time.time()
    vec, clf = train_baseline(parts["train"]["text"].tolist(), y["train"], labels, weights, args.baseline_iter,
                              args.baseline_features)
    info = export_baseline(vec, clf, len(labels), out_dir)
    lean = LinearTfidfClassifier.load(out_dir)               # evaluate exactly what production will run
    sk_syn = baseline_probs(vec, clf, parts["test_synth"]["text"].tolist(), len(labels))
    np_syn = lean.predict_proba(parts["test_synth"]["text"].tolist())
    t1 = time.time()
    lean.predict_proba(parts["test_synth"]["text"].head(50).tolist())
    lat = (time.time() - t1) * 1000
    out["baseline"] = {"name": f"tfidf_char_logreg_{args.baseline_features // 1000}k_numpy",
                       "fit_seconds": round(time.time() - t0, 1), **info, "latency_ms_50_items": round(lat, 1),
                       "numpy_vs_sklearn_argmax_agreement": float((np_syn.argmax(1) == sk_syn.argmax(1)).mean()),
                       "numpy_vs_sklearn_max_abs_diff": float(np.abs(np_syn - sk_syn).max()),
                       "test_synth": eu.split_metrics(y["test_synth"], np_syn.argmax(1), labels)}
    if probs_real is not None:
        bp_real = lean.predict_proba(real["text"].tolist())
        out["baseline"]["real_test"] = eu.split_metrics(y_real, bp_real.argmax(1), labels)
        out["baseline"]["real_test_views"] = eu.real_test_views(real, y_real, bp_real.argmax(1), labels, keys)
        ens_real = 0.5 * probs_real + 0.5 * bp_real     # plain mean, nothing tuned on any test data
        out["ensemble"] = {
            "name": "mean(char_cnn, lean_baseline)",
            "test_synth": eu.split_metrics(y["test_synth"], (0.5 * p_syn + 0.5 * np_syn).argmax(1), labels),
            "real_test": eu.split_metrics(y_real, ens_real.argmax(1), labels),
            "real_test_views": eu.real_test_views(real, y_real, ens_real.argmax(1), labels, keys)}
    print(f"[baseline] lean ({info['features']} features, {info['size_mb']} MB) fitted in {time.time() - t0:.0f}s")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(PROCESSED / "category_dataset.parquet"))
    ap.add_argument("--real-test", default=str(PROCESSED / "real_test_labeled.csv"))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--baseline-iter", type=int, default=100)
    ap.add_argument("--baseline-features", type=int, default=40_000,
                    help="size of the exported (lean) baseline; the 200k-feature one is only fitted for comparison")
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny subset, 1 epoch, small baseline")
    args = ap.parse_args()

    import torch

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}")

    labels = eu.load_labels()
    l2i = {s: i for i, s in enumerate(labels)}
    df = eu.load_dataset(args.data)
    df = df[df["subcategory"].isin(l2i)]
    parts = {k: df[df["split"] == k] for k in ("train", "val", "test_synth")}
    if args.smoke:
        parts["train"] = parts["train"].sample(min(3000, len(parts["train"])), random_state=args.seed)
        parts["val"] = parts["val"].head(500)
        parts["test_synth"] = parts["test_synth"].head(500)
        args.epochs, args.baseline_iter = 1, 20
    real = eu.load_real_test(args.real_test, labels)
    if real is None:
        print("[train] WARNING: real_test_labeled.csv not found - real-test metrics will be missing")

    enc = {k: encode_batch(v["text"].tolist()) for k, v in parts.items()}
    y = {k: v["subcategory"].map(l2i).to_numpy(dtype=np.int64) for k, v in parts.items()}
    print({k: len(v) for k, v in parts.items()}, "real:", None if real is None else len(real))

    # ---- CNN
    model, weights, history, best_epoch, n_params = train_cnn(enc["train"], y["train"], enc["val"], y["val"],
                                                              labels, args, device)
    MODELS_OUT.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, MODELS_OUT / "classifier.pt")
    (MODELS_OUT / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8")

    def evaluate(texts, y_true, name):
        logits = predict_logits(model, encode_batch(texts), device)
        probs = eu.softmax(logits)
        return probs

    result = {"model": "char_cnn_v1", "trained_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
              "seed": args.seed, "train_size": len(parts["train"]), "val_size": len(parts["val"]),
              "test_synth_size": len(parts["test_synth"]), "real_test_size": 0 if real is None else len(real),
              "n_params": n_params, "best_epoch": best_epoch, "epochs_run": len(history),
              "class_weight_range": [float(weights.min()), float(weights.max())],
              "device": str(device), "smoke": bool(args.smoke), "metrics": {}, "baseline": {}, "history": history}

    p_syn = evaluate(parts["test_synth"]["text"].tolist(), y["test_synth"], "test_synth")
    result["metrics"]["val"] = eu.split_metrics(y["val"], evaluate(parts["val"]["text"].tolist(), y["val"], "val").argmax(1), labels)
    result["metrics"]["test_synth"] = eu.split_metrics(y["test_synth"], p_syn.argmax(1), labels)

    probs_real = None
    if real is not None and len(real):
        y_real = real["subcategory"].map(l2i).to_numpy(dtype=np.int64)
        probs_real = evaluate(real["text"].tolist(), y_real, "real")
        pred_real = probs_real.argmax(1)
        views = eu.real_test_views(real, y_real, pred_real, labels, eu.seed_dictionary_keys())
        result["metrics"]["real_test"] = views["all_items"]
        result["metrics"]["real_test_views"] = views
        thr_pred = eu.apply_threshold(probs_real, labels)
        result["metrics"]["real_test_with_threshold_0.5"] = {
            **eu.split_metrics(y_real, thr_pred, labels),
            "share_routed_to_unknown": float((thr_pred == l2i[eu.UNKNOWN]).mean())}
        result["per_class_real_test"] = eu.per_class_report(y_real, pred_real, labels)
        eu.plot_confusion(eu.confusion_by_category(y_real, pred_real, labels), eu.category_names(labels),
                          REPORTS / "confusion_matrix.png", "char-CNN, real test (D6), 13 categories")
        eu.write_errors_md(REPORTS / "errors_examples.md", real, y_real, pred_real, probs_real, labels, "char-CNN")

    # ---- baselines (full for comparison, lean exported) and the ensemble
    if not args.no_baseline:
        result.update(fit_baselines(parts, y, labels, weights, args, real, y_real if probs_real is not None else None,
                                    probs_real, p_syn))

    # ---- production choice: fixed rule (eu.choose_production), see its docstring
    cnn_real = result["metrics"].get("real_test")
    base_real = result["baseline"].get("real_test")
    if cnn_real and base_real:
        result["production_choice"], result["production_choice_reason"] = eu.choose_production(
            cnn_real, base_real, result.get("ensemble", {}).get("real_test"))
        (MODELS_OUT / "model_choice.json").write_text(json.dumps(
            {"classifier": result["production_choice"], "reason": result["production_choice_reason"],
             "threshold": eu.THRESHOLD, "unknown": eu.UNKNOWN}, ensure_ascii=False, indent=1), encoding="utf-8")
    result["caveats"] = [
        "val and test_synth contain augmented lines of synthetic and Open*Facts names: they overestimate real quality.",
        "40 of 48 subcategories are produced from the hand-written seed dictionary (synthetic).",
        "real_test labels were assigned by Claude and re-checked only on a ~10% sample by Claude itself.",
        "real_test has no housing/education items and ~45% of its lines are eating_out.cafe from a few receipts.",
        "The cafe dish dictionary (about 300 names) was written on 2026-10-28 after the first run showed a recall of 9% "
        "for eating_out.cafe; it comes from general knowledge of menus, but the author had seen the real test, so quote "
        "holdout_receipts and without_dictionary_overlap rather than all_items.",
    ]

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / ("metrics_classifier_smoke.json" if args.smoke else "metrics_classifier.json")
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[train] wrote {out}")
    print("[train] test_synth:", {k: round(v, 4) for k, v in result["metrics"]["test_synth"].items() if isinstance(v, float)})
    if "real_test" in result["metrics"]:
        print("[train] real_test :", {k: round(v, 4) for k, v in result["metrics"]["real_test"].items() if isinstance(v, float)})
        print("[train] baseline real_test:", {k: round(v, 4) for k, v in result["baseline"].get("real_test", {}).items() if isinstance(v, float)})
        if "ensemble" in result:
            print("[train] ensemble real_test:", {k: round(v, 4) for k, v in result["ensemble"]["real_test"].items() if isinstance(v, float)})
        print("[train] production choice:", result.get("production_choice"), "|", result.get("production_choice_reason"))


if __name__ == "__main__":
    main()
