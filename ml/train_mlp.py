"""Neural classifier #2: a one-hidden-layer MLP on character n-gram TF-IDF features, trained from scratch (PyTorch).

  python -m ml.train_mlp               # full run (Colab GPU or CPU)
  python -m ml.train_mlp --smoke       # tiny subset, 2 epochs: checks that everything runs

It uses exactly the same features and the same lean baseline (TF-IDF 40k + logistic regression) as
ml/train_classifier.py, so the only difference between the two models is the hidden layer.
Writes (nothing in app/ is touched):
  ml/models_out/mlp_vocab.json, ml/models_out/mlp.npz     weights for app/services/mlp_classifier.py
  ml/models_out/mlp_choice.json                           which model the fixed rule picks (see eval_utils.choose_production)
  ml/reports/metrics_mlp.json                             metrics, same views as metrics_classifier.json

The real test (D6, labelled by hand / by Claude) is NEVER used for training or for choosing the epoch:
only validation macro-F1 chooses the epoch. The decision rule was fixed BEFORE this run: the MLP replaces the baseline
unless it is worse by more than 0.01 on accuracy or macro-F1 of categories on the whole real test (a tie goes to the
neural network, the same convention as for the char-CNN).
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

from app.services.normalize import normalize  # noqa: E402
from ml import eval_utils as eu  # noqa: E402
from ml.export_mlp import export_mlp  # noqa: E402
from ml.train_classifier import baseline_probs, train_baseline  # noqa: E402

PROCESSED = ROOT / "ml" / "data" / "processed"
REPORTS = ROOT / "ml" / "reports"
MODELS_OUT = ROOT / "ml" / "models_out"


def batch_from_csr(X, rows, device):
    """EmbeddingBag input for a slice of a CSR matrix: flat feature ids, row offsets, tf-idf values."""
    import torch

    sub = X[rows]
    idx = torch.from_numpy(sub.indices.astype(np.int64)).to(device)
    off = torch.from_numpy(sub.indptr[:-1].astype(np.int64)).to(device)
    w = torch.from_numpy(sub.data.astype(np.float32)).to(device)
    return idx, off, w


def build_model(n_features: int, hidden: int, n_classes: int, dropout: float):
    import torch
    import torch.nn as nn

    class SparseMLP(nn.Module):
        """x (sparse tf-idf) -> Linear(F, H) + ReLU + Dropout -> Linear(H, C). The first layer is an EmbeddingBag with
        per-sample weights, which is exactly x @ W1 for a sparse x."""

        def __init__(self):
            super().__init__()
            self.emb = nn.EmbeddingBag(n_features, hidden, mode="sum")
            nn.init.normal_(self.emb.weight, mean=0.0, std=0.1)
            self.b1 = nn.Parameter(torch.zeros(hidden))
            self.drop = nn.Dropout(dropout)
            self.out = nn.Linear(hidden, n_classes)

        def forward(self, idx, off, w):
            h = self.emb(idx, off, per_sample_weights=w) + self.b1
            return self.out(self.drop(torch.relu(h)))

    return SparseMLP()


def predict_logits(model, X, device, batch: int = 2048) -> np.ndarray:
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch):
            idx, off, w = batch_from_csr(X, np.arange(i, min(i + batch, X.shape[0])), device)
            out.append(model(idx, off, w).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 0), dtype=np.float32)


def train(Xtr, ytr, Xva, yva, labels, args, device):
    import torch
    import torch.nn.functional as F

    n = len(labels)
    model = build_model(Xtr.shape[1], args.hidden, n, args.dropout).to(device)
    weights = eu.class_weights(ytr, n, clip=10.0)
    w_t = torch.from_numpy(weights).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    y_t = torch.from_numpy(ytr).to(device)
    rng = np.random.default_rng(args.seed)

    history, best_f1, best_state, best_epoch, bad = [], -1.0, None, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = rng.permutation(Xtr.shape[0])
        total, seen, t0 = 0.0, 0, time.time()
        for i in range(0, len(perm), args.batch):
            rows = perm[i:i + args.batch]
            idx, off, w = batch_from_csr(Xtr, rows, device)
            opt.zero_grad()
            loss = F.cross_entropy(model(idx, off, w), y_t[torch.from_numpy(rows).to(device)], weight=w_t)
            loss.backward()
            opt.step()
            total += float(loss) * len(rows)
            seen += len(rows)
        m = eu.split_metrics(yva, predict_logits(model, Xva, device).argmax(axis=1), labels)
        history.append({"epoch": epoch, "train_loss": total / max(seen, 1), "val_macro_f1_sub": m["macro_f1_sub"],
                        "val_acc_cat": m["acc_cat"], "seconds": round(time.time() - t0, 1)})
        print(f"[mlp] epoch {epoch:2d}  loss {total / max(seen, 1):.4f}  val macro-F1(sub) {m['macro_f1_sub']:.4f}  "
              f"val acc(cat) {m['acc_cat']:.4f}  {history[-1]['seconds']}s")
        if m["macro_f1_sub"] > best_f1 + 1e-4:
            best_f1, best_epoch, bad = m["macro_f1_sub"], epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[mlp] early stopping (no improvement for {args.patience} epochs)")
                break
    model.load_state_dict(best_state)
    return model, weights, history, best_epoch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(PROCESSED / "category_dataset.parquet"))
    ap.add_argument("--real-test", default=str(PROCESSED / "real_test_labeled.csv"))
    ap.add_argument("--features", type=int, default=40_000)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--baseline-iter", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    import torch
    from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401  (fails early if scikit-learn is missing)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mlp] device: {device}")

    labels = eu.load_labels()
    l2i = {s: i for i, s in enumerate(labels)}
    df = eu.load_dataset(args.data)
    df = df[df["subcategory"].isin(l2i)]
    parts = {k: df[df["split"] == k] for k in ("train", "val", "test_synth")}
    if args.smoke:
        parts["train"] = parts["train"].sample(min(3000, len(parts["train"])), random_state=args.seed)
        parts["val"], parts["test_synth"] = parts["val"].head(500), parts["test_synth"].head(500)
        args.epochs, args.baseline_iter, args.features = 2, 20, 5000
    real = eu.load_real_test(args.real_test, labels)
    if real is None:
        print("[mlp] WARNING: real_test_labeled.csv not found - the choice between models cannot be made")
    y = {k: v["subcategory"].map(l2i).to_numpy(dtype=np.int64) for k, v in parts.items()}
    print({k: len(v) for k, v in parts.items()}, "real:", None if real is None else len(real))

    # the lean baseline (same features); its vectorizer is reused for the MLP
    weights = eu.class_weights(y["train"], len(labels), clip=10.0)
    t0 = time.time()
    vec, clf = train_baseline(parts["train"]["text"].tolist(), y["train"], labels, weights, args.baseline_iter,
                              args.features)
    print(f"[baseline] fitted in {time.time() - t0:.0f}s ({len(vec.vocabulary_)} features)")
    X = {k: vec.transform(v["text"].tolist()).tocsr().astype(np.float32) for k, v in parts.items()}

    model, weights, history, best_epoch = train(X["train"], y["train"], X["val"], y["val"], labels, args, device)
    n_params = int(sum(p.numel() for p in model.parameters()))

    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    w1, b1, w2, b2 = sd["emb.weight"], sd["b1"], sd["out.weight"].T.copy(), sd["out.bias"]
    MODELS_OUT.mkdir(parents=True, exist_ok=True)
    info = export_mlp(vec, w1, b1, w2, b2, len(labels), MODELS_OUT)

    # evaluate exactly what production will run (numpy, float16 first layer) and compare with torch
    from app.services.mlp_classifier import MlpTfidfClassifier

    prod = MlpTfidfClassifier.load(MODELS_OUT)

    def torch_probs(texts):
        Xt = vec.transform(texts).tocsr().astype(np.float32)
        return eu.softmax(predict_logits(model, Xt, device))

    result = {"model": "tfidf_mlp_v1", "trained_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
              "seed": args.seed, "train_size": len(parts["train"]), "val_size": len(parts["val"]),
              "test_synth_size": len(parts["test_synth"]), "real_test_size": 0 if real is None else len(real),
              "architecture": f"TF-IDF char 2-5-gram ({info['features']}) -> Linear({info['features']},{args.hidden}) "
                              f"-> ReLU -> Dropout({args.dropout}) -> Linear({args.hidden},{len(labels)})",
              "hyperparameters": {"lr": args.lr, "weight_decay": args.weight_decay, "batch": args.batch,
                                  "epochs_max": args.epochs, "patience": args.patience, "hidden": args.hidden,
                                  "dropout": args.dropout, "class_weight_clip": 10.0},
              "n_params": n_params, "best_epoch": best_epoch, "epochs_run": len(history), "device": str(device),
              "smoke": bool(args.smoke), "export": info, "history": history, "metrics": {}, "baseline": {}}

    texts_syn = parts["test_synth"]["text"].tolist()
    p_syn = prod.predict_proba(texts_syn)
    result["metrics"]["val"] = eu.split_metrics(y["val"], prod.predict_proba(parts["val"]["text"].tolist()).argmax(1), labels)
    result["metrics"]["test_synth"] = eu.split_metrics(y["test_synth"], p_syn.argmax(1), labels)
    result["baseline"] = {"name": f"tfidf_char_logreg_{args.features // 1000}k",
                          "test_synth": eu.split_metrics(y["test_synth"], baseline_probs(vec, clf, texts_syn, len(labels)).argmax(1), labels)}
    t1 = time.time()
    prod.predict_proba(texts_syn[:50])
    result["latency_ms_50_items"] = round((time.time() - t1) * 1000, 1)
    tp = torch_probs(texts_syn[:200])
    result["numpy_vs_torch"] = {"max_abs_diff": float(np.abs(tp - prod.predict_proba(texts_syn[:200])).max()),
                                "argmax_agreement": float((tp.argmax(1) == prod.predict_proba(texts_syn[:200]).argmax(1)).mean())}

    choice = {"classifier": "tfidf_logreg_baseline", "reason": "real test missing", "threshold": eu.THRESHOLD,
              "unknown": eu.UNKNOWN}
    if real is not None and len(real):
        keys = eu.seed_dictionary_keys()
        y_real = real["subcategory"].map(l2i).to_numpy(dtype=np.int64)
        texts_real = real["text"].tolist()
        pr = prod.predict_proba(texts_real)
        views = eu.real_test_views(real, y_real, pr.argmax(1), labels, keys)
        result["metrics"]["real_test"] = views["all_items"]
        result["metrics"]["real_test_views"] = views
        thr = eu.apply_threshold(pr, labels)
        result["metrics"]["real_test_with_threshold_0.5"] = {
            **eu.split_metrics(y_real, thr, labels), "share_routed_to_unknown": float((thr == l2i[eu.UNKNOWN]).mean())}
        result["per_class_real_test"] = eu.per_class_report(y_real, pr.argmax(1), labels)
        bp = baseline_probs(vec, clf, texts_real, len(labels))
        result["baseline"]["real_test"] = eu.split_metrics(y_real, bp.argmax(1), labels)
        result["baseline"]["real_test_views"] = eu.real_test_views(real, y_real, bp.argmax(1), labels, keys)
        eu.write_errors_md(REPORTS / "errors_examples_mlp.md", real, y_real, pr.argmax(1), pr, labels, "TF-IDF MLP")
        name, reason = eu.choose_production(result["metrics"]["real_test"], result["baseline"]["real_test"])
        kind = "mlp" if name == "char_cnn" else "tfidf_logreg_baseline"      # choose_production names the network "char_cnn"
        choice = {"classifier": kind, "reason": reason.replace("cnn", "mlp"), "threshold": eu.THRESHOLD, "unknown": eu.UNKNOWN}
        result["production_choice"], result["production_choice_reason"] = choice["classifier"], choice["reason"]
    (MODELS_OUT / "mlp_choice.json").write_text(json.dumps(choice, ensure_ascii=False, indent=1), encoding="utf-8")

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / ("metrics_mlp_smoke.json" if args.smoke else "metrics_mlp.json")
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[mlp] wrote {out}")
    for name in ("test_synth", "real_test"):
        if name in result["metrics"]:
            print(f"[mlp] {name:10s}", {k: round(v, 4) for k, v in result["metrics"][name].items() if isinstance(v, float)})
            print(f"[base] {name:10s}", {k: round(v, 4) for k, v in result["baseline"][name].items() if isinstance(v, float)})
    print("[mlp] numpy vs torch:", result["numpy_vs_torch"], " size MB:", info["size_mb"])
    print("[mlp] production choice:", choice["classifier"], "|", choice["reason"])


if __name__ == "__main__":
    main()
