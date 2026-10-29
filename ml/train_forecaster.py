"""Stage 2: train the LSTM forecaster on SYNTHETIC history and compare it with two baselines (guide 7.4).

  python -m ml.train_forecaster [--smoke]

Needs ml/data/processed/transactions_synth.parquet (run ml.scripts.05_generate_transactions first).
Writes:
  ml/models_out/forecaster.pt, ml/models_out/scaler.json
  ml/reports/metrics_forecaster.json, ml/reports/forecast_vs_baseline.png

The users are split 70/15/15 BEFORE any sample is built, so a user never appears in two parts.
All numbers describe synthetic data: they show that the pipeline works and how the methods compare on it,
not how accurate the forecast is for real households.
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
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.forecast_core import (HISTORY_DAYS, N_FEATURES, SCALE_FLOOR, WINDOW, decode_output)  # noqa: E402
from ml.forecast_data import (EVAL_DAYS, TRAIN_DAY_STEP, build_samples, build_series, choose_method,  # noqa: E402
                              evaluate_methods, interval_quantiles)
from ml.tableio import read_table  # noqa: E402
from ml.transactions import split_users  # noqa: E402

PROCESSED = ROOT / "ml" / "data" / "processed"
REPORTS = ROOT / "ml" / "reports"
MODELS_OUT = ROOT / "ml" / "models_out"
FEATURE_NAMES = (["log_daily"] + [f"dow_{i}" for i in range(7)] + ["dom_sin", "dom_cos", "payday", "log_month_to_date"])


def lstm_predict(model, S, device, batch: int = 1024) -> np.ndarray:
    """Remaining spending in roubles for every sample."""
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(S.y), batch):
            x = torch.from_numpy(S.x[i:i + batch]).to(device)
            r = torch.from_numpy(S.rem[i:i + batch]).to(device)
            out.append(model(x, r).squeeze(1).float().cpu().numpy())
    y = np.concatenate(out) if out else np.zeros(0)
    return np.array([decode_output(v, s) for v, s in zip(y, S.scale)])


def train_lstm(Str, Sva, args, device):
    import torch
    import torch.nn.functional as F
    from ml.models import ForecastLSTM, count_params

    model = ForecastLSTM().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    X, R, Y = (torch.from_numpy(a).to(device) for a in (Str.x, Str.rem, Str.y))
    Xv, Rv, Yv = (torch.from_numpy(a).to(device) for a in (Sva.x, Sva.rem, Sva.y))
    rng = np.random.default_rng(args.seed)
    best, best_state, best_epoch, bad, history = 1e9, None, 0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.from_numpy(rng.permutation(len(Y))).to(device)
        total, t0 = 0.0, time.time()
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad()
            loss = F.huber_loss(model(X[idx], R[idx]).squeeze(1), Y[idx], delta=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss) * len(idx)
        model.eval()
        with torch.no_grad():
            v = float(F.huber_loss(model(Xv, Rv).squeeze(1), Yv, delta=1.0))
        history.append({"epoch": epoch, "train_huber": total / len(Y), "val_huber": v, "seconds": round(time.time() - t0, 1)})
        print(f"[lstm] epoch {epoch:2d}  train {total / len(Y):.4f}  val {v:.4f}  {history[-1]['seconds']}s")
        if v < best - 1e-5:
            best, best_epoch, bad = v, epoch, 0
            best_state = copy.deepcopy({k: t.detach().cpu() for k, t in model.state_dict().items()})
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[lstm] early stopping (no improvement for {args.patience} epochs)")
                break
    model.load_state_dict(best_state)
    return model, history, best_epoch, count_params(model), best


def plot_results(S, preds, results, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(preds)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    days = [str(d) for d in EVAL_DAYS if all(str(d) in results[n]["by_day"] for n in names)]
    width = 0.8 / max(len(names), 1)
    for k, n in enumerate(names):
        axes[0].bar(np.arange(len(days)) + k * width, [results[n]["by_day"][d]["mae"] for d in days], width, label=n)
        axes[1].bar(np.arange(len(days)) + k * width, [100 * results[n]["by_day"][d]["mape"] for d in days], width, label=n)
    for ax, title in zip(axes[:2], ["MAE of the month total, roubles", "MAPE of the month total, %"]):
        ax.set_xticks(np.arange(len(days)) + width * (len(names) - 1) / 2, [f"day {d}" for d in days])
        ax.set_title(title)
        ax.legend()
    m = S.day == 15
    true_total = (S.spent + S.rest)[m]
    for n in names:
        axes[2].scatter(true_total, (S.spent + preds[n])[m], s=7, alpha=0.5, label=n)
    lim = [0, float(true_total.max()) * 1.05] if m.any() else [0, 1]
    axes[2].plot(lim, lim, "k--", lw=1)
    axes[2].set_xlabel("true month total")
    axes[2].set_ylabel("predicted month total (forecast made on day 15)")
    axes[2].set_title("Test users, day 15 (synthetic data)")
    axes[2].legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tx", default=str(PROCESSED / "transactions_synth.parquet"))
    ap.add_argument("--users", default=str(PROCESSED / "users_synth.csv"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    import torch
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[forecast] device: {device}")

    tx = read_table(args.tx)
    tx["date"] = pd.to_datetime(tx["date"])
    users = pd.read_csv(args.users)
    splits = split_users(users["user_id"].tolist(), args.seed)
    if args.smoke:
        splits = {k: v[:20] for k, v in splits.items()}
        args.epochs = 1
    print({k: len(v) for k, v in splits.items()}, "users per split")

    train_days = list(range(TRAIN_DAY_STEP, 28, TRAIN_DAY_STEP))
    t0 = time.time()
    ser = {k: build_series(tx, users, user_ids=v, seed=args.seed + i) for i, (k, v) in enumerate(splits.items())}
    Str = build_samples(ser["train"], train_days)
    Sva = build_samples(ser["val"], train_days)
    Sva_eval = build_samples(ser["val"], EVAL_DAYS)
    Ste = build_samples(ser["test"], EVAL_DAYS)
    print(f"[forecast] samples: train {len(Str.y)}, val {len(Sva.y)}, test {len(Ste.y)} ({time.time() - t0:.0f}s)")

    model, history, best_epoch, n_params, best_val = train_lstm(Str, Sva, args, device)
    MODELS_OUT.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, MODELS_OUT / "forecaster.pt")

    preds = {"baseline_avg": Ste.b1, "baseline_dow": Ste.b2, "lstm": lstm_predict(model, Ste, device)}
    results = evaluate_methods(Ste, preds)
    method, why = choose_method(results)
    q = interval_quantiles(Sva_eval, lstm_predict(model, Sva_eval, device))
    q_base = {n: interval_quantiles(Sva_eval, p) for n, p in (("baseline_avg", Sva_eval.b1), ("baseline_dow", Sva_eval.b2))}
    print("[forecast] test MAE of the month total (roubles) / MAPE:")
    for n, r in results.items():
        print(f"   {n:13s} MAE {r['overall']['mae']:9.0f}   MAPE {r['overall']['mape']:.3f}")
    print(f"[forecast] production method: {method} ({why})")

    report = {
        "model": "lstm_v1", "synthetic_data": True,
        "trained_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "seed": args.seed,
        "users": {k: len(v) for k, v in splits.items()},
        "samples": {"train": len(Str.y), "val": len(Sva.y), "test": len(Ste.y)},
        "eval_days": list(EVAL_DAYS), "n_params": n_params, "best_epoch": best_epoch, "best_val_huber": best_val,
        "results_test": results, "production_method": method, "production_reason": why,
        "interval": {"lstm_val": q, "baselines_val": q_base},
        "baselines": {"baseline_avg": "B1: mean daily spending of the last 28 days x remaining days",
                      "baseline_dow": "B2: day-of-week means of non-regular spending (56 days) over the remaining days "
                                      "+ detected regular payments (rent, utilities, phone, subscriptions) not yet paid"},
        "caveats": ["All data is synthetic (data/personas.yaml); the generator contains the same regularities the "
                    "model can learn (weekday profile, paydays, regular payments, seasonality).",
                    "Real households are noisier; the comparison of methods on real data is future work."],
        "smoke": bool(args.smoke), "history": history,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / ("metrics_forecaster_smoke.json" if args.smoke else "metrics_forecaster.json")).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    scaler = {"window": WINDOW, "history_days": HISTORY_DAYS, "n_features": N_FEATURES, "feature_names": FEATURE_NAMES,
              "scale_floor": SCALE_FLOOR, "scale_rule": "median of the positive daily totals in the last 56 days, >= scale_floor",
              "rem_norm": 31, "target": "log1p(remaining / scale)", "min_history_days_lstm": WINDOW,
              "min_history_days_baseline": 28, "interval": {"q_lo": q["q_lo"], "q_hi": q["q_hi"]},
              "interval_baseline_avg": q_base["baseline_avg"], "interval_baseline_dow": q_base["baseline_dow"],
              "production_method": method, "payday_flag": "0 when the user's pay days are unknown"}
    (MODELS_OUT / "scaler.json").write_text(json.dumps(scaler, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_results(Ste, preds, results, REPORTS / "forecast_vs_baseline.png")
    print("[forecast] wrote metrics_forecaster.json, forecast_vs_baseline.png, scaler.json")


if __name__ == "__main__":
    main()
