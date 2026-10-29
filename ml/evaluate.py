"""Stage 2 checks: ONNX output == PyTorch output (max |diff| < 1e-4 on 100 examples), size and latency.

  python -m ml.evaluate classifier
  python -m ml.evaluate forecaster

The numbers are written into ml/reports/metrics_*.json (onnx_size_mb, latency_ms_50_items, parity).
`--no-torch` skips the parity check and only measures the ONNX file (for machines without torch).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.normalize import encode_batch  # noqa: E402

MODELS_OUT = ROOT / "ml" / "models_out"
REPORTS = ROOT / "ml" / "reports"
PROCESSED = ROOT / "ml" / "data" / "processed"
PARITY_TOL = 1e-4


def _session(path: Path):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


def _latency_ms(fn, repeats: int = 30) -> float:
    fn()  # warm-up
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    return float(statistics.median(times))


def _update_report(name: str, **fields) -> None:
    p = REPORTS / name
    data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    data.update(fields)
    REPORTS.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def sample_texts(n: int) -> list:
    from ml import eval_utils as eu

    labels = eu.load_labels()
    real = eu.load_real_test(PROCESSED / "real_test_labeled.csv", labels)
    if real is not None and len(real) >= n:
        return real["text"].tolist()[:n]
    df = eu.load_dataset(PROCESSED / "category_dataset.parquet")
    return df["text"].tolist()[:n]


def check_classifier(no_torch: bool) -> dict:
    onnx_path = MODELS_OUT / "classifier.onnx"
    sess = _session(onnx_path)
    texts = sample_texts(100)
    ids = encode_batch(texts)
    onnx_logits = sess.run(["logits"], {"input": ids})[0]
    out = {"onnx_size_mb": round(onnx_path.stat().st_size / 1e6, 3)}
    batch50 = encode_batch(sample_texts(50))
    out["latency_ms_50_items"] = round(_latency_ms(lambda: sess.run(["logits"], {"input": batch50})), 2)
    if not no_torch:
        import torch
        from ml.models import CharCNN

        labels = json.loads((MODELS_OUT / "labels.json").read_text(encoding="utf-8"))
        model = CharCNN(len(labels))
        model.load_state_dict(torch.load(MODELS_OUT / "classifier.pt", map_location="cpu"))
        model.eval()
        with torch.no_grad():
            ref = model(torch.from_numpy(ids)).numpy()
        diff = float(np.abs(ref - onnx_logits).max())
        out.update({"onnx_parity_max_abs_diff": diff, "onnx_parity_ok": diff < PARITY_TOL,
                    "onnx_argmax_agreement": float((ref.argmax(1) == onnx_logits.argmax(1)).mean())})
    return out


def check_forecaster(no_torch: bool) -> dict:
    onnx_path = MODELS_OUT / "forecaster.onnx"
    sess = _session(onnx_path)
    rng = np.random.default_rng(0)
    from app.services.forecast_core import N_FEATURES, WINDOW

    x = rng.normal(0, 1, (100, WINDOW, N_FEATURES)).astype(np.float32)
    rem = rng.uniform(0, 1, (100, 1)).astype(np.float32)
    onnx_out = sess.run(["log_rest"], {"x": x, "rem": rem})[0]
    out = {"onnx_size_mb": round(onnx_path.stat().st_size / 1e6, 3),
           "latency_ms_1_item": round(_latency_ms(lambda: sess.run(["log_rest"], {"x": x[:1], "rem": rem[:1]})), 2)}
    if not no_torch:
        import torch
        from ml.models import ForecastLSTM

        model = ForecastLSTM()
        model.load_state_dict(torch.load(MODELS_OUT / "forecaster.pt", map_location="cpu"))
        model.eval()
        with torch.no_grad():
            ref = model(torch.from_numpy(x), torch.from_numpy(rem)).numpy()
        diff = float(np.abs(ref - onnx_out).max())
        out.update({"onnx_parity_max_abs_diff": diff, "onnx_parity_ok": diff < PARITY_TOL})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["classifier", "forecaster"])
    ap.add_argument("--no-torch", action="store_true")
    args = ap.parse_args()
    res = check_classifier(args.no_torch) if args.what == "classifier" else check_forecaster(args.no_torch)
    print(json.dumps(res, indent=2))
    _update_report(f"metrics_{args.what}.json", **res)
    if res.get("onnx_parity_ok") is False:
        raise SystemExit(f"ONNX output differs from PyTorch by more than {PARITY_TOL}")


if __name__ == "__main__":
    main()
