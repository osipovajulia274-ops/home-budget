"""Export the trained PyTorch models to ONNX (opset 17) and write the files production needs.

  python -m ml.export_onnx                 # ml/models_out/*.pt -> ml/models_out/*.onnx (+ vocab.json, labels.json)
  python -m ml.export_onnx --install       # also copy the artifacts to app/ml_artifacts/ (must stay < 10 MB)

Production loads only: classifier.onnx, vocab.json, labels.json, baseline.npz, baseline_vocab.json, model_choice.json
(model_choice.json says which of the char-CNN / baseline / ensemble the app uses). The LSTM forecaster is an optional
experiment and is never installed.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.forecast_core import N_FEATURES, WINDOW  # noqa: E402
from app.services.normalize import MAX_LEN, vocab_dict  # noqa: E402

MODELS_OUT = ROOT / "ml" / "models_out"
ARTIFACTS = ROOT / "app" / "ml_artifacts"
MAX_TOTAL_MB = 10.0
FILES = ("classifier.onnx", "vocab.json", "labels.json", "baseline.npz", "baseline_vocab.json", "model_choice.json")


def _export(model, args, path: Path, **kwargs) -> None:
    import torch

    try:                                   # torch >= 2.5: choose the stable TorchScript-based exporter
        torch.onnx.export(model, args, str(path), dynamo=False, **kwargs)
    except TypeError:                      # older torch has no `dynamo` argument
        torch.onnx.export(model, args, str(path), **kwargs)


def export_classifier(src: Path = MODELS_OUT) -> Path:
    import torch
    from ml.models import CharCNN

    labels = json.loads((src / "labels.json").read_text(encoding="utf-8"))
    model = CharCNN(len(labels))
    model.load_state_dict(torch.load(src / "classifier.pt", map_location="cpu"))
    model.eval()
    out = src / "classifier.onnx"
    _export(model, (torch.zeros(2, MAX_LEN, dtype=torch.long),), out, input_names=["input"], output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
    (src / "vocab.json").write_text(json.dumps(vocab_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def export_forecaster(src: Path = MODELS_OUT) -> Path:
    import torch
    from ml.models import ForecastLSTM

    model = ForecastLSTM()
    model.load_state_dict(torch.load(src / "forecaster.pt", map_location="cpu"))
    model.eval()
    out = src / "forecaster.onnx"
    _export(model, (torch.zeros(2, WINDOW, N_FEATURES), torch.zeros(2, 1)), out, input_names=["x", "rem"],
            output_names=["log_rest"], dynamic_axes={"x": {0: "batch"}, "rem": {0: "batch"}, "log_rest": {0: "batch"}},
            opset_version=17)
    return out


def install(src: Path = MODELS_OUT, dst: Path = ARTIFACTS) -> float:
    dst.mkdir(parents=True, exist_ok=True)
    total = 0.0
    for name in FILES:
        f = src / name
        if f.exists():
            shutil.copy2(f, dst / name)
            total += f.stat().st_size / 1e6
    print(f"[install] {dst}: {total:.2f} MB")
    if total >= MAX_TOTAL_MB:
        raise SystemExit(f"artifacts are {total:.1f} MB, the limit is {MAX_TOTAL_MB} MB")
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["classifier", "forecaster"], default=None)
    ap.add_argument("--install", action="store_true")
    args = ap.parse_args()
    if args.only in (None, "classifier") and (MODELS_OUT / "classifier.pt").exists():
        print("[export]", export_classifier(), f"{(MODELS_OUT / 'classifier.onnx').stat().st_size / 1e6:.2f} MB")
    if args.only in (None, "forecaster") and (MODELS_OUT / "forecaster.pt").exists():
        print("[export]", export_forecaster(), f"{(MODELS_OUT / 'forecaster.onnx').stat().st_size / 1e6:.2f} MB")
    if args.install:
        install()


if __name__ == "__main__":
    main()
