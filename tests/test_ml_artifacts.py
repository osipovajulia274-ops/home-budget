"""Checks of the files that production loads. Skipped until the models have been trained and installed
into app/ml_artifacts/ (ml.export_onnx --install)."""
import json
from pathlib import Path

import numpy as np
import pytest

ART = Path(__file__).resolve().parents[1] / "app" / "ml_artifacts"
pytestmark = pytest.mark.skipif(not (ART / "classifier.onnx").exists(), reason="models not installed yet")


def test_total_size_under_10mb():
    assert sum(f.stat().st_size for f in ART.iterdir() if f.is_file()) / 1e6 < 10


def test_vocab_is_the_one_from_normalize():
    from app.services.normalize import vocab_dict

    assert json.loads((ART / "vocab.json").read_text(encoding="utf-8")) == json.loads(json.dumps(vocab_dict()))


def test_labels_match_taxonomy_order():
    from ml.eval_utils import load_labels

    assert json.loads((ART / "labels.json").read_text(encoding="utf-8")) == load_labels()


def test_classifier_runs_and_output_matches_labels():
    ort = pytest.importorskip("onnxruntime")
    from app.services.normalize import encode_batch

    s = ort.InferenceSession(str(ART / "classifier.onnx"), providers=["CPUExecutionProvider"])
    labels = json.loads((ART / "labels.json").read_text(encoding="utf-8"))
    out = s.run(["logits"], {"input": encode_batch(["Молоко ПАСТ. 3,2% 900Г", "Яндекс Такси поездка"])})[0]
    assert out.shape == (2, len(labels)) and np.isfinite(out).all()
    assert labels[int(out[0].argmax())].startswith("food.")
    assert labels[int(out[1].argmax())] == "transport.taxi"


def test_baseline_files_run_without_sklearn():
    if not (ART / "baseline.npz").exists():
        pytest.skip("no baseline")
    from app.services.linear_classifier import LinearTfidfClassifier

    labels = json.loads((ART / "labels.json").read_text(encoding="utf-8"))
    p = LinearTfidfClassifier.load(ART).predict_proba(["Молоко ПАСТ. 3,2% 900Г", "Яндекс Такси поездка"])
    assert p.shape == (2, len(labels)) and np.allclose(p.sum(1), 1.0, atol=1e-3)
    assert labels[int(p[0].argmax())].startswith("food.")


def test_model_choice_is_valid():
    if not (ART / "model_choice.json").exists():
        pytest.skip("no model_choice.json")
    c = json.loads((ART / "model_choice.json").read_text(encoding="utf-8"))
    assert c["classifier"] in {"char_cnn", "tfidf_logreg_baseline", "ensemble_cnn_baseline"}
    assert 0 < c["threshold"] < 1
