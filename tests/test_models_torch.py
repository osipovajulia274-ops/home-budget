"""Model tests that need PyTorch. They are skipped where torch is not installed (the production image) and
run in Colab / any training environment: `pytest -q` in the first cells of colab_train.ipynb."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from app.services.forecast_core import N_FEATURES, WINDOW  # noqa: E402
from app.services.normalize import MAX_LEN, VOCAB_SIZE, encode_batch  # noqa: E402
from ml.models import CharCNN, ForecastLSTM, count_params  # noqa: E402


def test_charcnn_output_shape_and_size():
    m = CharCNN(48)
    out = m(torch.zeros(3, MAX_LEN, dtype=torch.long))
    assert out.shape == (3, 48)
    assert count_params(m) < 400_000          # keeps classifier.onnx far below 3 MB


def test_charcnn_can_overfit_a_tiny_batch():
    torch.manual_seed(0)
    texts = ["молоко 3,2%", "хлеб белый", "такси поездка", "бензин ай-95", "стрижка мужская", "билет в кино"]
    x = torch.from_numpy(encode_batch(texts))
    y = torch.arange(len(texts))
    m = CharCNN(len(texts))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(80):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(m(x), y)
        loss.backward()
        opt.step()
    m.eval()
    assert (m(x).argmax(1) == y).all()


def test_lstm_output_shape():
    m = ForecastLSTM()
    assert m(torch.zeros(4, WINDOW, N_FEATURES), torch.zeros(4, 1)).shape == (4, 1)


def test_vocab_matches_embedding():
    ids = torch.from_numpy(encode_batch(["Молоко 3,2% ПАСТ."]))
    assert int(ids.max()) < VOCAB_SIZE


def test_onnx_matches_torch_for_both_models(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    from ml.export_onnx import _export

    torch.manual_seed(1)
    cls = CharCNN(48).eval()
    ids = torch.from_numpy(encode_batch(["Молоко ПАСТ. 3,2% 900Г", "TAXI Москва", "", "Хлеб"] * 25))
    _export(cls, (ids[:2],), tmp_path / "c.onnx", input_names=["input"], output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
    s = ort.InferenceSession(str(tmp_path / "c.onnx"), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = cls(ids).numpy()
    got = s.run(["logits"], {"input": ids.numpy()})[0]
    assert np.abs(ref - got).max() < 1e-4
    assert s.run(["logits"], {"input": ids[:1].numpy()})[0].shape == (1, 48)       # dynamic batch

    lstm = ForecastLSTM().eval()
    x, r = torch.randn(100, WINDOW, N_FEATURES), torch.rand(100, 1)
    _export(lstm, (x[:2], r[:2]), tmp_path / "f.onnx", input_names=["x", "rem"], output_names=["log_rest"],
            dynamic_axes={"x": {0: "batch"}, "rem": {0: "batch"}, "log_rest": {0: "batch"}}, opset_version=17)
    s2 = ort.InferenceSession(str(tmp_path / "f.onnx"), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref2 = lstm(x, r).numpy()
    got2 = s2.run(["log_rest"], {"x": x.numpy(), "rem": r.numpy()})[0]
    assert np.abs(ref2 - got2).max() < 1e-4
    # production calls the forecaster with ONE sample, the export used batch 2 (torch warns about LSTM + dynamic batch)
    for n in (1, 7):
        part = s2.run(["log_rest"], {"x": x[:n].numpy(), "rem": r[:n].numpy()})[0]
        assert part.shape == (n, 1) and np.abs(part - ref2[:n]).max() < 1e-4
