import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from ml.downloader import download, file_is_valid

PAYLOAD = b"PAR1" + bytes(range(256)) * 400 + b"PAR1"


def make_server(cut_first: bool, support_range: bool = True):
    state = {"calls": 0}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            state["calls"] += 1
            start = 0
            rng = self.headers.get("Range")
            if rng and support_range:
                start = int(rng.split("=")[1].split("-")[0])
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}")
            else:
                self.send_response(200)
            body = PAYLOAD[start:]
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if cut_first and state["calls"] == 1:
                self.wfile.write(body[: len(body) // 2])          # announce everything, send half, close
                self.wfile.flush()
                self.close_connection = True
                return
            self.wfile.write(body)

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, state


@pytest.mark.parametrize("support_range", [True, False])
def test_truncated_response_is_resumed(tmp_path, support_range):
    srv, state = make_server(cut_first=True, support_range=support_range)
    try:
        dest = tmp_path / "x" / "food.parquet"
        assert download(f"http://127.0.0.1:{srv.server_port}/f", dest, pause=0, quiet=True)
        assert dest.read_bytes() == PAYLOAD and state["calls"] >= 2
        assert not dest.with_suffix(".parquet.part").exists()
    finally:
        srv.shutdown()


def test_damaged_existing_file_is_replaced(tmp_path):
    srv, _ = make_server(cut_first=False)
    try:
        dest = tmp_path / "food.parquet"
        dest.write_bytes(PAYLOAD[: len(PAYLOAD) // 2])                 # what the failed Colab run left behind
        assert not file_is_valid(dest)
        assert download(f"http://127.0.0.1:{srv.server_port}/f", dest, pause=0, quiet=True)
        assert dest.read_bytes() == PAYLOAD
    finally:
        srv.shutdown()


def test_valid_existing_file_is_kept(tmp_path):
    dest = tmp_path / "food.parquet"
    dest.write_bytes(PAYLOAD)
    assert download("http://127.0.0.1:1/never-called", dest, pause=0, quiet=True)


def test_gives_up_with_partial_file(tmp_path):
    assert not download("http://127.0.0.1:1/none", tmp_path / "a.csv.gz", retries=2, pause=0, quiet=True)
