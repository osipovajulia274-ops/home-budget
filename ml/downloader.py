"""Resumable download with integrity checks (standard library only).

Why: a download that the server cuts in the middle ends without an exception when the connection is closed
cleanly, and the old code then installed the truncated file (a Colab run stopped at 5432 of 8020 MB and DuckDB
failed with "No magic bytes found at end of file"). Now the size is compared with what the server announced, the
rest is requested with a Range header, and a finished Parquet file must start and end with the magic bytes.
"""
from __future__ import annotations

import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from http.client import HTTPException
from pathlib import Path
from typing import Optional, Tuple

UA = "home-budget-student-project/1.0"


def file_is_valid(path: Path) -> bool:
    """Cheap check: Parquet needs the PAR1 magic at both ends, gzip needs its two-byte header."""
    try:
        size = path.stat().st_size
        if size <= 0:
            return False
        with open(path, "rb") as f:
            head = f.read(4)
            if path.suffix == ".parquet":
                f.seek(max(size - 4, 0))
                return head == b"PAR1" and f.read(4) == b"PAR1"
            if path.suffix == ".gz":
                return head[:2] == b"\x1f\x8b"
        return True
    except OSError:
        return False


def _total_from(resp, done_before: int) -> Optional[int]:
    cr = resp.headers.get("Content-Range")
    if cr:
        m = re.match(r"bytes\s+\d+-\d+/(\d+)", cr)
        if m:
            return int(m.group(1))
    cl = resp.headers.get("Content-Length")
    if cl and cl.isdigit():
        return int(cl) + (done_before if resp.status == 206 else 0)
    return None


def download(url: str, dest: Path, retries: int = 12, timeout: int = 60, pause: float = 3.0,
             chunk: int = 1 << 20, quiet: bool = False) -> bool:
    dest = Path(dest)
    if dest.exists():
        if file_is_valid(dest):
            print(f"[01] already there: {dest}")
            return True
        print(f"[01] {dest.name} is incomplete or damaged - downloading again")
        dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    total: Optional[int] = None
    for attempt in range(1, retries + 1):
        done = tmp.stat().st_size if tmp.exists() else 0
        headers = {"User-Agent": UA}
        if done:
            headers["Range"] = f"bytes={done}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as resp:
                if done and resp.status != 206:           # the server ignored Range: start over
                    done = 0
                total = _total_from(resp, done) or total
                with open(tmp, "ab" if done else "wb") as f:
                    while True:
                        data = resp.read(chunk)
                        if not data:
                            break
                        f.write(data)
                        done += len(data)
                        if total and not quiet:
                            print(f"\r[01] {dest.name}: {done / 1e6:.0f}/{total / 1e6:.0f} MB", end="", flush=True)
            if not quiet:
                print()
        except (urllib.error.HTTPError,) as exc:
            if exc.code == 416 and total and tmp.exists() and tmp.stat().st_size >= total:
                pass                                       # everything was already downloaded
            elif exc.code in (404, 403, 401):
                print(f"[01] FAILED {url}\n     {exc}\n     -> record in PROGRESS.md, fall back to the seed dictionary",
                      file=sys.stderr)
                tmp.unlink(missing_ok=True)
                return False
            else:
                print(f"[01] attempt {attempt}/{retries}: {exc}", file=sys.stderr)
                time.sleep(pause)
                continue
        except (urllib.error.URLError, TimeoutError, OSError, HTTPException) as exc:
            print(f"\n[01] attempt {attempt}/{retries}: {exc}", file=sys.stderr)
            time.sleep(pause)
            continue
        size = tmp.stat().st_size if tmp.exists() else 0
        if total is None or size >= total:
            break
        print(f"[01] the server stopped at {size / 1e6:.0f} of {total / 1e6:.0f} MB - resuming "
              f"(attempt {attempt}/{retries})", file=sys.stderr)
        time.sleep(pause)
    else:
        print(f"[01] FAILED {url}: could not finish in {retries} attempts; the partial file stays as {tmp.name} "
              f"and the next run continues from it", file=sys.stderr)
        return False
    if total is not None and tmp.stat().st_size != total:
        print(f"[01] FAILED {url}: got {tmp.stat().st_size} bytes, expected {total}", file=sys.stderr)
        tmp.unlink(missing_ok=True)
        return False
    shutil.move(str(tmp), str(dest))
    if not file_is_valid(dest):
        print(f"[01] FAILED {url}: the file is damaged (bad magic bytes)", file=sys.stderr)
        dest.unlink(missing_ok=True)
        return False
    return True
