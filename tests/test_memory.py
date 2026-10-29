"""Guide 8.6: memory while importing a 50-item receipt stays far below the 250 MB budget.

This measures the process that runs the pure services (parser + classifier + analytics); the web framework and
the database driver add to it on Render, which is why the full-app figure still has to be taken from the Render
dashboard (guide 11.3).
"""
import re

import pytest

from app.services.categorizer import Categorizer
from app.services.importing import receipt_preview


def rss_mb():
    try:
        text = open("/proc/self/status").read()
    except OSError:
        pytest.skip("no /proc on this platform")
    return int(re.search(r"VmRSS:\s+(\d+) kB", text).group(1)) / 1024


def test_import_of_50_items_fits_in_memory_budget():
    cat = Categorizer.load()
    lines = ["ООО Тест", "-----"] + [f"МОЛОКО ПАСТ. 3,2% 900МЛ N{i}  1 x {50 + i}.00 ={50 + i}.00" for i in range(50)]
    total = sum(50 + i for i in range(50))
    text = "\n".join(lines + ["-----", f"ИТОГ {total}.00"])
    p = receipt_preview(text, cat)
    assert len(p["items"]) == 50 and p["sum_matches_total"] is True
    assert rss_mb() < 250
