"""Try the parser and the classifier on your own files, without the web server.

    python -m app.cli receipt path/to/receipt.txt
    python -m app.cli csv path/to/statement.csv

Nothing is saved or sent anywhere; the files are read locally.
"""
from __future__ import annotations

import sys

from app.services.categorizer import get_categorizer
from app.services.csv_import import CsvImportError
from app.services.importing import csv_preview, receipt_preview


def _table(items, with_date=False):
    for i in items:
        flag = "  ← проверить" if i["needs_review"] else ""
        d = f"{i['date']}  " if with_date else ""
        print(f"{d}{i['title'][:38]:38s} {i['amount']:10.2f}  {i['subcategory']:26s} {i['confidence']:.2f} "
              f"{i['category_source']}{flag}")


def main(argv) -> int:
    if len(argv) != 3 or argv[1] not in ("receipt", "csv"):
        print(__doc__)
        return 2
    cat = get_categorizer()
    if argv[1] == "receipt":
        text = open(argv[2], "rb").read().decode("utf-8", errors="replace")
        p = receipt_preview(text, cat)
        print(f"Магазин: {p['store']}   Дата: {p['purchased_at']}   ИТОГ: {p['total']}   Сумма позиций: {p['items_sum']}")
        _table(p["items"])
        ok = {True: "совпала ✓", False: "НЕ совпала ✗", None: "нечем сверить"}[p["sum_matches_total"]]
        print(f"\nСумма позиций {ok}")
        for w in p["warnings"]:
            print("!", w)
    else:
        try:
            p = csv_preview(open(argv[2], "rb").read(), cat)
        except CsvImportError as e:
            print("Ошибка:", e)
            return 1
        print(f"Кодировка {p['encoding']}, разделитель {p['delimiter']!r}, колонки: {p['mapping']}")
        _table(p["items"], with_date=True)
        print(f"\nТраты (без переводов): {p['items_sum']:.2f}; поступления: {p['income_total']:.2f}")
        for w in p["warnings"]:
            print("!", w)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
