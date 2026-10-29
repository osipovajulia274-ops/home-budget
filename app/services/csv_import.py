"""Bank statement CSV -> expense rows (guide 8.4).

Real bank files were NOT available to check the column names, so the parser guesses them from synonyms and
always returns the guessed mapping; the preview screen lets the user correct it and call again with `mapping`.

  * encoding: utf-8-sig, then cp1251
  * delimiter: csv.Sniffer over `; , \\t` with a header-based fallback
  * negative amounts are expenses (absolute value is taken), positive ones are income and are not spending
  * transfers (перевод, пополнение, ... себе) are marked `other.transfers`
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

from app.services.normalize import normalize

FIELDS = ("date", "amount", "description", "category", "mcc")

# synonyms in priority order (compared after lowercase / ё->е / whitespace collapse)
SYNONYMS: Dict[str, List[str]] = {
    "date": ["дата операции", "дата платежа", "дата транзакции", "дата", "date", "время операции", "timestamp"],
    "amount": ["сумма операции", "сумма в валюте счета", "сумма", "amount", "сумма платежа", "sum"],
    "description": ["описание", "назначение платежа", "назначение", "merchant", "description", "наименование",
                    "контрагент", "получатель", "детали"],
    "category": ["категория", "category"],
    "mcc": ["mcc", "мсс"],
}
# every column matching one of these is joined into the description (merchant + description, ...)
DESCRIPTION_FIELDS = SYNONYMS["description"]

TRANSFER_RE = re.compile(
    r"перевод|пополнени(?!\w*\s+(?:мобильн|телефон|связи|баланса))|между своими|на свою карту|со своей карты|"
    r"внутренн\w+ перевод", re.I)

_DATE_FORMATS = ("%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M", "%d.%m.%Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
                 "%Y-%m-%d %H:%M", "%Y-%m-%d", "%d.%m.%y %H:%M", "%d.%m.%y", "%d/%m/%Y")

Mapping = Dict[str, Union[str, List[str], None]]


@dataclass
class CsvRow:
    date: dt.date
    title: str
    amount: float                      # positive: how much was spent
    bank_category: Optional[str] = None
    mcc: Optional[str] = None
    is_transfer: bool = False


@dataclass
class CsvParseResult:
    encoding: str = "utf-8-sig"
    delimiter: str = ";"
    columns: List[str] = field(default_factory=list)
    mapping: Mapping = field(default_factory=dict)
    rows: List[CsvRow] = field(default_factory=list)
    income_total: float = 0.0
    income_count: int = 0
    skipped: int = 0
    warnings: List[str] = field(default_factory=list)


class CsvImportError(ValueError):
    """The file cannot be read as a table of operations (the message is shown to the user)."""


def _norm_col(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower().replace("ё", "е"))


def decode(data: bytes) -> tuple[str, str]:
    for enc in ("utf-8-sig", "cp1251"):
        try:
            return data.decode(enc), enc
        except UnicodeDecodeError:
            continue
    raise CsvImportError("Не удалось определить кодировку файла (ожидается UTF-8 или Windows-1251)")


def sniff_delimiter(text: str) -> str:
    head = "\n".join(text.splitlines()[:20])
    try:
        return csv.Sniffer().sniff(head, delimiters=";,\t").delimiter
    except csv.Error:
        first = text.splitlines()[0] if text.strip() else ""
        return max(";,\t", key=first.count)


def detect_mapping(columns: Sequence[str]) -> Mapping:
    """Guess which column is which. `description` is a list (merchant + description are both useful)."""
    norm = {c: _norm_col(c) for c in columns}
    out: Mapping = {}
    for fld in ("date", "amount", "category", "mcc"):
        out[fld] = None
        for syn in SYNONYMS[fld]:                       # exact match first
            hit = next((c for c, n in norm.items() if n == syn), None)
            if hit:
                out[fld] = hit
                break
        if out[fld] is None:                            # then "starts with"
            for syn in SYNONYMS[fld]:
                hit = next((c for c, n in norm.items() if n.startswith(syn)), None)
                if hit:
                    out[fld] = hit
                    break
    taken = {v for v in out.values() if isinstance(v, str)}
    desc = [c for c, n in norm.items() if c not in taken and any(n == s or n.startswith(s) for s in DESCRIPTION_FIELDS)]
    # keep the synonyms' priority order (merchant before description is not required, just stable)
    desc.sort(key=lambda c: next(i for i, s in enumerate(DESCRIPTION_FIELDS) if norm[c] == s or norm[c].startswith(s)))
    out["description"] = desc
    return out


def parse_date(s: str) -> Optional[dt.date]:
    s = (s or "").strip()
    for fmt in _DATE_FORMATS:
        try:
            return dt.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def parse_amount(s: str) -> Optional[float]:
    s = (s or "").strip().replace("−", "-").replace(" ", "").replace(" ", "")
    s = re.sub(r"[^\d,.\-+]", "", s)
    if not s or s in "+-":
        return None
    if "," in s and "." in s:                           # 1,234.56 or 1.234,56
        s = s.replace(",", "") if s.rfind(".") > s.rfind(",") else s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _single(mapping: Mapping, key: str) -> Optional[str]:
    """The column name of a one-column field (date, amount, category, mcc)."""
    v = mapping.get(key)
    if isinstance(v, list):
        return v[0] if v else None
    return v or None


def _desc_columns(mapping: Mapping) -> List[str]:
    d = mapping.get("description")
    if not d:
        return []
    return [d] if isinstance(d, str) else list(d)


def parse_csv(data: bytes, mapping: Optional[Mapping] = None) -> CsvParseResult:
    text, enc = decode(data)
    if not text.strip():
        raise CsvImportError("Файл пустой")
    delim = sniff_delimiter(text)
    reader = csv.reader(io.StringIO(text), delimiter=delim)
    table = [r for r in reader if any(c.strip() for c in r)]
    if len(table) < 2:
        raise CsvImportError("В файле нет строк с операциями")
    header = [h.strip() for h in table[0]]
    res = CsvParseResult(encoding=enc, delimiter=delim, columns=header)
    res.mapping = detect_mapping(header)
    for k, v in (mapping or {}).items():                # the user's mapping wins over the guess
        if k in FIELDS and v not in (None, "", []):
            res.mapping[k] = v
    idx = {h: i for i, h in enumerate(header)}

    def col(row: List[str], name: Optional[str]) -> str:
        i = idx.get(name) if name else None
        return row[i].strip() if i is not None and i < len(row) else ""

    missing = [f for f in ("date", "amount") if not res.mapping.get(f) or res.mapping[f] not in idx]
    if not _desc_columns(res.mapping):
        missing.append("description")
    if missing:
        names = {"date": "дата", "amount": "сумма", "description": "описание"}
        raise CsvImportError("Не удалось определить колонки: " + ", ".join(names[m] for m in missing)
                             + ". Укажите их вручную.")
    for name in [res.mapping["date"], res.mapping["amount"], *_desc_columns(res.mapping)]:
        if name not in idx:
            raise CsvImportError(f"В файле нет колонки «{name}»")

    for row in table[1:]:
        d = parse_date(col(row, _single(res.mapping, "date")))
        a = parse_amount(col(row, _single(res.mapping, "amount")))
        if d is None or a is None:
            res.skipped += 1
            continue
        if a > 0:
            res.income_total = round(res.income_total + a, 2)
            res.income_count += 1
            continue
        if a == 0:
            res.skipped += 1
            continue
        parts = []
        for c in _desc_columns(res.mapping):
            v = col(row, c)
            if v and v.lower() not in (p.lower() for p in parts):
                parts.append(v)
        title = re.sub(r"\s+", " ", " ".join(parts)).strip()
        if not title:
            res.skipped += 1
            continue
        bank_cat = col(row, _single(res.mapping, "category")) or None
        mcc = col(row, _single(res.mapping, "mcc")) or None
        is_tr = bool(TRANSFER_RE.search(title)) or (bank_cat is not None and "перевод" in normalize(bank_cat))
        res.rows.append(CsvRow(date=d, title=title, amount=round(abs(a), 2), bank_category=bank_cat,
                               mcc=mcc, is_transfer=is_tr))
    if res.skipped:
        res.warnings.append(f"Пропущено строк, которые не удалось разобрать: {res.skipped}")
    if not res.rows:
        res.warnings.append("В файле не найдено ни одного расхода (отрицательных сумм)")
    return res
