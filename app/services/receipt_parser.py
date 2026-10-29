"""Receipt text -> items (guide 8.3). Rules and regular expressions only, no ML.

Recognised item layouts (comma or dot as the decimal separator):
  1. `NAME  QTY x PRICE =SUM`  /  `NAME QTY * PRICE = SUM`
  2. `NAME` on one line, `QTY x PRICE = SUM` (or just `SUM`) on the next one
  3. `NAME ...... SUM`  (a single price at the end of the line)
  4. weighed goods: `0.448 кг x 289.90 = 129.88` under the name
Service lines (ИТОГ, НАЛИЧНЫМИ, НДС, ККТ, ФН, ИНН, КАССИР ...) are ignored; a `СКИДКА` line with a negative
amount is subtracted from the previous item.

Privacy: the INN and the name of a sole proprietor are never kept; `ИП Сидоров А.` becomes `ИП С`.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from typing import List, Optional

MONEY = r"\d+[.,]\d{2}"
QTY = r"\d+(?:[.,]\d+)?"
UNIT = r"(?:кг|шт|л|г|м|уп)\.?"

_SEPARATOR = re.compile(r"^\s*[-=_*~]{4,}\s*$")
_TOTAL = re.compile(rf"^\s*итого?\b[^\d\-]*(?P<sum>{MONEY})", re.I)
_SERVICE = re.compile(
    r"^\s*(?:итого?|наличными|безналичными|электронными|сдача|ндс|ккт|фн|фд|фпд|инн|сно|кассир|смена|"
    r"приход|расход|сайт\s+фнс|кассовый\s+чек|чек\b|дата|оплата|номер|рн\s+ккт|зн\s+ккт|адрес|тел)\b",
    re.I,
)
_DISCOUNT = re.compile(rf"^\s*(?:скидк\w*|акци\w*)[^\d\-]*(?P<sum>-\s*{MONEY})\s*(?:₽|руб\.?|р\.?)?\s*$", re.I)
_DATE = re.compile(r"(?P<d>\d{1,2})\.(?P<m>\d{1,2})\.(?P<y>\d{4}|\d{2})(?:\s+(?P<h>\d{1,2}):(?P<mi>\d{2}))?")
_STORE = re.compile(r"\b(?:ООО|АО|ПАО|ЗАО|ОАО|ИП)\b")
_IP = re.compile(r"^\s*ИП\s+(?P<sur>[А-ЯЁA-Z])[\w\-]*", re.I)

# NAME QTY x PRICE [= SUM]
_P_QTY_PRICE = re.compile(
    rf"^(?P<name>.*?[A-Za-zА-Яа-яЁё].*?)\s+(?P<qty>{QTY})\s*(?:{UNIT})?\s*[xх×*]\s*(?P<price>{MONEY})"
    rf"\s*(?:=\s*)?(?P<sum>{MONEY})?\s*(?:₽|руб\.?|р\.?)?\s*$", re.I)
# QTY [unit] x PRICE [= SUM]   (continuation line, or a weighed item whose name is on the line above)
_P_CONT = re.compile(
    rf"^(?P<qty>{QTY})\s*(?:{UNIT})?\s*[xх×*]\s*(?P<price>{MONEY})\s*(?:=\s*)?(?P<sum>{MONEY})?\s*(?:₽|руб\.?|р\.?)?\s*$",
    re.I)
_P_SUM_ONLY = re.compile(rf"^=?\s*(?P<sum>{MONEY})\s*(?:₽|руб\.?|р\.?)?$")
# NAME ..... SUM
_P_NAME_SUM = re.compile(
    rf"^(?P<name>.*?[A-Za-zА-Яа-яЁё].*?)[\s.…_]+(?P<sum>{MONEY})\s*(?:₽|руб\.?|р\.?)?\s*$")


@dataclass
class ParsedItem:
    title: str
    qty: float
    price: float
    amount: float


@dataclass
class ParsedReceipt:
    store: Optional[str] = None
    purchased_at: Optional[dt.datetime] = None
    total: Optional[float] = None
    items: List[ParsedItem] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def items_sum(self) -> float:
        return round(sum(i.amount for i in self.items), 2)

    @property
    def sum_matches_total(self) -> Optional[bool]:
        if self.total is None or not self.items:
            return None
        return abs(self.items_sum - self.total) <= 1.0


def _num(s: str) -> float:
    return float(s.replace(",", ".").replace(" ", ""))


def _clean_title(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip(" \t…_")


def _anonymize_store(line: str) -> str:
    """Drop the INN and a sole proprietor's name, keep the first letter of the surname."""
    line = re.sub(r"\bИНН\b\s*:?\s*\d+", "", line, flags=re.I).strip()
    m = _IP.match(line)
    if m:
        return f"ИП {m.group('sur').upper()}"
    return _clean_title(line)


def _parse_date(lines: List[str], default: Optional[dt.date]) -> Optional[dt.datetime]:
    order = [l for l in lines if re.search(r"дата", l, re.I)] + lines
    for line in order:
        m = _DATE.search(line)
        if not m:
            continue
        y = int(m["y"]) + (2000 if len(m["y"]) == 2 else 0)
        try:
            return dt.datetime(y, int(m["m"]), int(m["d"]), int(m["h"] or 0), int(m["mi"] or 0))
        except ValueError:
            continue
    if default is not None:
        return dt.datetime(default.year, default.month, default.day)
    return None


def _item(name: str, qty: float, price: float, total: Optional[float]) -> ParsedItem:
    amount = round(total if total is not None else qty * price, 2)
    return ParsedItem(_clean_title(name), qty, price, amount)


def parse_receipt(text: str, default_date: Optional[dt.date] = None) -> ParsedReceipt:
    out = ParsedReceipt()
    lines = [l.rstrip() for l in text.replace("\r", "").split("\n")]
    nonempty = [l for l in lines if l.strip()]

    # --- header ---
    for l in nonempty:
        if _STORE.search(l) and not _SERVICE.match(l):
            out.store = _anonymize_store(l)
            break
    out.purchased_at = _parse_date(nonempty, default_date)
    for l in nonempty:
        m = _TOTAL.match(l)
        if m:
            out.total = _num(m["sum"])
            break

    # --- which lines are items: between the first separator and the next separator / ИТОГ ---
    seps = [i for i, l in enumerate(lines) if _SEPARATOR.match(l)]
    start = seps[0] + 1 if seps else 0
    end = len(lines)
    for i in range(start, len(lines)):
        if _SEPARATOR.match(lines[i]) or _TOTAL.match(lines[i]):
            end = i
            break
    body = lines[start:end]

    pending: Optional[str] = None       # a name waiting for its price line

    def flush_pending():
        nonlocal pending
        if pending:
            out.warnings.append(f"Не найдена цена для строки: «{_clean_title(pending)}»")
            pending = None

    for raw in body:
        line = raw.strip()
        if not line:
            continue
        d = _DISCOUNT.match(line)
        if d:
            flush_pending()
            if out.items:
                prev = out.items[-1]
                prev.amount = round(prev.amount + _num(d["sum"].replace(" ", "")), 2)
            else:
                out.warnings.append("Скидка без позиции выше — пропущена")
            continue
        if _SERVICE.match(line) or (not seps and _DATE.search(line) and not re.search(MONEY, line)):
            continue

        m = _P_QTY_PRICE.match(line)
        if m:
            flush_pending()
            out.items.append(_item(m["name"], _num(m["qty"]), _num(m["price"]),
                                   _num(m["sum"]) if m["sum"] else None))
            continue
        m = _P_CONT.match(line)
        if m:
            if pending:
                out.items.append(_item(pending, _num(m["qty"]), _num(m["price"]),
                                       _num(m["sum"]) if m["sum"] else None))
                pending = None
            else:
                out.warnings.append(f"Строка без названия пропущена: «{line}»")
            continue
        m = _P_SUM_ONLY.match(line)
        if m:
            if pending:
                s = _num(m["sum"])
                out.items.append(_item(pending, 1.0, s, s))
                pending = None
            continue
        m = _P_NAME_SUM.match(line)
        if m and not _is_header_noise(line):
            flush_pending()
            s = _num(m["sum"])
            out.items.append(_item(m["name"], 1.0, s, s))
            continue
        # a line with letters and no price: the name of the next price line
        if re.search(r"[A-Za-zА-Яа-яЁё]", line) and not _is_header_noise(line):
            flush_pending()
            pending = line
    flush_pending()

    if not out.items:
        out.warnings.append("Не удалось найти позиции в тексте чека")
    if out.total is None:
        out.warnings.append("Не найдена строка ИТОГ — сумму сверить не с чем")
    elif out.sum_matches_total is False:
        diff = abs(out.items_sum - out.total)
        out.warnings.append(f"Сумма позиций не совпала с ИТОГ на {diff:.2f} ₽".replace(".", ","))
    return out


def _is_header_noise(line: str) -> bool:
    """Address-like lines that may appear inside the body when a receipt has no separators."""
    return bool(re.match(r"^\s*(?:г\.|ул\.|пр\.|пр-т|д\.|магазин|кафе\s+«)", line, re.I))
