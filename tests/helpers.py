"""Builders for analytics / rules tests."""
import datetime as dt
import itertools

from app.services.analytics import Tx
from app.services.normalize import normalize

_ids = itertools.count(1)
D = dt.date
TODAY = D(2026, 10, 29)


def mk(date, sub, amount, title=None, qty=None, receipt=None):
    title = title or f"{sub} покупка"
    return Tx(id=next(_ids), date=date, title=title, title_norm=normalize(title), amount=float(amount),
              subcategory=sub, qty=qty, receipt_id=receipt)


def base_history(months=((2026, 7), (2026, 8), (2026, 9), (2026, 10))):
    """A small rent payment on the 2nd of every month: gives complete months Jul-Sep for the pace rules."""
    return [mk(D(y, m, 2), "housing.rent", 10000, "Аренда") for y, m in months]
