"""What the user actually spent: monthly totals, a typical month and the pace against earlier months.

This replaces the "forecast" of the original guide (decision of 2026-10-28): the app does not predict the
future, it shows the user's own past and compares the current month with it. Pure Python, no pandas/numpy,
so it runs on Render without extra dependencies.

Input everywhere: `txs` = iterable of (date, amount) for SPENDING only (transfers already excluded).
"""
from __future__ import annotations

import calendar
import datetime as dt
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

Tx = Tuple[dt.date, float]

TYPICAL_MONTHS = 3          # the "usual" month is judged by the last three complete months
START_GRACE_DAYS = 6        # a month counts as complete if the user's data starts within its first week


def month_key(d: dt.date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def first_of_month(d: dt.date) -> dt.date:
    return d.replace(day=1)


def shift_month(first: dt.date, k: int) -> dt.date:
    m = first.month - 1 + k
    return dt.date(first.year + m // 12, m % 12 + 1, 1)


def days_in_month(d: dt.date) -> int:
    return calendar.monthrange(d.year, d.month)[1]


@dataclass(frozen=True)
class MonthTotal:
    month: str                 # "2026-09"
    total: float
    days_with_data: int
    complete: bool             # a finished month with data from (nearly) its first day


@dataclass(frozen=True)
class TypicalMonth:
    median: float
    mean: float
    lowest: float
    highest: float
    n_months: int


@dataclass(frozen=True)
class PaceComparison:
    spent_now: float           # spending of the current month up to and including today
    usual_by_now: float        # median spending by the same day of month in the earlier months
    difference: float          # spent_now - usual_by_now
    share_of_usual: Optional[float]   # spent_now / usual_by_now (None when the usual amount is 0)
    n_months: int


def monthly_totals(txs: Iterable[Tx], today: dt.date, n_months: int = 6) -> List[MonthTotal]:
    """The last `n_months` calendar months including the current one, oldest first. Months before the
    user's first transaction are not listed."""
    rows = list(txs)
    if not rows:
        return []
    first_day = min(d for d, _ in rows)
    sums: Dict[str, float] = defaultdict(float)
    days: Dict[str, set] = defaultdict(set)
    for d, a in rows:
        if d <= today:
            sums[month_key(d)] += float(a)
            days[month_key(d)].add(d)
    cur = first_of_month(today)
    out: List[MonthTotal] = []
    for k in range(-(n_months - 1), 1):
        m0 = shift_month(cur, k)
        if m0 < first_of_month(first_day):
            continue
        key = month_key(m0)
        complete = m0 < cur and first_day <= m0 + dt.timedelta(days=START_GRACE_DAYS)
        out.append(MonthTotal(key, round(sums.get(key, 0.0), 2), len(days.get(key, ())), complete))
    return out


def typical_month(totals: List[MonthTotal], max_months: int = TYPICAL_MONTHS) -> Optional[TypicalMonth]:
    """Median/mean/range over the last complete months; None until the user has one complete month."""
    done = [t.total for t in totals if t.complete][-max_months:]
    if not done:
        return None
    return TypicalMonth(median=round(statistics.median(done), 2), mean=round(sum(done) / len(done), 2),
                        lowest=min(done), highest=max(done), n_months=len(done))


def pace_comparison(txs: Iterable[Tx], today: dt.date, max_months: int = TYPICAL_MONTHS) -> Optional[PaceComparison]:
    """How much the user had spent by this day of the month in earlier months, versus now. A comparison of
    facts, not an extrapolation."""
    rows = list(txs)
    totals = monthly_totals(rows, today, n_months=max_months + 1)
    complete = [t for t in totals if t.complete][-max_months:]
    if not complete:
        return None
    cur = first_of_month(today)
    spent_now = sum(float(a) for d, a in rows if cur <= d <= today)
    by_day: List[float] = []
    for t in complete:
        y, m = int(t.month[:4]), int(t.month[5:])
        cutoff = min(today.day, calendar.monthrange(y, m)[1])
        by_day.append(sum(float(a) for d, a in rows if d.year == y and d.month == m and d.day <= cutoff))
    usual = statistics.median(by_day)
    return PaceComparison(spent_now=round(spent_now, 2), usual_by_now=round(usual, 2),
                          difference=round(spent_now - usual, 2),
                          share_of_usual=(spent_now / usual) if usual > 0 else None, n_months=len(complete))


def fmt_rub(x: float) -> str:
    """12345.6 -> '12 346 ₽' (non-breaking spaces)."""
    return f"{round(x):,}".replace(",", " ") + " ₽"
