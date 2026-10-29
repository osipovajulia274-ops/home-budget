"""Aggregates over a user's transactions (pure Python: the DB layer only loads rows into `Tx`).

Everything here describes what the user has already spent; nothing predicts the future.
"""
from __future__ import annotations

import calendar
import datetime as dt
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from app.services import taxonomy as tax
from app.services.goals import TOTAL, GoalStatus

MONTH_RE = re.compile(r"^(\d{4})-(0[1-9]|1[0-2])$")


@dataclass(frozen=True)
class Tx:
    id: Optional[int]
    date: dt.date
    title: str
    title_norm: str
    amount: float
    subcategory: str
    qty: Optional[float] = None
    receipt_id: Optional[int] = None
    category_source: str = "model"

    @property
    def category(self) -> str:
        return tax.category_of(self.subcategory)


def parse_month(s: str) -> dt.date:
    """'2026-10' -> date(2026, 10, 1). Raises ValueError for anything else."""
    m = MONTH_RE.match(s or "")
    if not m:
        raise ValueError("month must look like 2026-10")
    return dt.date(int(m[1]), int(m[2]), 1)


def month_key(d: dt.date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def month_end(first: dt.date) -> dt.date:
    return first.replace(day=calendar.monthrange(first.year, first.month)[1])


def prev_month(first: dt.date) -> dt.date:
    return (first - dt.timedelta(days=1)).replace(day=1)


def as_of_for(first: dt.date, today: dt.date) -> dt.date:
    """The day up to which a month is looked at: today for the current month, the last day for a past one."""
    return min(today, month_end(first))


def spending(txs: Iterable[Tx]) -> List[Tx]:
    return [t for t in txs if tax.is_spending(t.subcategory)]


def in_month(txs: Iterable[Tx], first: dt.date) -> List[Tx]:
    return [t for t in txs if t.date.year == first.year and t.date.month == first.month]


def matches(sub: str, scope: str) -> bool:
    """Does a subcategory belong to a goal/category scope ('total', 'food', 'eating_out.coffee')?"""
    return scope == TOTAL or sub == scope or sub.startswith(scope + ".")


def total_of(txs: Iterable[Tx], scope: str = TOTAL) -> float:
    return round(sum(t.amount for t in txs if matches(t.subcategory, scope)), 2)


def spend_by_sub(txs: Iterable[Tx]) -> Dict[str, float]:
    out: Dict[str, float] = defaultdict(float)
    for t in txs:
        out[t.subcategory] += t.amount
    return dict(out)


def _delta(cur: float, prev: float) -> Optional[float]:
    return round((cur - prev) / prev, 4) if prev > 0 else None


def summary(all_spend: Sequence[Tx], first: dt.date, today: dt.date,
            goal_statuses: Sequence[GoalStatus] = ()) -> dict:
    """The `Summary` of the API for one month. `all_spend` = every spending transaction of the user."""
    cur = in_month(all_spend, first)
    prev_first = prev_month(first)
    prev = in_month(all_spend, prev_first)
    total = total_of(cur)
    prev_total = total_of(prev) if prev else None
    goals = {g.scope: g for g in goal_statuses}

    by_cat: Dict[str, List[Tx]] = defaultdict(list)
    for t in cur:
        by_cat[t.category].append(t)
    prev_by_cat = spend_by_sub(prev)

    cats = []
    for cid, items in by_cat.items():
        amount = total_of(items)
        prev_amount = round(sum(v for k, v in prev_by_cat.items() if tax.category_of(k) == cid), 2)
        subs = spend_by_sub(items)
        g = goals.get(cid)
        cats.append({
            "category": cid, "title": tax.category_titles().get(cid, cid), "amount": amount,
            "share": round(amount / total, 4) if total > 0 else 0.0,
            "prev_amount": prev_amount if prev else None,
            "delta_vs_prev": _delta(amount, prev_amount) if prev else None,
            "goal_limit": g.limit if g else None,
            "goal_used": round(g.used_share, 4) if g else None,
            "subcategories": sorted(
                ({"subcategory": s, "title": tax.subcategory_titles().get(s, s), "amount": round(a, 2)}
                 for s, a in subs.items()), key=lambda r: -r["amount"]),
        })
    cats.sort(key=lambda r: -r["amount"])

    top = sorted(cur, key=lambda t: -t.amount)[:5]
    gt = goals.get(TOTAL)
    return {
        "month": month_key(first), "as_of": as_of_for(first, today).isoformat(),
        "total": total, "prev_month_total": prev_total,
        "delta_vs_prev": _delta(total, prev_total) if prev_total is not None else None,
        "by_category": cats,
        "top_items": [{"id": t.id, "title": t.title, "amount": t.amount, "date": t.date.isoformat(),
                       "subcategory": t.subcategory} for t in top],
        "days_with_data": len({t.date for t in cur}),
        "goal_total": ({"limit": gt.limit, "spent": gt.spent, "used_share": round(gt.used_share, 4),
                        "remaining": gt.remaining, "level": gt.level} if gt else None),
    }


# ---------------------------------------------------------------- recurring payments ----

@dataclass(frozen=True)
class Recurring:
    title: str
    title_norm: str
    subcategory: str
    category: str
    amount: float             # the most recent charge
    occurrences: int
    last_date: dt.date
    interval_days: int
    tx_ids: tuple

    @property
    def per_year(self) -> float:
        return round(self.amount * 12, 2)


INTERVAL = (28, 31)           # days between charges
AMOUNT_TOLERANCE = 0.05
STALE_AFTER_DAYS = 62         # a subscription not charged for two months is no longer listed


def detect_recurring(txs: Iterable[Tx], today: dt.date) -> List[Recurring]:
    """Same description, amount within +-5%, 28-31 days apart, at least two charges (guide, rule R10)."""
    groups: Dict[str, List[Tx]] = defaultdict(list)
    for t in txs:
        if t.title_norm and tax.is_spending(t.subcategory):
            groups[t.title_norm].append(t)
    out: List[Recurring] = []
    for key, rows in groups.items():
        rows.sort(key=lambda t: t.date)
        series: List[Tx] = []
        gaps: List[int] = []
        for a, b in zip(rows, rows[1:]):
            gap = (b.date - a.date).days
            if INTERVAL[0] <= gap <= INTERVAL[1] and abs(b.amount - a.amount) <= AMOUNT_TOLERANCE * a.amount:
                if not series or series[-1] is not a:     # a new run starts: keep only the latest run
                    series, gaps = [a], []
                series.append(b)
                gaps.append(gap)
        if len(series) < 2:
            continue
        last = series[-1]
        if (today - last.date).days > STALE_AFTER_DAYS:
            continue
        out.append(Recurring(title=last.title, title_norm=key, subcategory=last.subcategory,
                             category=last.category, amount=last.amount, occurrences=len(series),
                             last_date=last.date, interval_days=round(statistics.median(gaps)),
                             tx_ids=tuple(t.id for t in series if t.id is not None)))
    out.sort(key=lambda r: -r.amount)
    return out


def daily_totals(txs: Iterable[Tx]) -> Dict[dt.date, float]:
    out: Dict[dt.date, float] = defaultdict(float)
    for t in txs:
        out[t.date] += t.amount
    return dict(out)
