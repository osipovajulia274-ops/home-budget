"""Forecast feature construction and baselines (numpy only; shared by training and production).

Like `normalize.py`, this module is the single source of truth: `ml/train_forecaster.py` imports it, and
the production forecaster uses exactly the same functions, so the LSTM sees identical inputs.

Task (guide 7.4): on date `d` inside a month predict the total spending from d+1 to the end of the month.

LSTM input  x   : float32 [56, 12]
    0      log1p(daily spending / scale)
    1..7   one-hot day of week (Mon..Sun)
    8, 9   sin / cos of day of month
    10     payday flag (0 when the user's pay days are unknown)
    11     log1p(spending of the month so far / scale)
LSTM scalar rem : remaining days of the month / 31
LSTM output     : log1p(remaining spending / scale)  ->  remaining = scale * expm1(output)
scale = median of the positive daily totals in the 56-day window, at least SCALE_FLOOR roubles.
"""
from __future__ import annotations

import calendar
import datetime as dt
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

WINDOW = 56            # days fed to the LSTM
HISTORY_DAYS = WINDOW + 31   # daily totals needed so that "month so far" is exact inside the window
N_FEATURES = 12
SCALE_FLOOR = 100.0
MIN_DAYS_BASELINE = 28   # less history than this: only the rough average baseline, flagged as approximate
REGULAR_PREFIXES = ("housing.", "subscriptions.")


def month_length(day: dt.date) -> int:
    return calendar.monthrange(day.year, day.month)[1]


def remaining_days(day: dt.date) -> int:
    return month_length(day) - day.day


def history_scale(history: np.ndarray) -> float:
    """Per-user spending scale from the last WINDOW days."""
    w = np.asarray(history[-WINDOW:], dtype=np.float64)
    pos = w[w > 0]
    return float(max(np.median(pos), SCALE_FLOOR)) if pos.size else SCALE_FLOOR


def build_features(history: np.ndarray, end: dt.date,
                   salary_days: Optional[Iterable[int]] = None) -> Tuple[np.ndarray, float, float]:
    """Return (x [56,12] float32, scale, rem/31) for the day `end`.

    history: daily spending totals ending on `end` inclusive (older first). Shorter than HISTORY_DAYS is
    padded with zeros in front; the caller decides whether that much history is enough to use the LSTM.
    """
    h = np.asarray(history, dtype=np.float64)[-HISTORY_DAYS:]
    if h.size < HISTORY_DAYS:
        h = np.concatenate([np.zeros(HISTORY_DAYS - h.size), h])
    scale = history_scale(h)

    end64 = np.datetime64(end, "D")
    days = end64 - np.arange(HISTORY_DAYS - 1, -1, -1).astype("timedelta64[D]")
    months = days.astype("datetime64[M]")
    first_of_month = months.astype("datetime64[D]")
    dom = (days - first_of_month).astype(np.int64) + 1
    dow = (days.astype(np.int64) + 3) % 7            # 1970-01-01 was a Thursday
    # month-to-date spending including the day itself
    cum = np.cumsum(h)
    month_id = months.astype(np.int64)
    first_idx = np.searchsorted(month_id, month_id, side="left")
    mtd = cum - cum[first_idx] + h[first_idx]

    x = np.zeros((HISTORY_DAYS, N_FEATURES), dtype=np.float64)
    x[:, 0] = np.log1p(h / scale)
    x[np.arange(HISTORY_DAYS), 1 + dow] = 1.0
    x[:, 8] = np.sin(2 * np.pi * dom / 31.0)
    x[:, 9] = np.cos(2 * np.pi * dom / 31.0)
    if salary_days:
        sd = np.asarray(sorted(set(int(v) for v in salary_days)))
        x[:, 10] = np.isin(dom, sd).astype(np.float64)
    x[:, 11] = np.log1p(mtd / scale)
    return x[-WINDOW:].astype(np.float32), scale, remaining_days(end) / 31.0


def decode_output(y: float, scale: float) -> float:
    """LSTM output -> roubles remaining until the end of the month (never negative)."""
    return float(max(scale * np.expm1(float(y)), 0.0))


# --------------------------------------------------------------------------------------------- baselines

def baseline_avg(history: np.ndarray, rem: int) -> float:
    """B1: mean daily spending of the last 28 days x remaining days."""
    h = np.asarray(history, dtype=np.float64)[-MIN_DAYS_BASELINE:]
    return float(h.mean() * rem) if h.size else 0.0


def baseline_dow(nonregular_history: np.ndarray, end: dt.date, regular_rest: float = 0.0) -> float:
    """B2: day-of-week means of non-regular spending (last 8 weeks) summed over the remaining days of the
    month, plus the regular payments that are expected but have not happened yet this month."""
    h = np.asarray(nonregular_history, dtype=np.float64)[-WINDOW:]
    n = h.size
    if n == 0:
        return float(regular_rest)
    end64 = np.datetime64(end, "D")
    hist_days = end64 - np.arange(n - 1, -1, -1).astype("timedelta64[D]")
    hist_dow = (hist_days.astype(np.int64) + 3) % 7
    overall = float(h.mean())
    means = np.array([h[hist_dow == k].mean() if np.any(hist_dow == k) else overall for k in range(7)])
    rem = remaining_days(end)
    if rem <= 0:
        return float(regular_rest)
    future = end64 + np.arange(1, rem + 1).astype("timedelta64[D]")
    fut_dow = (future.astype(np.int64) + 3) % 7
    return float(means[fut_dow].sum() + regular_rest)


def _month_shift(first_of_month: dt.date, k: int) -> dt.date:
    m = first_of_month.month - 1 + k
    return dt.date(first_of_month.year + m // 12, m % 12 + 1, 1)


def expected_regular_rest(reg_tx: Sequence[Tuple[dt.date, str, float]], today: dt.date,
                          amount_tol: float = 0.25, day_tol: int = 4) -> float:
    """Regular payments (rent, utilities, phone, subscriptions) expected after `today` this month.

    reg_tx: (date, subcategory, amount) of the user's regular-type payments (see REGULAR_PREFIXES).
    A payment is regular when the previous month has it and one of the two months before has a payment of
    the same subcategory with a similar amount on a similar day of month. If it already happened this
    month it is not counted again.
    """
    cur = today.replace(day=1)
    by_month: Dict[dt.date, List[Tuple[int, str, float]]] = {}
    for d, sub, a in reg_tx:
        if d <= today:
            by_month.setdefault(d.replace(day=1), []).append((d.day, sub, float(a)))
    prev1, prev2, prev3 = (_month_shift(cur, -k) for k in (1, 2, 3))
    anchors = by_month.get(prev1, [])
    if not anchors:
        return 0.0

    def match(items: List[Tuple[int, str, float]], dom0: int, sub0: str, a0: float):
        return [(dom, a) for dom, sub, a in items
                if sub == sub0 and abs(dom - dom0) <= day_tol and abs(a - a0) <= amount_tol * max(a, a0)]

    total = 0.0
    for dom0, sub0, a0 in anchors:
        earlier = match(by_month.get(prev2, []), dom0, sub0, a0) or match(by_month.get(prev3, []), dom0, sub0, a0)
        if not earlier:
            continue
        exp_dom = int(np.median([dom0] + [d for d, _ in earlier]))
        if exp_dom <= today.day:
            continue                               # the date has passed (or is today: already known)
        if match(by_month.get(cur, []), dom0, sub0, a0):
            continue                               # already paid this month
        total += float(a0)
    return total


def is_regular_sub(sub: str) -> bool:
    return sub.startswith(REGULAR_PREFIXES)
