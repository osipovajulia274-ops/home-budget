"""Forecaster data preparation and evaluation (numpy/pandas; torch-free so it can be tested anywhere).

Samples are (user, date d): history up to d -> total spending from d+1 to the end of d's month.
The user split (70/15/15 by user) happens before any sample is built.
"""
from __future__ import annotations

import calendar
import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from app.services.forecast_core import (HISTORY_DAYS, MIN_DAYS_BASELINE, N_FEATURES, WINDOW, baseline_avg,
                                        baseline_dow, build_features, expected_regular_rest, history_scale,
                                        is_regular_sub, remaining_days)

EVAL_DAYS = (5, 10, 15, 20, 25)
TRAIN_DAY_STEP = 3
TRANSFER = "other.transfers"


@dataclass
class UserSeries:
    user_id: str
    start: dt.date
    daily: np.ndarray                    # all spending per day (transfers excluded)
    regular_daily: np.ndarray            # the part that belongs to regular-type subcategories
    reg_tx: List[Tuple[dt.date, str, float]] = field(default_factory=list)
    salary_days: Optional[Tuple[int, ...]] = None

    @property
    def n_days(self) -> int:
        return len(self.daily)

    def date_at(self, i: int) -> dt.date:
        return self.start + dt.timedelta(days=int(i))


def build_series(tx: pd.DataFrame, users: pd.DataFrame, salary_known_frac: float = 0.5,
                 seed: int = 42, user_ids: Optional[Sequence[str]] = None) -> Dict[str, UserSeries]:
    """Daily series per user on one shared calendar. The payday flag is shown to the model for only a
    fraction of users (the app does not always know the pay day), so the LSTM works with and without it."""
    spend = tx[tx["subcategory"] != TRANSFER]
    start = spend["date"].min().date()
    end = spend["date"].max().date()
    n_days = (end - start).days + 1
    rng = np.random.default_rng(seed)
    salary = {r.user_id: tuple(int(x) for x in str(r.salary_days).split(",")) for r in users.itertuples()}
    wanted = set(user_ids) if user_ids is not None else None
    out: Dict[str, UserSeries] = {}
    for uid, g in spend.groupby("user_id", sort=True):
        if wanted is not None and uid not in wanted:
            continue
        idx = (g["date"].dt.date.map(lambda d: (d - start).days)).to_numpy()
        amt = g["amount"].to_numpy(dtype=np.float64)
        daily = np.bincount(idx, weights=amt, minlength=n_days)
        reg_mask = g["subcategory"].map(is_regular_sub).to_numpy()
        reg_daily = np.bincount(idx[reg_mask], weights=amt[reg_mask], minlength=n_days)
        reg_tx = [(d.date(), s, float(a)) for d, s, a in
                  zip(g.loc[reg_mask, "date"], g.loc[reg_mask, "subcategory"], amt[reg_mask])]
        known = rng.random() < salary_known_frac
        out[uid] = UserSeries(uid, start, daily, reg_daily, reg_tx, salary.get(uid) if known else None)
    return out


def sample_points(series: UserSeries, days: Sequence[int]) -> List[Tuple[int, int]]:
    """(index of day d, index of last day of its month) for every requested day of every full month with
    at least WINDOW days of history before d."""
    pts: List[Tuple[int, int]] = []
    first = series.start
    d0 = dt.date(first.year, first.month, 1)
    month = d0
    while True:
        dim = calendar.monthrange(month.year, month.month)[1]
        m_start = (month - series.start).days
        m_end = m_start + dim - 1
        if m_end > series.n_days - 1:
            break
        for d in days:
            if 1 <= d < dim:
                i = m_start + d - 1
                if i >= WINDOW - 1:
                    pts.append((i, m_end))
        month = dt.date(month.year + month.month // 12, month.month % 12 + 1, 1)
    return pts


@dataclass
class Samples:
    x: np.ndarray            # [N, WINDOW, N_FEATURES] float32
    rem: np.ndarray          # [N, 1] float32
    y: np.ndarray            # [N] float32, log1p(rest / scale)
    scale: np.ndarray        # [N] float64
    spent: np.ndarray        # [N] spending of the month up to and including d
    rest: np.ndarray         # [N] true spending d+1 .. end of month
    day: np.ndarray          # [N] day of month d
    user: np.ndarray         # [N] object, user id
    b1: np.ndarray           # [N] baseline B1 prediction of `rest`
    b2: np.ndarray           # [N] baseline B2 prediction of `rest`


def build_samples(series: Dict[str, UserSeries], days: Sequence[int]) -> Samples:
    xs, rems, ys, scales, spent, rest, dayv, users, b1, b2 = ([] for _ in range(10))
    for uid, s in series.items():
        csum = np.concatenate([[0.0], np.cumsum(s.daily)])
        for i, m_end in sample_points(s, days):
            end = s.date_at(i)
            lo = max(0, i - HISTORY_DAYS + 1)
            hist = s.daily[lo:i + 1]
            x, scale, rem31 = build_features(hist, end, s.salary_days)
            rest_true = float(csum[m_end + 1] - csum[i + 1])
            month_start = i - (end.day - 1)
            spent_now = float(csum[i + 1] - csum[month_start])
            nonreg = (s.daily - s.regular_daily)[lo:i + 1]
            reg_rest = expected_regular_rest(s.reg_tx, end)
            xs.append(x); rems.append([rem31]); ys.append(np.log1p(rest_true / scale)); scales.append(scale)
            spent.append(spent_now); rest.append(rest_true); dayv.append(end.day); users.append(uid)
            b1.append(baseline_avg(hist, remaining_days(end)))
            b2.append(baseline_dow(nonreg, end, reg_rest))
    return Samples(
        x=np.asarray(xs, dtype=np.float32).reshape(-1, WINDOW, N_FEATURES),
        rem=np.asarray(rems, dtype=np.float32).reshape(-1, 1),
        y=np.asarray(ys, dtype=np.float32), scale=np.asarray(scales), spent=np.asarray(spent),
        rest=np.asarray(rest), day=np.asarray(dayv), user=np.asarray(users, dtype=object),
        b1=np.asarray(b1), b2=np.asarray(b2))


# ------------------------------------------------------------------------------------------------ metrics

def month_total_errors(s: Samples, rest_pred: np.ndarray) -> Dict[str, float]:
    """MAE and MAPE of the predicted month total (spent so far + predicted rest)."""
    true_total = s.spent + s.rest
    pred_total = s.spent + rest_pred
    err = np.abs(pred_total - true_total)
    ok = true_total > 0
    return {"mae": float(err.mean()), "mape": float((err[ok] / true_total[ok]).mean()), "n": int(len(err))}


def evaluate_methods(s: Samples, preds: Dict[str, np.ndarray]) -> Dict[str, dict]:
    """Per method: overall and per evaluation day MAE/MAPE of the month total."""
    out: Dict[str, dict] = {}
    for name, p in preds.items():
        entry = {"overall": month_total_errors(s, p), "by_day": {}}
        for d in EVAL_DAYS:
            m = s.day == d
            if m.any():
                sub = Samples(*(getattr(s, f)[m] for f in s.__dataclass_fields__))
                entry["by_day"][str(d)] = month_total_errors(sub, p[m])
        out[name] = entry
    return out


def interval_quantiles(s: Samples, rest_pred: np.ndarray, lo: float = 0.10, hi: float = 0.90) -> Dict[str, float]:
    """Multiplicative interval for the remaining spending: rest_true / rest_pred quantiles on validation."""
    mask = rest_pred > 1.0
    ratio = s.rest[mask] / rest_pred[mask]
    return {"q_lo": float(np.quantile(ratio, lo)), "q_hi": float(np.quantile(ratio, hi)), "n": int(mask.sum())}


def choose_method(results: Dict[str, dict], margin: float = 0.05) -> Tuple[str, str]:
    """Guide 7.4: the LSTM goes to production only when its MAE is at least 5% below the best baseline."""
    maes = {k: v["overall"]["mae"] for k, v in results.items()}
    best_base = min(("baseline_avg", "baseline_dow"), key=lambda k: maes[k])
    if "lstm" in maes and maes["lstm"] <= (1 - margin) * maes[best_base]:
        return "lstm", (f"LSTM MAE {maes['lstm']:.0f} <= {1 - margin:.2f} x best baseline "
                        f"({best_base} {maes[best_base]:.0f})")
    why = (f"LSTM MAE {maes['lstm']:.0f} is not {margin:.0%} below the best baseline ({best_base} "
           f"{maes[best_base]:.0f})") if "lstm" in maes else "no LSTM result"
    return best_base, why
