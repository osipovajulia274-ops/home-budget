"""Synthetic transaction history (guide 7.3). Everything here is SYNTHETIC and labelled as such.

generate_history(n_users, seed) -> DataFrame[user_id, persona, date, subcategory, category, amount, regular]
generate_users(...)              -> DataFrame[user_id, persona, income, salary_days, ...]  (ground truth)

Design: for every user and month, every variable subcategory gets a monthly spending target
(income x spend ratio - regular payments) x share x taste x season x inflation. The target is split
into Poisson-many purchases on days drawn with a day-of-week profile, a payday boost and random
impulse days. Regular payments (rent, utilities, phone/internet, subscriptions) are fixed on their day.
"""
from __future__ import annotations

import calendar
import datetime as dt
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
PERSONAS_PATH = ROOT / "data" / "personas.yaml"

# Subcategories that count as regular payments: rent, utilities, phone/internet and subscriptions.
REGULAR_SUBS = ("housing.", "subscriptions.")


def load_config(path: Path = PERSONAS_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_regular_sub(sub: str) -> bool:
    return sub.startswith(REGULAR_SUBS)


def month_starts(start: str, months: int) -> List[dt.date]:
    y, m, _ = (int(x) for x in start.split("-"))
    out = []
    for i in range(months):
        mm = (m - 1 + i) % 12 + 1
        yy = y + (m - 1 + i) // 12
        out.append(dt.date(yy, mm, 1))
    return out


def _season(cfg: dict, key: str, month: int) -> float:
    arr = cfg["seasonality"].get(key)
    return float(arr[month - 1]) if arr else 1.0


def _lognormal_mean_one(rng: np.random.Generator, sigma: float, size: int) -> np.ndarray:
    """Lognormal with mean exactly 1 (so totals stay on target)."""
    return rng.lognormal(-0.5 * sigma * sigma, sigma, size)


def generate_users(n_users: int, seed: int = 42, cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    rng = np.random.default_rng(seed)
    names = list(cfg["personas"])
    weights = np.array([cfg["personas"][n]["weight"] for n in names], dtype=float)
    weights /= weights.sum()
    rows = []
    for i in range(n_users):
        persona = str(rng.choice(names, p=weights))
        p = cfg["personas"][persona]
        income = float(p["income"]["median"] * np.exp(rng.normal(0, p["income"]["sigma"])))
        rents = rng.random() < p.get("rent_prob", 0.0)
        rent_frac = p.get("rent_frac_if_rent", 0.0) if rents else 0.0
        n_sub_lo, n_sub_hi = p["n_subscriptions"]
        rows.append({
            "user_id": f"u{i:04d}", "persona": persona, "income": round(income, -2),
            "spend_ratio": float(rng.uniform(*p["spend_ratio"])),
            "rent_frac": float(rent_frac) * float(rng.uniform(0.85, 1.15)),
            "utilities_frac": float(p.get("utilities_frac", 0.0)) * float(rng.uniform(0.8, 1.2)),
            "salary_days": ",".join(str(d) for d in p["salary_days"]),
            "n_subscriptions": int(rng.integers(n_sub_lo, n_sub_hi + 1)),
            "inflation": float(rng.uniform(*cfg["simulation"]["inflation_per_month"])),
            "save_frac": float(p.get("save_frac", 0.0)),
        })
    return pd.DataFrame(rows)


def _category(sub: str) -> str:
    return sub.split(".", 1)[0]


def _regular_plan(cfg: dict, u: pd.Series, rng: np.random.Generator) -> List[dict]:
    """Fixed regular payments of a user: day of month, base amount, subcategory, seasonal key."""
    plan: List[dict] = []
    income = float(u["income"])
    if u["rent_frac"] > 0:
        plan.append({"sub": "housing.rent", "day": int(rng.integers(1, 11)), "amount": income * u["rent_frac"],
                     "winter": False, "noise": 0.0})
    if u["utilities_frac"] > 0:
        plan.append({"sub": "housing.utilities", "day": int(rng.integers(12, 26)),
                     "amount": income * u["utilities_frac"], "winter": True, "noise": 0.06})
    plan.append({"sub": "housing.internet_mobile", "day": int(rng.integers(1, 29)),
                 "amount": float(rng.choice(cfg["regular"]["mobile_pool"])), "winter": False, "noise": 0.0})
    if rng.random() < 0.8:
        plan.append({"sub": "housing.internet_mobile", "day": int(rng.integers(1, 29)),
                     "amount": float(rng.choice(cfg["regular"]["internet_pool"])), "winter": False, "noise": 0.0})
    pool = cfg["regular"]["subscription_pool"]
    for idx in rng.choice(len(pool), size=min(int(u["n_subscriptions"]), len(pool)), replace=False):
        sub, price = pool[int(idx)]
        plan.append({"sub": sub, "day": int(rng.integers(1, 29)), "amount": float(price), "winter": False,
                     "noise": 0.0})
    return plan


def generate_history(n_users: int = 300, seed: int = 42, cfg: dict | None = None,
                     users: pd.DataFrame | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (transactions, users). Amounts are positive roubles; `other.transfers` rows exist but are
    excluded from spending everywhere downstream."""
    cfg = cfg or load_config()
    sim = cfg["simulation"]
    users = users if users is not None else generate_users(n_users, seed, cfg)
    rng = np.random.default_rng(seed + 1)
    months = month_starts(sim["start"], int(sim["months"]))
    events: Dict[str, dict] = cfg["events"]
    out: List[tuple] = []

    for _, u in users.iterrows():
        p = cfg["personas"][u["persona"]]
        shares = {s: float(v) for s, v in p["shares"].items() if s in events}
        freq_mult = {s: float(p.get("freq_mult", {}).get(s, 1.0)) for s in shares}
        shares = {s: v for s, v in shares.items() if freq_mult[s] > 0}
        total_share = sum(shares.values())
        taste = {s: float(np.exp(rng.normal(0, sim["taste_sigma"]))) for s in shares}
        salary_days = [int(x) for x in str(u["salary_days"]).split(",")]
        plan = _regular_plan(cfg, u, rng)
        regular_base = sum(item["amount"] for item in plan)
        var_budget = max(float(u["income"]) * float(u["spend_ratio"]) - regular_base, float(u["income"]) * 0.25)

        for mi, m0 in enumerate(months):
            n_days = calendar.monthrange(m0.year, m0.month)[1]
            infl = (1.0 + float(u["inflation"])) ** mi
            dows = np.array([(m0 + dt.timedelta(days=d)).weekday() for d in range(n_days)])
            doms = np.arange(1, n_days + 1)
            payday = np.zeros(n_days, dtype=bool)
            for sd in salary_days:
                for k in range(3):                       # payday and the two days after it
                    if sd - 1 + k < n_days:
                        payday[sd - 1 + k] = True
            impulse = rng.random(n_days) < sim["impulse_day_prob"]

            # --- variable spending
            for sub, share in shares.items():
                ev = events[sub]
                target = (var_budget * share / total_share * taste[sub]
                          * _season(cfg, ev.get("season", "flat"), m0.month) * infl)
                rate = ev["per_week"] * freq_mult[sub] * n_days / 7.0
                if rate <= 0:
                    continue
                dow_w = np.asarray(ev["dow"], dtype=float)[dows]
                if ev.get("discretionary"):
                    dow_w = dow_w * np.where(payday, sim["payday_boost"], 1.0) * np.where(impulse, 3.0, 1.0)
                n_ev = int(rng.poisson(rate))
                if n_ev == 0:
                    continue
                days = rng.choice(n_days, size=n_ev, p=dow_w / dow_w.sum())
                amounts = target / rate * _lognormal_mean_one(rng, ev["sigma"], n_ev)
                for d, a in zip(days, amounts):
                    out.append((u["user_id"], u["persona"], m0 + dt.timedelta(days=int(d)), sub, round(float(a), 2), False))

            # --- rare large expenses
            for sub, prob, (lo, hi) in sim["big_expenses"]:
                if sub in shares or sub in ("home.appliances", "leisure.travel", "health.doctors", "transport.car_service"):
                    if rng.random() < prob * (0.5 + float(u["income"]) / 100_000):
                        d = int(rng.integers(0, n_days))
                        a = rng.uniform(lo, hi) * infl * min(1.5, 0.6 + float(u["income"]) / 100_000)
                        out.append((u["user_id"], u["persona"], m0 + dt.timedelta(days=d), sub, round(float(a), 2), False))

            # --- regular payments
            for item in plan:
                amount = item["amount"] * infl ** 0   # fixed price (subscriptions rarely change in a year)
                if item["winter"]:
                    amount *= cfg["regular"]["utilities_winter"][m0.month - 1]
                if item["noise"]:
                    amount *= 1.0 + rng.normal(0, item["noise"])
                day = min(item["day"], n_days)
                out.append((u["user_id"], u["persona"], m0 + dt.timedelta(days=day - 1), item["sub"],
                            round(float(amount), 2), True))

            # --- savings transfers on payday (excluded from spending)
            if u["save_frac"] > 0:
                d = min(salary_days[0] + 1, n_days)
                out.append((u["user_id"], u["persona"], m0 + dt.timedelta(days=d - 1), "other.transfers",
                            round(float(u["income"] * u["save_frac"] * rng.uniform(0.6, 1.2)), 2), False))

    df = pd.DataFrame(out, columns=["user_id", "persona", "date", "subcategory", "amount", "regular"])
    df["category"] = df["subcategory"].map(_category)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["user_id", "date"], kind="stable").reset_index(drop=True)
    return df[["user_id", "persona", "date", "subcategory", "category", "amount", "regular"]], users


def food_share(tx: pd.DataFrame) -> float:
    """Share of `food` (groceries) in total spending, transfers excluded."""
    spend = tx[tx["subcategory"] != "other.transfers"]
    return float(spend.loc[spend["category"] == "food", "amount"].sum() / spend["amount"].sum())


def split_users(user_ids: List[str], seed: int = 42, fractions=(0.70, 0.15, 0.15)) -> Dict[str, List[str]]:
    """Forecast split BY USER (guide 7.3): no user appears in two parts."""
    ids = sorted(user_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    a, b = int(round(n * fractions[0])), int(round(n * (fractions[0] + fractions[1])))
    return {"train": sorted(ids[:a]), "val": sorted(ids[a:b]), "test": sorted(ids[b:])}
