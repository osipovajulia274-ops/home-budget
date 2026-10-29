"""User goals: "I do not want to spend more than X this month" (in total or in one category).

The app shows how far the month is from the limit and turns it into recommendation cards. It states what
has happened and simple arithmetic ("X roubles left for N days"); it does not predict future spending.
Not financial advice: only the user's own limits and their own spending.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional

from app.services.history import days_in_month, fmt_rub

TOTAL = "total"
WATCH_AT = 0.90            # "almost used up"
PACE_GAP = 0.15            # spent share exceeds the share of the month that has passed by this much
PACE_MIN_USED = 0.20       # ...and at least this much of the limit is used (early-month noise)


@dataclass(frozen=True)
class Goal:
    scope: str             # "total", a category id ("food") or a subcategory id ("eating_out.coffee")
    limit: float

    def __post_init__(self):
        if not self.limit > 0:
            raise ValueError("limit must be positive")


@dataclass(frozen=True)
class GoalStatus:
    scope: str
    limit: float
    spent: float
    used_share: float
    remaining: float            # >= 0
    over_by: float              # >= 0
    days_left: int              # days after today until the end of the month
    elapsed_share: float        # share of the month that has passed (today included)
    per_day_left: float         # remaining / days to go (today counts when it is the last day)
    level: str                  # ok | ahead_of_pace | watch | exceeded


def spent_for_scope(scope: str, spend_by_sub: Mapping[str, float]) -> float:
    """Sum of the month's spending that belongs to the scope. `spend_by_sub` is keyed by subcategory id and
    must not contain transfers."""
    if scope == TOTAL:
        return float(sum(spend_by_sub.values()))
    return float(sum(v for k, v in spend_by_sub.items() if k == scope or k.startswith(scope + ".")))


def evaluate_goal(goal: Goal, spent: float, today: dt.date) -> GoalStatus:
    dim = days_in_month(today)
    days_left = dim - today.day
    used = spent / goal.limit
    elapsed = today.day / dim
    remaining = max(goal.limit - spent, 0.0)
    if spent >= goal.limit:
        level = "exceeded"
    elif used >= WATCH_AT:
        level = "watch"
    elif used >= PACE_MIN_USED and used - elapsed >= PACE_GAP:
        level = "ahead_of_pace"
    else:
        level = "ok"
    return GoalStatus(scope=goal.scope, limit=goal.limit, spent=round(spent, 2), used_share=used,
                      remaining=round(remaining, 2), over_by=round(max(spent - goal.limit, 0.0), 2),
                      days_left=days_left, elapsed_share=elapsed,
                      per_day_left=round(remaining / max(days_left, 1), 2), level=level)


def evaluate_goals(goals: Iterable[Goal], spend_by_sub: Mapping[str, float], today: dt.date) -> List[GoalStatus]:
    return [evaluate_goal(g, spent_for_scope(g.scope, spend_by_sub), today) for g in goals]


def _title_of(scope: str, titles: Mapping[str, str]) -> str:
    return "общий лимит" if scope == TOTAL else f"лимит «{titles.get(scope, scope)}»"


def goal_recommendations(statuses: Iterable[GoalStatus], titles: Optional[Mapping[str, str]] = None) -> List[dict]:
    """Recommendation cards (guide section 9, rules R01/R02 reworked): no forecast, only the facts."""
    titles = titles or {}
    out: List[dict] = []
    for s in statuses:
        name = _title_of(s.scope, titles)
        category = None if s.scope == TOTAL else s.scope.split(".")[0]
        payload = {"scope": s.scope, "limit": s.limit, "spent": s.spent, "used_share": round(s.used_share, 3),
                   "elapsed_share": round(s.elapsed_share, 3), "remaining": s.remaining, "days_left": s.days_left}
        if s.level == "exceeded":
            out.append({"rule_id": "R02_goal_exceeded", "category": category, "severity": "alert",
                        "title": f"Превышен {name}: на {fmt_rub(s.over_by)}",
                        "body": f"Потрачено {fmt_rub(s.spent)} при лимите {fmt_rub(s.limit)}. "
                                f"До конца месяца осталось {s.days_left} дн.",
                        "action": "Посмотреть, из чего сложилась сумма", "payload": payload, "effect": s.over_by})
        elif s.level == "watch":
            out.append({"rule_id": "R02_goal_watch", "category": category, "severity": "warning",
                        "title": f"{name.capitalize()} почти исчерпан: {round(s.used_share * 100)}%",
                        "body": f"Осталось {fmt_rub(s.remaining)} на {max(s.days_left, 1)} дн. — "
                                f"примерно {fmt_rub(s.per_day_left)} в день.",
                        "action": "Посмотреть траты по этой цели", "payload": payload, "effect": s.remaining})
        elif s.level == "ahead_of_pace":
            out.append({"rule_id": "R01_goal_pace", "category": category, "severity": "warning",
                        "title": f"{name.capitalize()}: потрачено {round(s.used_share * 100)}%, "
                                 f"а месяца прошло {round(s.elapsed_share * 100)}%",
                        "body": f"Осталось {fmt_rub(s.remaining)} на {max(s.days_left, 1)} дн. — "
                                f"примерно {fmt_rub(s.per_day_left)} в день.",
                        "action": "Посмотреть траты по этой цели", "payload": payload, "effect": s.remaining})
    return out
