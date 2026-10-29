"""Month summary, the user's own history and subscriptions."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app import crud
from app.db import get_session
from app.models import User
from app.routers.deps import current_user, month_param, today
from app.services import analytics as an
from app.services import history

router = APIRouter(prefix="/api", tags=["summary"])


@router.get("/summary")
def summary(month: Optional[str] = None, user: User = Depends(current_user), db: Session = Depends(get_session)):
    first, now = month_param(month), today()
    spend = an.spending(crud.all_txs(db, user.id))
    return an.summary(spend, first, now, crud.goal_statuses(db, user.id, spend, first, now))


@router.get("/history")
def history_(month: Optional[str] = None, user: User = Depends(current_user), db: Session = Depends(get_session)):
    """Facts about the user's own past: monthly totals, the typical month, the pace against it. No forecast."""
    first, now = month_param(month), today()
    as_of = an.as_of_for(first, now)
    rows = [(t.date, t.amount) for t in an.spending(crud.all_txs(db, user.id)) if t.date <= as_of]
    totals = history.monthly_totals(rows, as_of, n_months=6)
    typical = history.typical_month([t for t in totals if t.month != an.month_key(first)])
    pace = history.pace_comparison(rows, as_of)
    return {
        "month": an.month_key(first), "as_of": as_of.isoformat(),
        "months": [t.__dict__ for t in totals],
        "typical": typical.__dict__ if typical else None,
        "pace": pace.__dict__ if pace else None,
        "note": None if typical else "Сравнения появятся после первого полного месяца с вашими данными.",
    }


@router.get("/subscriptions")
def subscriptions(user: User = Depends(current_user), db: Session = Depends(get_session)):
    found = crud.mark_recurring(db, user.id, today())
    items = [{"title": r.title, "subcategory": r.subcategory, "category": r.category, "amount": r.amount,
              "occurrences": r.occurrences, "last_date": r.last_date.isoformat(), "interval_days": r.interval_days,
              "per_year": r.per_year} for r in found]
    subs = [i for i in items if i["category"] == "subscriptions"]
    return {"items": items,
            "subscriptions_per_month": round(sum(i["amount"] for i in subs), 2),
            "subscriptions_per_year": round(sum(i["amount"] for i in subs) * 12, 2)}
