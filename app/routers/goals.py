"""Monthly limits (replace the guide's /api/budgets): total, per category or per subcategory."""
from __future__ import annotations

from decimal import Decimal
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import crud, schemas
from app.db import get_session
from app.models import Goal, User
from app.routers.deps import current_user, month_param, today
from app.services import analytics as an
from app.services import taxonomy as tax

router = APIRouter(prefix="/api/goals", tags=["goals"])


def _title(scope: str) -> str:
    return "Общий лимит" if scope == "total" else tax.all_titles().get(scope, scope)


def _out(g: Goal) -> dict:
    return {"id": g.id, "scope": g.scope, "title": _title(g.scope), "limit_amount": float(g.limit_amount)}


def _own(db: Session, user: User, goal_id: int) -> Goal:
    g = db.get(Goal, goal_id)
    if g is None or g.user_id != user.id:
        raise HTTPException(status_code=404, detail="Цель не найдена")
    return g


@router.get("")
def list_(month: Optional[str] = None, user: User = Depends(current_user), db: Session = Depends(get_session)):
    """Goals with their status for the month: spent, remaining, per-day, level (ok / ahead_of_pace / watch / exceeded)."""
    first, now = month_param(month), today()
    spend = an.spending(crud.all_txs(db, user.id))
    statuses = {s.scope: s for s in crud.goal_statuses(db, user.id, spend, first, now)}
    out = []
    for g in crud.goals_of(db, user.id):
        s = statuses[g.scope]
        out.append({**_out(g), "spent": s.spent, "used_share": round(s.used_share, 4), "remaining": s.remaining,
                    "over_by": s.over_by, "days_left": s.days_left, "per_day_left": s.per_day_left,
                    "elapsed_share": round(s.elapsed_share, 4), "level": s.level})
    return out


@router.post("", status_code=201)
def create(body: schemas.GoalIn, user: User = Depends(current_user), db: Session = Depends(get_session)):
    if not schemas.valid_scope(body.scope):
        raise HTTPException(status_code=422, detail="Неизвестная категория цели")
    if db.scalar(select(Goal).where(Goal.user_id == user.id, Goal.scope == body.scope)):
        raise HTTPException(status_code=409, detail="Для этой категории цель уже есть — измените её")
    g = Goal(user_id=user.id, scope=body.scope, limit_amount=Decimal(str(body.limit_amount)))
    db.add(g)
    db.commit()
    return _out(g)


@router.put("/{goal_id}")
def update(goal_id: int, body: schemas.GoalIn, user: User = Depends(current_user), db: Session = Depends(get_session)):
    g = _own(db, user, goal_id)
    if body.scope != g.scope:
        raise HTTPException(status_code=422, detail="Категорию цели менять нельзя — создайте новую")
    g.limit_amount = Decimal(str(body.limit_amount))
    db.commit()
    return _out(g)


@router.delete("/{goal_id}", status_code=204)
def delete(goal_id: int, user: User = Depends(current_user), db: Session = Depends(get_session)):
    db.delete(_own(db, user, goal_id))
    db.commit()
