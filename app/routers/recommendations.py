from __future__ import annotations

import datetime as dt
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import crud
from app.db import get_session
from app.models import Recommendation, User
from app.routers.deps import current_user, month_param, today

router = APIRouter(prefix="/api/recommendations", tags=["recommendations"])


def _out(r: Recommendation) -> dict:
    return {"id": r.id, "rule_id": r.rule_id, "category": r.category, "severity": r.severity, "title": r.title,
            "body": r.body, "action": r.action, "payload": {k: v for k, v in (r.payload or {}).items() if k != "key"}}


@router.get("")
def list_(month: Optional[str] = None, user: User = Depends(current_user), db: Session = Depends(get_session)):
    """Recomputed on every request; a card hidden by the user does not come back this month."""
    return [_out(r) for r in crud.refresh_recommendations(db, user, month_param(month), today())]


@router.post("/{rec_id}/dismiss", status_code=204)
def dismiss(rec_id: int, user: User = Depends(current_user), db: Session = Depends(get_session)):
    r = db.get(Recommendation, rec_id)
    if r is None or r.user_id != user.id:
        raise HTTPException(status_code=404, detail="Карточка не найдена")
    r.dismissed_at = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    db.commit()
