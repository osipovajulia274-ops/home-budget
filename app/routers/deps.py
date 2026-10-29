"""Request-scoped helpers: the anonymous demo user (httpOnly cookie), "today", month parsing."""
from __future__ import annotations

import datetime as dt
from typing import Optional

from fastapi import Depends, HTTPException, Request, Response
from sqlalchemy.orm import Session

from app import crud
from app.config import get_settings
from app.db import get_session
from app.models import User
from app.services import analytics as an
from app.services import clock

COOKIE = "uid"


def current_user(request: Request, response: Response, db: Session = Depends(get_session)) -> User:
    """Demo mode: no passwords. The first visit creates a user and stores its id in an httpOnly cookie."""
    user = crud.get_user(db, request.cookies.get(COOKIE))
    if user is None:
        user = crud.create_user(db)
        set_user_cookie(response, user)
    return user


def set_user_cookie(response, user: User) -> None:
    response.set_cookie(COOKIE, str(user.id), httponly=True, samesite="lax", secure=get_settings().is_prod,
                        max_age=365 * 24 * 3600)


def today() -> dt.date:
    return clock.today(get_settings().timezone)


def month_param(month: Optional[str]) -> dt.date:
    """`?month=2026-10` -> first day of that month; the current month when omitted."""
    if not month:
        return today().replace(day=1)
    try:
        return an.parse_month(month)
    except ValueError:
        raise HTTPException(status_code=422, detail="month должен иметь вид 2026-10")
