"""Settings, taxonomy, demo data and "delete my data"."""
from __future__ import annotations

from decimal import Decimal

from fastapi import APIRouter, Depends, Response
from sqlalchemy.orm import Session

from app import crud, schemas
from app.db import get_session
from app.models import User
from app.routers.deps import COOKIE, current_user, today
from app.services import taxonomy as tax

router = APIRouter(prefix="/api", tags=["me"])


@router.get("/taxonomy")
def taxonomy():
    return tax.tree()


@router.get("/me")
def me(user: User = Depends(current_user)):
    return {"monthly_income": None if user.monthly_income is None else float(user.monthly_income),
            "currency": user.currency, "demo_mode": True}


@router.put("/me")
def update_me(body: schemas.MeIn, user: User = Depends(current_user), db: Session = Depends(get_session)):
    user.monthly_income = None if body.monthly_income is None else Decimal(str(body.monthly_income))
    db.commit()
    return {"monthly_income": body.monthly_income, "currency": user.currency, "demo_mode": True}


@router.post("/demo/load", status_code=204)
def load_demo(user: User = Depends(current_user), db: Session = Depends(get_session)):
    """Three full months plus the current one of a fictional couple. SYNTHETIC data."""
    crud.load_demo(db, user, today())


@router.post("/me/delete", status_code=204)
def delete_me(response: Response, user: User = Depends(current_user), db: Session = Depends(get_session)):
    crud.delete_user_data(db, user)
    response.delete_cookie(COOKIE)
