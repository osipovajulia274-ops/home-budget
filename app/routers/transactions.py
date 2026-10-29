from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import crud, schemas
from app.db import get_session
from app.models import CategoryFeedback, Transaction, User
from app.routers.deps import current_user, month_param
from app.services import analytics as an
from app.services import taxonomy as tax
from app.services.categorizer import get_categorizer

router = APIRouter(prefix="/api/transactions", tags=["transactions"])


def _own(db: Session, user: User, tx_id: int) -> Transaction:
    tx = db.get(Transaction, tx_id)
    if tx is None or tx.user_id != user.id:
        raise HTTPException(status_code=404, detail="Транзакция не найдена")
    return tx


@router.post("", response_model=schemas.TransactionOut, status_code=201)
def create(body: schemas.TransactionIn, user: User = Depends(current_user), db: Session = Depends(get_session)):
    if body.subcategory is not None and not tax.is_valid_sub(body.subcategory):
        raise HTTPException(status_code=422, detail="Неизвестная подкатегория")
    if body.subcategory:
        sub, conf, source = body.subcategory, 1.0, "user"
    else:
        pred = get_categorizer().categorize([body.title], crud.feedback_map(db, user.id))[0]
        sub, conf, source = pred.subcategory, pred.confidence, pred.source
    row = crud.new_transaction(user.id, date=body.date, title=body.title, amount=body.amount, subcategory=sub,
                               qty=body.qty, confidence=conf, source=source)
    db.add(row)
    db.commit()
    return crud.tx_out(row)


@router.get("", response_model=List[schemas.TransactionOut])
def list_(month: Optional[str] = None, category: Optional[str] = None, subcategory: Optional[str] = None,
          user: User = Depends(current_user), db: Session = Depends(get_session)):
    first = month_param(month)
    last = an.month_end(first)
    q = select(Transaction).where(Transaction.user_id == user.id, Transaction.date >= first,
                                  Transaction.date <= last)
    if category:
        q = q.where(Transaction.category == category)
    if subcategory:
        q = q.where(Transaction.subcategory == subcategory)
    return [crud.tx_out(r) for r in db.scalars(q.order_by(Transaction.date.desc(), Transaction.id.desc()))]


@router.patch("/{tx_id}", response_model=schemas.TransactionOut)
def patch(tx_id: int, body: schemas.TransactionPatch, user: User = Depends(current_user),
          db: Session = Depends(get_session)):
    """A correction: remembered per normalised title and applied to the next imports (category_feedback)."""
    if not tax.is_valid_sub(body.subcategory):
        raise HTTPException(status_code=422, detail="Неизвестная подкатегория")
    row = _own(db, user, tx_id)
    if body.subcategory != row.subcategory or row.category_source != "user":
        crud.add_feedback(db, user.id, row, row.subcategory, body.subcategory)
    row.subcategory = body.subcategory
    row.category = tax.category_of(body.subcategory)
    row.category_source = "user"
    row.confidence = 1.0
    db.commit()
    return crud.tx_out(row)


@router.delete("/{tx_id}", status_code=204)
def delete(tx_id: int, user: User = Depends(current_user), db: Session = Depends(get_session)):
    row = _own(db, user, tx_id)
    for fb in list(db.scalars(select(CategoryFeedback).where(CategoryFeedback.transaction_id == tx_id))):
        fb.transaction_id = None                  # keep the lesson, drop the link to the deleted row
    db.delete(row)
    db.commit()
