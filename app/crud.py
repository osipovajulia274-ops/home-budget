"""Database access. Thin on purpose: all business logic lives in app/services (pure Python, unit-tested)."""
from __future__ import annotations

import datetime as dt
import uuid
from decimal import Decimal
from typing import Dict, List, Optional, Sequence

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app import models as m
from app.services import analytics as an
from app.services import demo as demo_mod
from app.services import rules
from app.services import taxonomy as tax
from app.services.categorizer import Categorizer
from app.services.goals import Goal, GoalStatus, evaluate_goals
from app.services.normalize import normalize


def _f(x) -> Optional[float]:
    return None if x is None else float(x)


# ------------------------------------------------------------------------------ users ----

def get_user(db: Session, raw_id: Optional[str]) -> Optional[m.User]:
    if not raw_id:
        return None
    try:
        return db.get(m.User, uuid.UUID(raw_id))
    except ValueError:
        return None


def create_user(db: Session) -> m.User:
    user = m.User()
    db.add(user)
    db.commit()
    return user


def delete_user_data(db: Session, user: m.User) -> None:
    """Remove every row of the user (FK-safe order, no reliance on ON DELETE CASCADE) and the user itself."""
    uid = user.id
    for model in (m.CategoryFeedback, m.Recommendation, m.Goal, m.Transaction, m.Receipt):
        db.execute(delete(model).where(model.user_id == uid))
    db.execute(delete(m.User).where(m.User.id == uid))
    db.commit()


# ------------------------------------------------------------------------ transactions ----

def to_tx(row: m.Transaction) -> an.Tx:
    return an.Tx(id=row.id, date=row.date, title=row.title, title_norm=row.title_norm, amount=float(row.amount),
                 subcategory=row.subcategory, qty=_f(row.qty), receipt_id=row.receipt_id,
                 category_source=row.category_source)


def tx_out(row: m.Transaction) -> dict:
    """API shape of a transaction (money as float, explicit so that no implicit Decimal conversion is needed)."""
    return {"id": row.id, "date": row.date, "title": row.title, "amount": float(row.amount), "qty": _f(row.qty),
            "subcategory": row.subcategory, "category": row.category, "confidence": row.confidence,
            "category_source": row.category_source, "receipt_id": row.receipt_id,
            "is_recurring": bool(row.is_recurring)}


def all_txs(db: Session, user_id) -> List[an.Tx]:
    rows = db.scalars(select(m.Transaction).where(m.Transaction.user_id == user_id).order_by(m.Transaction.date))
    return [to_tx(r) for r in rows]


def feedback_map(db: Session, user_id) -> Dict[str, str]:
    """normalised title -> the user's latest subcategory for it."""
    out: Dict[str, str] = {}
    rows = db.execute(select(m.CategoryFeedback.title_norm, m.CategoryFeedback.new_subcategory)
                      .where(m.CategoryFeedback.user_id == user_id).order_by(m.CategoryFeedback.id))
    for title_norm, sub in rows:
        out[title_norm] = sub
    return out


def new_transaction(user_id, *, date: dt.date, title: str, amount: float, subcategory: str, qty: Optional[float],
                    confidence: Optional[float], source: str, receipt_id: Optional[int] = None) -> m.Transaction:
    return m.Transaction(user_id=user_id, receipt_id=receipt_id, date=date, title=title.strip(),
                         title_norm=normalize(title), amount=Decimal(str(round(amount, 2))),
                         qty=None if qty is None else Decimal(str(qty)), subcategory=subcategory,
                         category=tax.category_of(subcategory), confidence=confidence, category_source=source)


def add_feedback(db: Session, user_id, tx: m.Transaction, old: Optional[str], new: str) -> None:
    db.add(m.CategoryFeedback(user_id=user_id, transaction_id=tx.id, title_norm=tx.title_norm,
                              old_subcategory=old, new_subcategory=new))


def save_confirmed(db: Session, user: m.User, body, today: dt.date) -> dict:
    """Store a confirmed preview. Items whose category differs from the preview's become feedback."""
    receipt_id = None
    purchased_at = body.purchased_at.replace(tzinfo=None) if body.purchased_at else None   # columns are naive
    default_date = purchased_at.date() if purchased_at else today
    if body.source == "text":
        receipt = m.Receipt(user_id=user.id, source="text", store=body.store, purchased_at=purchased_at,
                            total=None if body.total is None else Decimal(str(body.total)),
                            sum_matches_total=body.sum_matches_total)
        db.add(receipt)
        db.flush()
        receipt_id = receipt.id
    rows: List[m.Transaction] = []
    corrected: List[tuple] = []
    for it in body.items:
        if not tax.is_valid_sub(it.subcategory):
            raise ValueError(f"Неизвестная подкатегория: {it.subcategory}")
        user_changed = it.predicted_subcategory is not None and it.predicted_subcategory != it.subcategory
        source = "user" if user_changed else it.category_source
        row = new_transaction(user.id, date=it.date or default_date, title=it.title, amount=it.amount,
                              subcategory=it.subcategory, qty=it.qty,
                              confidence=1.0 if user_changed else it.confidence, source=source,
                              receipt_id=receipt_id)
        db.add(row)
        rows.append(row)
        if user_changed:
            corrected.append((row, it.predicted_subcategory, it.subcategory))
    db.flush()                                   # ids for the feedback rows
    for row, old, new in corrected:
        add_feedback(db, user.id, row, old, new)
    db.commit()
    return {"receipt_id": receipt_id, "saved": len(rows)}


def load_demo(db: Session, user: m.User, today: dt.date) -> int:
    n = 0
    for group in demo_mod.generate(today):
        receipt_id = None
        if group["receipt"]:
            r = group["receipt"]
            rec = m.Receipt(user_id=user.id, source="manual", store=r["store"], purchased_at=r["purchased_at"],
                            total=Decimal(str(r["total"])), sum_matches_total=True)
            db.add(rec)
            db.flush()
            receipt_id = rec.id
        for it in group["items"]:
            db.add(new_transaction(user.id, date=it["date"], title=it["title"], amount=it["amount"],
                                   subcategory=it["subcategory"], qty=it["qty"], confidence=None, source="rule",
                                   receipt_id=receipt_id))
            n += 1
    if user.monthly_income is None:
        user.monthly_income = Decimal("120000")
    db.commit()
    return n


# ------------------------------------------------------------------------------ goals ----

def goals_of(db: Session, user_id) -> List[m.Goal]:
    return list(db.scalars(select(m.Goal).where(m.Goal.user_id == user_id).order_by(m.Goal.id)))


def goal_statuses(db: Session, user_id, spend: Sequence[an.Tx], first: dt.date, today: dt.date) -> List[GoalStatus]:
    goals = [Goal(g.scope, float(g.limit_amount)) for g in goals_of(db, user_id)]
    month_spend = an.spend_by_sub(t for t in an.in_month(spend, first) if t.date <= an.as_of_for(first, today))
    return evaluate_goals(goals, month_spend, an.as_of_for(first, today))


# ------------------------------------------------------------------- recommendations ----

def refresh_recommendations(db: Session, user: m.User, first: dt.date, today: dt.date) -> List[m.Recommendation]:
    """Recompute the cards, keep the user's "hide" decisions for this month, return the active ones (max 5)."""
    spend = an.spending(all_txs(db, user.id))
    statuses = goal_statuses(db, user.id, spend, first, today)
    ctx = rules.UserCtx(today=today, month=first, txs=spend, goal_statuses=statuses, income=_f(user.monthly_income))
    cards = rules.run_rules(ctx)
    order = {c["payload"]["key"]: i for i, c in enumerate(cards)}

    existing = list(db.scalars(select(m.Recommendation).where(m.Recommendation.user_id == user.id,
                                                              m.Recommendation.month == first)))
    by_key = {(r.payload or {}).get("key"): r for r in existing}
    for c in cards:
        key = c["payload"]["key"]
        row = by_key.get(key)
        if row is None:
            row = m.Recommendation(user_id=user.id, month=first, rule_id=c["rule_id"])
            db.add(row)
        row.rule_id, row.category, row.severity = c["rule_id"], c["category"], c["severity"]
        row.title, row.body, row.action, row.payload = c["title"], c["body"], c["action"], c["payload"]
    for key, row in by_key.items():
        if key not in order and row.dismissed_at is None:
            db.delete(row)                         # the condition no longer holds; dismissed rows stay hidden
    db.commit()
    rows = [r for r in db.scalars(select(m.Recommendation).where(m.Recommendation.user_id == user.id,
                                                                 m.Recommendation.month == first))
            if r.dismissed_at is None and (r.payload or {}).get("key") in order]
    rows.sort(key=lambda r: order[r.payload["key"]])
    return rows[:rules.MAX_ACTIVE]


def mark_recurring(db: Session, user_id, today: dt.date) -> List[an.Recurring]:
    spend = an.spending(all_txs(db, user_id))
    found = an.detect_recurring(spend, today)
    ids = {i for r in found for i in r.tx_ids}
    for row in list(db.scalars(select(m.Transaction).where(m.Transaction.user_id == user_id))):
        flag = row.id in ids
        if row.is_recurring != flag:
            row.is_recurring = flag
    db.commit()
    return found
