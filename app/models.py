"""ORM tables (guide 8.1) with the Stage-2 change: `budgets` -> `goals` (monthly limit, no forecast)."""
from __future__ import annotations

import datetime as dt
import uuid
from decimal import Decimal
from typing import Optional

from sqlalchemy import (JSON, Boolean, CheckConstraint, Date, DateTime, Float, ForeignKey, Index, Integer, Numeric,
                        String, Text, UniqueConstraint, Uuid)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base

JsonType = JSON().with_variant(JSONB(), "postgresql")


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


class User(Base):
    __tablename__ = "users"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_now)
    monthly_income: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 2), nullable=True)
    currency: Mapped[str] = mapped_column(String(3), default="RUB")


class Receipt(Base):
    __tablename__ = "receipts"
    __table_args__ = (CheckConstraint("source in ('text','csv','manual','photo')", name="ck_receipts_source"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    source: Mapped[str] = mapped_column(String(10))
    store: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    purchased_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime, nullable=True)
    total: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 2), nullable=True)
    raw_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)      # never filled: it would hold INN/names
    sum_matches_total: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_now)


class Transaction(Base):
    __tablename__ = "transactions"
    __table_args__ = (
        CheckConstraint("amount >= 0", name="ck_transactions_amount"),
        CheckConstraint("category_source in ('model','user','rule','bank')", name="ck_transactions_source"),
        Index("ix_transactions_user_date", "user_id", "date"),
        Index("ix_transactions_user_category", "user_id", "category"),
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    receipt_id: Mapped[Optional[int]] = mapped_column(ForeignKey("receipts.id"), nullable=True)
    date: Mapped[dt.date] = mapped_column(Date)
    title: Mapped[str] = mapped_column(String(300))
    title_norm: Mapped[str] = mapped_column(String(300))
    amount: Mapped[Decimal] = mapped_column(Numeric(12, 2))
    qty: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 3), nullable=True)
    subcategory: Mapped[str] = mapped_column(String(40))
    category: Mapped[str] = mapped_column(String(30))
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    category_source: Mapped[str] = mapped_column(String(10), default="model")
    is_recurring: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_now)


class CategoryFeedback(Base):
    __tablename__ = "category_feedback"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    transaction_id: Mapped[Optional[int]] = mapped_column(ForeignKey("transactions.id"), nullable=True)
    title_norm: Mapped[str] = mapped_column(String(300))
    old_subcategory: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    new_subcategory: Mapped[str] = mapped_column(String(40))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_now)


class Goal(Base):
    """A monthly limit: `scope` is "total", a category ("food") or a subcategory ("eating_out.coffee")."""
    __tablename__ = "goals"
    __table_args__ = (UniqueConstraint("user_id", "scope", name="uq_goals_user_scope"),
                      CheckConstraint("limit_amount > 0", name="ck_goals_limit"))
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    scope: Mapped[str] = mapped_column(String(60))
    limit_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_now)


class Recommendation(Base):
    __tablename__ = "recommendations"
    __table_args__ = (CheckConstraint("severity in ('info','warning','alert')", name="ck_reco_severity"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    rule_id: Mapped[str] = mapped_column(String(60))
    category: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    severity: Mapped[str] = mapped_column(String(10))
    title: Mapped[str] = mapped_column(Text)
    body: Mapped[str] = mapped_column(Text)
    action: Mapped[Optional[str]] = mapped_column(Text, nullable=True)          # added to the guide's schema
    payload: Mapped[Optional[dict]] = mapped_column(JsonType, nullable=True)    # includes the card's "key"
    month: Mapped[dt.date] = mapped_column(Date)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_now)
    dismissed_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime, nullable=True)
