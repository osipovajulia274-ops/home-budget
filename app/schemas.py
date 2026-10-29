"""Pydantic models of the API (guide 8.2). Previews are plain dicts built by app.services.importing."""
from __future__ import annotations

import datetime as dt
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.services import taxonomy as tax


class ReceiptTextIn(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)
    date: Optional[dt.date] = None            # used when the receipt text has no date


class ConfirmItem(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    amount: float = Field(ge=0, le=100_000_000)
    qty: Optional[float] = Field(default=None, ge=0)
    date: Optional[dt.date] = None            # CSV rows carry their own date
    subcategory: str
    predicted_subcategory: Optional[str] = None   # what the preview showed; a different value = user correction
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    category_source: Literal["model", "user", "rule", "bank"] = "model"


class ConfirmIn(BaseModel):
    source: Literal["text", "csv"]
    store: Optional[str] = Field(default=None, max_length=200)
    purchased_at: Optional[dt.datetime] = None
    total: Optional[float] = None
    sum_matches_total: Optional[bool] = None
    items: list[ConfirmItem] = Field(min_length=1, max_length=2000)


class TransactionIn(BaseModel):
    date: dt.date
    title: str = Field(min_length=1, max_length=300)
    amount: float = Field(gt=0, le=100_000_000)
    qty: Optional[float] = Field(default=None, ge=0)
    subcategory: Optional[str] = None         # empty -> the model decides


class TransactionPatch(BaseModel):
    subcategory: str


class TransactionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    date: dt.date
    title: str
    amount: float
    qty: Optional[float] = None
    subcategory: str
    category: str
    confidence: Optional[float] = None
    category_source: str
    receipt_id: Optional[int] = None
    is_recurring: bool = False


class GoalIn(BaseModel):
    scope: str = Field(min_length=1, max_length=60)
    limit_amount: float = Field(gt=0, le=100_000_000)


class GoalOut(BaseModel):
    id: int
    scope: str
    title: str
    limit_amount: float


class MeIn(BaseModel):
    monthly_income: Optional[float] = Field(default=None, ge=0, le=1_000_000_000)


def valid_scope(scope: str) -> bool:
    return scope == "total" or scope in tax.category_titles() or tax.is_valid_sub(scope)
