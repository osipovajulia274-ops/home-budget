"""Receipt text / bank CSV -> preview dicts (nothing is saved here). Shared by the import endpoints and tests."""
from __future__ import annotations

import datetime as dt
from typing import Dict, List, Mapping, Optional

from app.services import bank_maps
from app.services import taxonomy as tax
from app.services.categorizer import Categorizer, Prediction
from app.services.csv_import import CsvRow, Mapping as ColMapping, parse_csv
from app.services.receipt_parser import parse_receipt


def _item_dict(pred: Prediction, **base) -> dict:
    return {**base, "subcategory": pred.subcategory, "category": pred.category,
            "confidence": pred.confidence, "category_source": pred.source, "needs_review": pred.needs_review,
            "alternatives": [{"subcategory": s, "p": p} for s, p in pred.alternatives]}


def receipt_preview(text: str, categorizer: Categorizer, feedback: Optional[Mapping[str, str]] = None,
                    default_date: Optional[dt.date] = None) -> dict:
    parsed = parse_receipt(text, default_date)
    preds = categorizer.categorize([i.title for i in parsed.items], feedback)
    items = [_item_dict(p, title=i.title, qty=i.qty, price=i.price, amount=i.amount)
             for i, p in zip(parsed.items, preds)]
    return {
        "store": parsed.store,
        "purchased_at": parsed.purchased_at.isoformat() if parsed.purchased_at else None,
        "total": parsed.total, "items_sum": parsed.items_sum, "sum_matches_total": parsed.sum_matches_total,
        "items": items, "warnings": parsed.warnings,
    }


def _csv_prediction(row: CsvRow, pred: Prediction) -> Prediction:
    if pred.source == "user":
        return pred
    if row.is_transfer:
        return Prediction(tax.TRANSFERS, tax.category_of(tax.TRANSFERS), 1.0, "rule")
    if pred.subcategory == tax.UNKNOWN:                      # model unsure: the bank's own hint, if any
        sub = bank_maps.from_mcc(row.mcc) or bank_maps.from_bank_category(row.bank_category)   # MCC is more specific
        if sub and tax.is_valid_sub(sub):
            return Prediction(sub, tax.category_of(sub), pred.confidence, "bank", pred.alternatives)
    return pred


def csv_preview(data: bytes, categorizer: Categorizer, feedback: Optional[Mapping[str, str]] = None,
                mapping: Optional[ColMapping] = None) -> dict:
    parsed = parse_csv(data, mapping)
    preds = categorizer.categorize([r.title for r in parsed.rows], feedback)
    items = []
    for row, pred in zip(parsed.rows, preds):
        pred = _csv_prediction(row, pred)
        items.append(_item_dict(pred, title=row.title, qty=1.0, price=row.amount, amount=row.amount,
                                date=row.date.isoformat(), bank_category=row.bank_category, mcc=row.mcc))
    spend = round(sum(i["amount"] for i in items if i["subcategory"] != tax.TRANSFERS), 2)
    return {
        "encoding": parsed.encoding, "delimiter": parsed.delimiter, "columns": parsed.columns,
        "mapping": parsed.mapping, "items": items, "items_sum": spend,
        "income_total": parsed.income_total, "income_count": parsed.income_count,
        "skipped": parsed.skipped, "warnings": parsed.warnings,
    }
