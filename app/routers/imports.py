"""Receipt text / CSV preview and confirmation (nothing is saved before /confirm)."""
from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app import crud, schemas
from app.config import get_settings
from app.db import get_session
from app.models import User
from app.routers.deps import current_user, today
from app.services import importing
from app.services.categorizer import get_categorizer
from app.services.csv_import import CsvImportError

router = APIRouter(prefix="/api/import", tags=["import"])


@router.post("/receipt-text")
def receipt_text(body: schemas.ReceiptTextIn, user: User = Depends(current_user), db: Session = Depends(get_session)):
    preview = importing.receipt_preview(body.text, get_categorizer(), crud.feedback_map(db, user.id),
                                        body.date or today())
    return preview


@router.post("/csv")
def csv_file(file: UploadFile = File(...), mapping: Optional[str] = Form(default=None),
                   user: User = Depends(current_user), db: Session = Depends(get_session)):
    limit = get_settings().max_csv_bytes
    data = file.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status_code=413, detail=f"Файл больше {limit // 1_000_000} МБ")
    cols = None
    if mapping:
        try:
            cols = json.loads(mapping)
            if not isinstance(cols, dict):
                raise ValueError
        except ValueError:
            raise HTTPException(status_code=422, detail="mapping должен быть JSON-объектом")
    try:
        return importing.csv_preview(data, get_categorizer(), crud.feedback_map(db, user.id), cols)
    except CsvImportError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.post("/confirm")
def confirm(body: schemas.ConfirmIn, user: User = Depends(current_user), db: Session = Depends(get_session)):
    try:
        return crud.save_confirmed(db, user, body, today())
    except ValueError as e:
        db.rollback()
        raise HTTPException(status_code=422, detail=str(e))
