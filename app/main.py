"""FastAPI application: JSON API (Stage 3) and HTML screens (Stage 4, routers/pages.py)."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.db import init_db
from app.routers import goals, imports, me, pages, recommendations, summary, transactions
from app.services.categorizer import get_categorizer

log = logging.getLogger("home-budget")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    try:
        get_categorizer()                          # load the model once, before the first request
    except Exception:                              # the app still starts; /health reports the problem
        log.exception("classifier failed to load")
    yield


app = FastAPI(title="Домашняя бухгалтерия", version="0.3.0", lifespan=lifespan,
              description="Демо-режим: данные привязаны к браузеру. Не финансовая консультация.")

for r in (imports.router, transactions.router, summary.router, goals.router, recommendations.router, me.router,
          pages.router):
    app.include_router(r)

app.mount("/static", StaticFiles(directory=str(Path(__file__).resolve().parent / "static")), name="static")


@app.get("/health", tags=["service"])
def health():
    try:
        c = get_categorizer()
    except Exception as e:                         # noqa: BLE001
        return JSONResponse(status_code=503, content={"status": "error", "models": {"classifier": False},
                                                      "detail": type(e).__name__})
    return {"status": "ok", "models": {"classifier": True, "kind": c.kind}}
