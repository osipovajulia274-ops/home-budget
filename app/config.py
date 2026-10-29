"""Settings from the environment (a `.env` file is read in dev). Production: DATABASE_URL comes from Render.

Plain dataclass instead of pydantic-settings: four variables do not need another dependency.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = Path(__file__).resolve().parent / "ml_artifacts"
DATA_DIR = ROOT / "data"


def normalize_db_url(url: str) -> str:
    """Render hands out `postgres://...`; SQLAlchemy 2 + psycopg 3 needs `postgresql+psycopg://...`."""
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


def _read_dotenv(path: Path) -> dict:
    out: dict = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


@dataclass(frozen=True)
class Settings:
    database_url: str = "sqlite:///./dev.db"     # dev default; Render sets a Postgres URL
    app_env: str = "dev"                         # dev | prod | test
    timezone: str = "Europe/Moscow"              # the day boundary for "today" and for demo data
    max_csv_bytes: int = 2_000_000
    max_receipt_chars: int = 20_000

    @property
    def db_url(self) -> str:
        return normalize_db_url(self.database_url)

    @property
    def is_prod(self) -> bool:
        return self.app_env == "prod"

    @classmethod
    def from_env(cls) -> "Settings":
        env = {**_read_dotenv(ROOT / ".env"), **os.environ}   # real environment wins over .env
        return cls(
            database_url=env.get("DATABASE_URL", cls.database_url),
            app_env=env.get("APP_ENV", cls.app_env),
            timezone=env.get("APP_TIMEZONE", cls.timezone),
        )


@lru_cache
def get_settings() -> Settings:
    return Settings.from_env()
