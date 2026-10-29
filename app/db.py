"""Engine and sessions. Created lazily so that tests can set DATABASE_URL first."""
from __future__ import annotations

from typing import Iterator, Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import get_settings


class Base(DeclarativeBase):
    pass


_engine: Optional[Engine] = None
_factory: Optional[sessionmaker] = None


def get_engine() -> Engine:
    global _engine, _factory
    if _engine is None:
        url = get_settings().db_url
        kwargs: dict = {"pool_pre_ping": True}          # Render's free Postgres drops idle connections
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
            if url in ("sqlite://", "sqlite:///:memory:"):
                kwargs["poolclass"] = StaticPool        # one shared in-memory database for the tests
        _engine = create_engine(url, **kwargs)
        _factory = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def init_db() -> None:
    """`create_all` at start-up (no Alembic within the two-day scope)."""
    from app import models  # noqa: F401  (registers the tables)

    Base.metadata.create_all(get_engine())


def reset_engine() -> None:
    """For tests: forget the engine so the next call builds one from the current settings."""
    global _engine, _factory
    if _engine is not None:
        _engine.dispose()
    _engine = _factory = None


def get_session() -> Iterator[Session]:
    get_engine()
    assert _factory is not None
    with _factory() as session:
        yield session
