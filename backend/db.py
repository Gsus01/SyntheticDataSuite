"""Database helpers for the backend service.

The component registry is stored in PostgreSQL.

We keep the schema lightweight and create tables on startup (when enabled)
instead of introducing Alembic immediately. This keeps the repo simpler while
we iterate on the registry model.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)

_ENGINE: Engine | None = None
SessionLocal: sessionmaker | None = None


def _get_database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Example: postgresql+psycopg://user:pass@host:5432/db"
        )
    return url


def get_engine() -> Engine:
    global _ENGINE
    if _ENGINE is None:
        # pool_pre_ping helps in dev where port-forwards bounce.
        _ENGINE = create_engine(_get_database_url(), pool_pre_ping=True)
    return _ENGINE


def get_session_factory() -> sessionmaker:
    global SessionLocal
    if SessionLocal is None:
        SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=get_engine()
        )
    return SessionLocal


@contextmanager
def db_session() -> Iterator[Session]:
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def should_run_migrations() -> bool:
    value = os.getenv("RUN_MIGRATIONS", "").strip().lower()
    return value in {"1", "true", "yes", "on"}
