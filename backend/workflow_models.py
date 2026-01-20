"""SQLAlchemy models for stored workflow definitions."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class WorkflowBase(DeclarativeBase):
    pass


class WorkflowRecordDB(WorkflowBase):
    __tablename__ = "workflows"

    workflow_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    session_id: Mapped[str] = mapped_column(String(128))

    nodes: Mapped[list[Dict[str, Any]]] = mapped_column(JSONB)
    edges: Mapped[list[Dict[str, Any]]] = mapped_column(JSONB)
    node_slug_map: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)

    manifest_filename: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    compiled_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    last_workflow_name: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )
    last_namespace: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    last_bucket: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    last_key: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    last_manifest_filename: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )
    last_cli_output: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    last_submitted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    version: Mapped[int] = mapped_column(Integer, default=1)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


def ensure_workflow_tables(engine) -> None:
    WorkflowBase.metadata.create_all(bind=engine)
