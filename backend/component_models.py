"""SQLAlchemy models for the component registry.

The registry is global (no users/tenancy). Each component can have multiple
versions, and one version is marked as the active default.

We store the full ComponentSpec as JSON for forward compatibility.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import DateTime, ForeignKey, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Component(Base):
    __tablename__ = "components"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    active_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    versions: Mapped[list["ComponentVersion"]] = relationship(
        back_populates="component", cascade="all, delete-orphan"
    )


class ComponentVersion(Base):
    __tablename__ = "component_versions"
    __table_args__ = (
        UniqueConstraint("component_id", "version", name="uq_component_version"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    component_id: Mapped[int] = mapped_column(
        ForeignKey("components.id", ondelete="CASCADE")
    )
    version: Mapped[str] = mapped_column(String(64))

    spec_json: Mapped[Dict[str, Any]] = mapped_column(JSONB)
    runtime_image: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    component: Mapped[Component] = relationship(back_populates="versions")
