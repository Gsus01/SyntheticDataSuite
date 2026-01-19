"""Component registry backed by PostgreSQL.

- Global registry (no users).
- Supports multiple versions per component.
- One active version per component (default/latest).

We intentionally store the full validated ComponentSpec in a JSONB column so
we can evolve the schema without heavy migrations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from component_models import Base, Component, ComponentVersion
from component_spec import ComponentSpec


def ensure_tables(engine) -> None:
    Base.metadata.create_all(bind=engine)


def _spec_to_row(spec: ComponentSpec) -> Dict[str, Any]:
    payload = spec.model_dump(by_alias=True)
    runtime_image = None
    if spec.runtime:
        runtime_image = spec.runtime.image
    return {
        "payload": payload,
        "runtime_image": runtime_image,
    }


class ComponentRegistry:
    def __init__(self, session: Session):
        self.session = session

    def list_components(self) -> List[Component]:
        stmt = select(Component).order_by(Component.name.asc())
        return list(self.session.scalars(stmt).all())

    def get_component(self, name: str) -> Optional[Component]:
        stmt = select(Component).where(Component.name == name)
        return self.session.scalars(stmt).first()

    def list_versions(self, name: str) -> List[ComponentVersion]:
        component = self.get_component(name)
        if not component:
            return []
        stmt = (
            select(ComponentVersion)
            .where(ComponentVersion.component_id == component.id)
            .order_by(ComponentVersion.created_at.desc())
        )
        return list(self.session.scalars(stmt).all())

    def get_version(self, name: str, version: str) -> Optional[ComponentVersion]:
        component = self.get_component(name)
        if not component:
            return None
        stmt = select(ComponentVersion).where(
            ComponentVersion.component_id == component.id,
            ComponentVersion.version == version,
        )
        return self.session.scalars(stmt).first()

    def resolve_active_spec(self, name: str) -> Optional[ComponentSpec]:
        component = self.get_component(name)
        if not component or not component.active_version:
            return None
        row = self.get_version(name, component.active_version)
        if not row:
            return None
        return ComponentSpec.model_validate(row.spec_json)

    def register(self, spec: ComponentSpec, *, activate: bool = True) -> Component:
        name = spec.metadata.name
        version = spec.metadata.version

        component = self.get_component(name)
        if not component:
            component = Component(name=name, active_version=None)
            self.session.add(component)
            self.session.flush()

        existing = self.get_version(name, version)
        if existing:
            # Overwrite is allowed (global registry, no users). Keep it simple.
            row_data = _spec_to_row(spec)
            existing.spec_json = row_data["payload"]
            existing.runtime_image = row_data["runtime_image"]
        else:
            row_data = _spec_to_row(spec)
            new_row = ComponentVersion(
                component_id=component.id,
                version=version,
                spec_json=row_data["payload"],
                runtime_image=row_data["runtime_image"],
            )
            self.session.add(new_row)

        if activate:
            component.active_version = version

        self.session.flush()
        return component

    def activate(self, name: str, version: str) -> Optional[Component]:
        component = self.get_component(name)
        if not component:
            return None
        row = self.get_version(name, version)
        if not row:
            return None
        component.active_version = version
        self.session.flush()
        return component
