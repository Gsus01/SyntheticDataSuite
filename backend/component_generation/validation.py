from __future__ import annotations

from typing import Any, Mapping

from component_spec import ComponentSpec


def _type_error_message(component_type: Any) -> str | None:
    if component_type == "input":
        return "input components are not allowed for generated components (use the built-in input node)"
    if component_type == "output":
        return "output components are not allowed for generated components (use the built-in output node)"
    return None


def validate_generated_component_plan(component: Mapping[str, Any]) -> None:
    name = str(component.get("name") or "component")
    message = _type_error_message(component.get("type"))
    if message:
        raise ValueError(f"{name}: {message}")


def validate_generated_componentspec(
    spec: ComponentSpec, *, component_name: str | None = None
) -> None:
    name = component_name or spec.metadata.name
    message = _type_error_message(spec.metadata.type)
    if message:
        raise ValueError(f"{name}: {message}")
