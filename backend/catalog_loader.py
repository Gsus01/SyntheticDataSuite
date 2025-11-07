"""Catalog models and helpers for node templates."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field  # type: ignore[import-not-found]


logger = logging.getLogger(__name__)


CATALOG_PATH = Path(__file__).parent / "catalog" / "nodes.yaml"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ArtifactSpec(BaseModel):
    name: str
    path: Optional[str] = None


class Artifacts(BaseModel):
    inputs: List[ArtifactSpec] = Field(default_factory=list)
    outputs: List[ArtifactSpec] = Field(default_factory=list)


class NodeTemplate(BaseModel):
    name: str
    type: str
    parameters: List[str] = Field(default_factory=list)
    artifacts: Artifacts
    limits: Optional[Dict[str, Any]] = None
    version: Optional[str] = None
    parameter_defaults: Optional[Dict[str, Any]] = None


def load_catalog() -> List[NodeTemplate]:
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(f"Catalog file not found at {CATALOG_PATH}")
    try:
        with CATALOG_PATH.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    except Exception as exc:  # pragma: no cover - surface upstream
        raise RuntimeError(f"Failed to read catalog: {exc}")

    nodes = raw.get("nodes", [])
    templates: List[NodeTemplate] = []

    for node in nodes:
        node_data = dict(node)
        param_defaults = None
        params_file = node_data.pop("parameters_file", None)

        if params_file:
            params_path = Path(params_file)
            if not params_path.is_absolute():
                params_path = PROJECT_ROOT / params_path
            try:
                with params_path.open("r", encoding="utf-8") as pf:
                    param_defaults = json.load(pf)
            except FileNotFoundError:
                logger.warning(
                    "Parameters file not found for node '%s': %s",
                    node_data.get("name"),
                    params_path,
                )
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Invalid JSON in parameters file for node '%s': %s (%s)",
                    node_data.get("name"),
                    params_path,
                    exc,
                )
            except Exception as exc:  # pragma: no cover - logging branch
                logger.warning(
                    "Failed to load parameters file for node '%s': %s (%s)",
                    node_data.get("name"),
                    params_path,
                    exc,
                )

        if param_defaults is not None:
            node_data["parameter_defaults"] = param_defaults

        try:
            templates.append(NodeTemplate(**node_data))
        except Exception as exc:  # pragma: no cover - validation error surfaces upstream
            raise RuntimeError(
                f"Invalid catalog format for node '{node_data.get('name')}': {exc}"
            )

    return templates


