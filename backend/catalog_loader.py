"""Catalog models for node templates."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field  # type: ignore[import-not-found]


class ArtifactSpec(BaseModel):
    name: str
    path: Optional[str] = None
    role: Optional[str] = None


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
    runtime: Optional[Dict[str, Any]] = None
    argo: Optional[Dict[str, Any]] = None
