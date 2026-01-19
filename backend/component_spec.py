"""Pydantic models for the component registry spec.

This is intentionally minimal for v1. We store the full validated spec in DB
and derive the UI catalog and Argo templates from it.

Design goals:
- Forward compatible: allow unknown extra fields.
- Practical: only model what we need to run containers + connect artifacts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


ComponentType = Literal[
    "input",
    "preprocessing",
    "training",
    "generation",
    "output",
    "other",
]

ArtifactRole = Literal[
    "data",
    "model",
    "config",
    "metrics",
    "other",
]


class ArtifactPort(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    path: Optional[str] = None
    role: ArtifactRole = "data"


class ComponentIO(BaseModel):
    model_config = ConfigDict(extra="allow")

    inputs: List[ArtifactPort] = Field(default_factory=list)
    outputs: List[ArtifactPort] = Field(default_factory=list)


class ComponentRuntime(BaseModel):
    model_config = ConfigDict(extra="allow")

    image: str
    image_pull_policy: Optional[str] = Field(None, alias="imagePullPolicy")
    command: Optional[List[str]] = None
    args: Optional[List[str]] = None


class ComponentParameters(BaseModel):
    model_config = ConfigDict(extra="allow")

    defaults: Optional[Dict[str, Any]] = None
    schema_: Optional[Dict[str, Any]] = Field(None, alias="schema")


class ComponentMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    version: str
    title: Optional[str] = None
    description: Optional[str] = None
    type: ComponentType = "other"


class ComponentSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    api_version: str = Field("sds/v1", alias="apiVersion")
    kind: str = "Component"

    metadata: ComponentMetadata
    io: ComponentIO
    runtime: Optional[ComponentRuntime] = None
    parameters: Optional[ComponentParameters] = None
    argo: Optional[Dict[str, Any]] = None
