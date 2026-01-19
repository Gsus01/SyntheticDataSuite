"""Adapter: registry ComponentSpec -> legacy NodeTemplate shape.

The frontend currently expects the catalog at GET /workflow-templates to return
NodeTemplate objects with this shape:
- name, type, artifacts.{inputs,outputs}, parameter_defaults

We keep the response stable while migrating internals to the DB-backed registry.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from catalog_loader import ArtifactSpec, Artifacts, NodeTemplate
from component_spec import ComponentSpec


def component_to_node_template(spec: ComponentSpec) -> NodeTemplate:
    inputs = [
        ArtifactSpec(name=p.name, path=p.path, role=p.role) for p in spec.io.inputs
    ]
    outputs = [
        ArtifactSpec(name=p.name, path=p.path, role=p.role) for p in spec.io.outputs
    ]

    defaults: Optional[Dict[str, Any]] = None
    if spec.parameters and isinstance(spec.parameters.defaults, dict):
        defaults = spec.parameters.defaults

    return NodeTemplate(
        name=spec.metadata.name,
        type=spec.metadata.type,
        version=spec.metadata.version,
        parameters=list(defaults.keys()) if defaults else [],
        artifacts=Artifacts(inputs=inputs, outputs=outputs),
        limits={},
        parameter_defaults=defaults,
        runtime=spec.runtime.model_dump(by_alias=True) if spec.runtime else None,
        argo=spec.argo,
    )


def components_to_catalog(specs: List[ComponentSpec]) -> List[NodeTemplate]:
    return [component_to_node_template(spec) for spec in specs]
