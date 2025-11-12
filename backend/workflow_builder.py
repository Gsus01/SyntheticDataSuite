"""Build an Argo workflow plan from a React Flow graph."""

from __future__ import annotations

import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import os

import yaml
from pydantic import BaseModel, Field  # type: ignore[import-not-found]

from minio_helper import (
    build_session_node_prefix,
    get_input_bucket,
    get_minio_client,
    sanitize_path_segment,
    upload_bytes,
)
from catalog_loader import ArtifactSpec, NodeTemplate, load_catalog


SERVICE_ACCOUNT_NAME = os.getenv("ARGO_SERVICE_ACCOUNT", "default")
_LABEL_MAX_LEN = 63


def _sanitize_label_value(value: str, fallback: str) -> str:
    candidate = sanitize_path_segment(value, fallback)
    if len(candidate) > _LABEL_MAX_LEN:
        trimmed = candidate[:_LABEL_MAX_LEN].rstrip("-._")
        candidate = trimmed or candidate[:_LABEL_MAX_LEN]
    candidate = candidate or sanitize_path_segment(fallback, fallback)
    return candidate


_TEMPLATE_REGISTRY_PATH = Path(__file__).resolve().parent / "workflow-templates.yaml"
if not _TEMPLATE_REGISTRY_PATH.exists():
    raise FileNotFoundError(
        f"Workflow templates registry not found at {_TEMPLATE_REGISTRY_PATH}"
    )
with _TEMPLATE_REGISTRY_PATH.open("r", encoding="utf-8") as fh:
    _raw_registry = yaml.safe_load(fh) or {}
_WORKFLOW_TEMPLATE_REGISTRY: Dict[str, Dict[str, Any]] = {
    name: body for name, body in (_raw_registry.get("templates") or {}).items()
}


class NodeArtifactPayload(BaseModel):
    bucket: Optional[str] = None
    key: Optional[str] = None
    size: Optional[int] = None
    contentType: Optional[str] = None
    originalFilename: Optional[str] = None


class FlowNodeDataPayload(BaseModel):
    label: str
    templateName: Optional[str] = None
    tone: Optional[str] = None
    parameterKeys: Optional[List[str]] = None
    parameterDefaults: Optional[Dict[str, object]] = None
    parameters: Optional[Dict[str, object]] = None
    uploadedArtifact: Optional[NodeArtifactPayload] = None


class FlowNodePayload(BaseModel):
    id: str
    type: str
    data: FlowNodeDataPayload


class FlowEdgePayload(BaseModel):
    id: Optional[str] = None
    source: str
    target: str


class WorkflowGraphPayload(BaseModel):
    session_id: str = Field(..., alias="sessionId")
    nodes: List[FlowNodePayload]
    edges: List[FlowEdgePayload]

    class Config:
        allow_population_by_field_name = True


class ArtifactPlan(BaseModel):
    name: str
    path: Optional[str]
    bucket: str
    key: str
    workflow_input_name: Optional[str] = None


class ArtifactBinding(BaseModel):
    input_name: str
    source_node_id: Optional[str]
    source_artifact_name: str
    bucket: str
    key: str
    workflow_input_name: Optional[str] = None


class NodePlan(BaseModel):
    node_id: str
    slug: str
    template: NodeTemplate
    is_input: bool
    is_output: bool = False
    outputs: Dict[str, ArtifactPlan]
    input_bindings: Dict[str, ArtifactBinding] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)
    config_inputs: Dict[str, ArtifactPlan] = Field(default_factory=dict)

    @property
    def template_name(self) -> str:
        return self.template.name


class WorkflowPlan(BaseModel):
    session_id: str
    bucket: str
    nodes: Dict[str, NodePlan]

    def tasks(self) -> List[NodePlan]:
        return [plan for plan in self.nodes.values() if not plan.is_input and not plan.is_output]


class WorkflowBuilderError(RuntimeError):
    """Raised when workflow plan generation fails."""


class UnknownTemplateError(WorkflowBuilderError):
    pass


class MissingArtifactError(WorkflowBuilderError):
    pass


def _index_catalog() -> Dict[str, NodeTemplate]:
    catalog = load_catalog()
    return {template.name: template for template in catalog}


def _slug_for_node(node: FlowNodePayload) -> str:
    data = node.data
    preferred = data.templateName or data.label or node.id
    return sanitize_path_segment(preferred, node.id)


def _artifact_suffix(path: Optional[str]) -> str:
    if not path:
        return ""
    suffix = Path(path).suffix
    return suffix or ""


def _ensure_uploaded_artifact(node: FlowNodePayload) -> NodeArtifactPayload:
    artifact = node.data.uploadedArtifact
    if not artifact or not artifact.key:
        raise MissingArtifactError(
            f"Input node '{node.id}' is missing uploaded artifact information"
        )
    if not artifact.bucket:
        artifact.bucket = get_input_bucket()
    return artifact


def _is_config_artifact(spec: ArtifactSpec) -> bool:
    name = spec.name.lower()
    path = (spec.path or "").lower()
    if name == "processed-data":
        return False
    if "config" in name:
        return True
    if "/config/" in path:
        return True
    if path.endswith(('.yaml', '.yml', '.json')):
        return True
    return False


def _merge_parameters(defaults: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(defaults)

    def merge(dest: Dict[str, Any], source: Dict[str, Any]) -> None:
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(dest.get(key), dict):
                merge(dest[key], value)
            else:
                dest[key] = value

    merge(result, overrides)
    return result


def _build_config_document(node: FlowNodePayload) -> Dict[str, Any]:
    defaults = node.data.parameterDefaults
    overrides = node.data.parameters

    defaults_dict = deepcopy(defaults) if isinstance(defaults, dict) else {}
    overrides_dict = deepcopy(overrides) if isinstance(overrides, dict) else {}

    if not defaults_dict and not overrides_dict:
        return {}
    return _merge_parameters(defaults_dict, overrides_dict)


def _render_config_bytes(config: Dict[str, Any], spec: ArtifactSpec) -> tuple[bytes, str, str]:
    target_path = (spec.path or "").lower()
    if target_path.endswith(('.yaml', '.yml')):
        rendered = yaml.safe_dump(config, sort_keys=False)
        extension = '.yaml' if target_path.endswith('.yaml') else '.yml'
        content_type = 'application/x-yaml'
    else:
        rendered = json.dumps(config, indent=2, ensure_ascii=False)
        extension = '.json'
        content_type = 'application/json'
    return rendered.encode('utf-8'), content_type, extension


def _build_config_inputs(
    session_id: str,
    slug: str,
    template: NodeTemplate,
    node: FlowNodePayload,
    bucket: str,
) -> Dict[str, ArtifactPlan]:
    relevant_specs = [spec for spec in template.artifacts.inputs if _is_config_artifact(spec)]
    if not relevant_specs:
        return {}

    config_document = _build_config_document(node)
    client = get_minio_client()
    prefix = build_session_node_prefix(session_id, slug)
    config_inputs: Dict[str, ArtifactPlan] = {}

    for spec in relevant_specs:
        data_bytes, content_type, extension = _render_config_bytes(config_document, spec)
        object_name = f"{prefix}/config/{sanitize_path_segment(spec.name, 'config')}{extension}"
        upload_bytes(client, bucket, object_name, data_bytes, content_type=content_type)
        workflow_input_name = sanitize_path_segment(f"{slug}-{spec.name}-config", spec.name)
        config_inputs[spec.name] = ArtifactPlan(
            name=spec.name,
            path=spec.path,
            bucket=bucket,
            key=object_name,
            workflow_input_name=workflow_input_name,
        )

    return config_inputs


def _build_output_plan(
    session_id: str,
    slug: str,
    node: FlowNodePayload,
    template: NodeTemplate,
    bucket: str,
) -> Dict[str, ArtifactPlan]:
    outputs: Dict[str, ArtifactPlan] = {}
    prefix = build_session_node_prefix(session_id, slug)
    for spec in template.artifacts.outputs:
        suffix = _artifact_suffix(spec.path)
        key = f"{prefix}/outputs/{sanitize_path_segment(spec.name, 'artifact')}{suffix}"
        outputs[spec.name] = ArtifactPlan(
            name=spec.name,
            path=spec.path,
            bucket=bucket,
            key=key,
        )
    return outputs


def _build_input_node_plan(
    node: FlowNodePayload,
    template: NodeTemplate,
) -> Dict[str, ArtifactPlan]:
    artifact = _ensure_uploaded_artifact(node)
    outputs: Dict[str, ArtifactPlan] = {}
    for spec in template.artifacts.outputs:
        outputs[spec.name] = ArtifactPlan(
            name=spec.name,
            path=spec.path,
            bucket=artifact.bucket or get_input_bucket(),
            key=artifact.key,
        )
    return outputs


def _assign_bindings(
    plans: Dict[str, NodePlan],
    edges: List[FlowEdgePayload],
) -> None:
    inputs_lookup: Dict[str, List[str]] = {
        plan.node_id: [artifact.name for artifact in plan.template.artifacts.inputs]
        for plan in plans.values()
        if not plan.is_input
    }

    bindings_per_target: Dict[str, Dict[str, ArtifactBinding]] = defaultdict(dict)

    for edge in edges:
        source_plan = plans.get(edge.source)
        target_plan = plans.get(edge.target)
        if not source_plan or not target_plan or target_plan.is_input:
            continue

        target_inputs = inputs_lookup.get(target_plan.node_id, [])
        source_outputs = list(source_plan.outputs.values())
        if not source_outputs or not target_inputs:
            continue

        # Greedy match by name intersection first.
        remaining_outputs = {artifact.name: artifact for artifact in source_outputs}
        for input_name in target_inputs:
            if input_name in bindings_per_target[target_plan.node_id]:
                continue
            artifact = remaining_outputs.get(input_name)
            if artifact:
                bindings_per_target[target_plan.node_id][input_name] = ArtifactBinding(
                    input_name=input_name,
                    source_node_id=source_plan.node_id,
                    source_artifact_name=artifact.name,
                    bucket=artifact.bucket,
                    key=artifact.key,
                )
                remaining_outputs.pop(input_name, None)

        # Fallback: assign any remaining outputs to first unfilled inputs.
        remaining_inputs = [
            name
            for name in target_inputs
            if name not in bindings_per_target[target_plan.node_id]
        ]
        for input_name, artifact in zip(remaining_inputs, remaining_outputs.values()):
            bindings_per_target[target_plan.node_id][input_name] = ArtifactBinding(
                input_name=input_name,
                source_node_id=source_plan.node_id,
                source_artifact_name=artifact.name,
                bucket=artifact.bucket,
                key=artifact.key,
            )

        if not source_plan.is_input and source_plan.slug not in target_plan.dependencies:
            target_plan.dependencies.append(source_plan.slug)

    # Apply bindings to node plans
    for target_id, bindings in bindings_per_target.items():
        plan = plans[target_id]
        plan.input_bindings.update(bindings)

    for plan in plans.values():
        if plan.is_input:
            continue
        for name, artifact in plan.config_inputs.items():
            if name in plan.input_bindings:
                continue
            plan.input_bindings[name] = ArtifactBinding(
                input_name=name,
                source_node_id=None,
                source_artifact_name=name,
                bucket=artifact.bucket,
                key=artifact.key,
                workflow_input_name=artifact.workflow_input_name,
            )


def build_workflow_plan(payload: WorkflowGraphPayload) -> WorkflowPlan:
    catalog_index = _index_catalog()
    bucket = get_input_bucket()
    plans: Dict[str, NodePlan] = {}
    used_slugs: Dict[str, int] = {}

    for node in payload.nodes:
        template_name = node.data.templateName
        if not template_name:
            raise UnknownTemplateError(
                f"Node '{node.id}' does not specify a templateName"
            )
        template = catalog_index.get(template_name)
        if not template:
            raise UnknownTemplateError(
                f"Template '{template_name}' referenced by node '{node.id}' not found"
            )

        base_slug = _slug_for_node(node)
        occurrence = used_slugs.get(base_slug, 0)
        slug = f"{base_slug}-{occurrence}" if occurrence else base_slug
        used_slugs[base_slug] = occurrence + 1
        if template.type == "input":
            outputs = _build_input_node_plan(node, template)
            plan = NodePlan(
                node_id=node.id,
                slug=slug,
                template=template,
                is_input=True,
                is_output=False,
                outputs=outputs,
            )
        elif template.type == "output":
            plan = NodePlan(
                node_id=node.id,
                slug=slug,
                template=template,
                is_input=False,
                is_output=True,
                outputs={},
            )
        else:
            outputs = _build_output_plan(payload.session_id, slug, node, template, bucket)
            config_inputs = _build_config_inputs(
                payload.session_id,
                slug,
                template,
                node,
                bucket,
            )
            plan = NodePlan(
                node_id=node.id,
                slug=slug,
                template=template,
                is_input=False,
                is_output=False,
                outputs=outputs,
                config_inputs=config_inputs,
            )

        plans[node.id] = plan

    _assign_bindings(plans, payload.edges)

    return WorkflowPlan(
        session_id=payload.session_id,
        bucket=bucket,
        nodes=plans,
    )


def _workflow_input_name(node_plan: NodePlan, artifact: ArtifactPlan) -> str:
    base = f"{node_plan.slug}-{artifact.name}"
    return sanitize_path_segment(base, artifact.name)


def _make_task_arguments(
    plan: WorkflowPlan,
    node_plan: NodePlan,
    input_lookup: Dict[tuple[str, str], str],
) -> Optional[Dict[str, List[Dict[str, str]]]]:
    if not node_plan.template.artifacts.inputs:
        return None

    artifacts_args: List[Dict[str, str]] = []
    for spec in node_plan.template.artifacts.inputs:
        binding = node_plan.input_bindings.get(spec.name)
        if not binding:
            raise MissingArtifactError(
                f"Node '{node_plan.node_id}' missing binding for input artifact '{spec.name}'"
            )

        if binding.source_node_id is None:
            if not binding.workflow_input_name:
                raise MissingArtifactError(
                    f"Binding for input '{spec.name}' in node '{node_plan.node_id}' lacks source information"
                )
            artifacts_args.append(
                {
                    "name": spec.name,
                    "from": f"{{{{inputs.artifacts.{binding.workflow_input_name}}}}}",
                }
            )
            continue

        source_plan = plan.nodes[binding.source_node_id]

        if source_plan.is_input:
            mapping_key = (binding.source_node_id, binding.source_artifact_name)
            input_name = input_lookup.get(mapping_key)
            if not input_name:
                raise MissingArtifactError(
                    f"Workflow input mapping missing for node '{binding.source_node_id}' artifact '{binding.source_artifact_name}'"
                )
            artifacts_args.append(
                {
                    "name": spec.name,
                    "from": f"{{{{inputs.artifacts.{input_name}}}}}",
                }
            )
        else:
            artifacts_args.append(
                {
                    "name": spec.name,
                    "from": (
                        f"{{{{tasks.{source_plan.slug}.outputs.artifacts.{binding.source_artifact_name}}}}}"
                    ),
                }
            )

    if not artifacts_args:
        return None

    return {"artifacts": artifacts_args}


def build_workflow_manifest(plan: WorkflowPlan) -> Dict[str, object]:
    workflow_inputs: List[Dict[str, object]] = []
    input_lookup: Dict[tuple[str, str], str] = {}

    for node_plan in plan.nodes.values():
        if not node_plan.is_input:
            continue
        for artifact in node_plan.outputs.values():
            input_name = _workflow_input_name(node_plan, artifact)
            workflow_inputs.append(
                {
                    "name": input_name,
                    "s3": {
                        "bucket": artifact.bucket,
                        "key": artifact.key,
                    },
                }
            )
            input_lookup[(node_plan.node_id, artifact.name)] = input_name

    seen_manual: set[str] = set()
    for node_plan in plan.tasks():
        for artifact in node_plan.config_inputs.values():
            if not artifact.workflow_input_name or artifact.workflow_input_name in seen_manual:
                continue
            workflow_inputs.append(
                {
                    "name": artifact.workflow_input_name,
                    "s3": {
                        "bucket": artifact.bucket,
                        "key": artifact.key,
                    },
                }
            )
            input_lookup[(node_plan.node_id, artifact.name)] = artifact.workflow_input_name
            seen_manual.add(artifact.workflow_input_name)

    tasks: List[Dict[str, object]] = []
    for node_plan in plan.tasks():
        task: Dict[str, object] = {
            "name": node_plan.slug,
            "template": node_plan.slug,
        }

        if node_plan.dependencies:
            task["dependencies"] = node_plan.dependencies

        arguments = _make_task_arguments(plan, node_plan, input_lookup)
        if arguments:
            task["arguments"] = arguments

        tasks.append(task)

    entry_template: Dict[str, object] = {
        "name": "generated-pipeline",
        "dag": {"tasks": tasks},
    }

    if workflow_inputs:
        entry_template["inputs"] = {"artifacts": workflow_inputs}

    manifest: Dict[str, object] = {
        "apiVersion": "argoproj.io/v1alpha1",
        "kind": "Workflow",
        "metadata": {
            "generateName": f"sds-{sanitize_path_segment(plan.session_id[:16], 'run')}-",
        },
        "spec": {
            "entrypoint": "generated-pipeline",
            "archiveLogs": True,
            "templates": [entry_template],
        },
    }

    manifest["spec"]["serviceAccountName"] = SERVICE_ACCOUNT_NAME

    session_label = _sanitize_label_value(plan.session_id, "session")

    metadata = manifest.setdefault("metadata", {})
    metadata_labels = metadata.setdefault("labels", {})
    metadata_labels.setdefault("sds.dev/session", session_label)

    spec_metadata = manifest["spec"].setdefault("podMetadata", {})
    spec_metadata_labels = spec_metadata.setdefault("labels", {})
    spec_metadata_labels.setdefault("sds.dev/session", session_label)

    customized_templates: List[Dict[str, Any]] = []
    for node_plan in plan.tasks():
        registry_entry = _WORKFLOW_TEMPLATE_REGISTRY.get(node_plan.template.name)
        if not registry_entry:
            continue

        template_spec = deepcopy(registry_entry)
        template_spec["name"] = node_plan.slug

        outputs_section = template_spec.get("outputs", {})
        artifacts_section = outputs_section.get("artifacts", [])
        if isinstance(artifacts_section, list):
            for artifact_spec in artifacts_section:
                if not isinstance(artifact_spec, dict):
                    continue
                artifact_name = artifact_spec.get("name")
                if not isinstance(artifact_name, str):
                    continue
                plan_artifact = node_plan.outputs.get(artifact_name)
                if not plan_artifact:
                    continue

                s3_section = artifact_spec.setdefault("s3", {})
                if isinstance(s3_section, dict):
                    s3_section["key"] = plan_artifact.key

        customized_templates.append(template_spec)

        node_label = _sanitize_label_value(node_plan.slug, "node")
        node_id_label = _sanitize_label_value(node_plan.node_id, "node")
        template_spec["archiveLocation"] = {
            "s3": {
                "bucket": plan.bucket,
                "key": f"sessions/{session_label}/nodes/{node_label}/logs/{{{{pod.name}}}}.log",
            }
        }

        template_metadata = template_spec.setdefault("metadata", {})
        template_labels = template_metadata.setdefault("labels", {})
        template_labels.setdefault("sds.dev/session", session_label)
        template_labels.setdefault("sds.dev/node", node_label)
        template_labels.setdefault("sds.dev/node-id", node_id_label)

    if customized_templates:
        manifest["spec"]["templates"].extend(customized_templates)

    return manifest


def render_workflow_yaml(plan: WorkflowPlan) -> str:
    manifest = build_workflow_manifest(plan)
    return yaml.safe_dump(manifest, sort_keys=False)


def suggest_workflow_filename(plan: WorkflowPlan) -> str:
    slug = sanitize_path_segment(plan.session_id[:16], "workflow")
    return f"workflow-{slug}.yaml"

