"""Persist workflow definitions and metadata in MinIO."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from minio.error import S3Error  # type: ignore[import-not-found]
from pydantic import BaseModel, ConfigDict, Field

from minio_helper import ensure_bucket, get_input_bucket, get_minio_client, upload_bytes

logger = logging.getLogger(__name__)

_WORKFLOW_PREFIX = "workflows"


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _json_clone(value: Any) -> Any:
    """Ensure the payload is JSON serializable and detached from React state."""

    return json.loads(json.dumps(value))


class WorkflowStoreError(RuntimeError):
    """Base error for workflow store operations."""


class WorkflowNotFoundError(WorkflowStoreError):
    """Raised when a workflow definition is missing."""


class WorkflowStoreValidationError(WorkflowStoreError):
    """Raised when the workflow payload lacks required information."""

class StoredWorkflowRecord(BaseModel):
    """Full representation of a workflow definition."""

    model_config = ConfigDict(populate_by_name=True)

    workflow_id: str = Field(..., alias="workflowId")
    name: str
    description: Optional[str] = None
    session_id: str = Field(..., alias="sessionId")
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    compiled_manifest: Optional[str] = Field(None, alias="compiledManifest")
    manifest_filename: Optional[str] = Field(None, alias="manifestFilename")
    compiled_at: Optional[str] = Field(None, alias="compiledAt")
    node_slug_map: Dict[str, str] = Field(default_factory=dict, alias="nodeSlugMap")
    created_at: str = Field(..., alias="createdAt")
    updated_at: str = Field(..., alias="updatedAt")
    last_workflow_name: Optional[str] = Field(None, alias="lastWorkflowName")
    last_namespace: Optional[str] = Field(None, alias="lastNamespace")
    last_bucket: Optional[str] = Field(None, alias="lastBucket")
    last_key: Optional[str] = Field(None, alias="lastKey")
    last_manifest_filename: Optional[str] = Field(None, alias="lastManifestFilename")
    last_cli_output: Optional[str] = Field(None, alias="lastCliOutput")
    last_submitted_at: Optional[str] = Field(None, alias="lastSubmittedAt")
    version: int = 1

    def summary(self) -> "StoredWorkflowSummary":
        return StoredWorkflowSummary(
            workflowId=self.workflow_id,
            name=self.name,
            description=self.description,
            createdAt=self.created_at,
            updatedAt=self.updated_at,
            lastSubmittedAt=self.last_submitted_at,
            lastWorkflowName=self.last_workflow_name,
            lastNamespace=self.last_namespace,
        )


class StoredWorkflowSummary(BaseModel):
    """Lightweight metadata used for listing workflows."""

    model_config = ConfigDict(populate_by_name=True)

    workflow_id: str = Field(..., alias="workflowId")
    name: str
    description: Optional[str] = None
    created_at: str = Field(..., alias="createdAt")
    updated_at: str = Field(..., alias="updatedAt")
    last_submitted_at: Optional[str] = Field(None, alias="lastSubmittedAt")
    last_workflow_name: Optional[str] = Field(None, alias="lastWorkflowName")
    last_namespace: Optional[str] = Field(None, alias="lastNamespace")


class WorkflowStore:
    """Simple persistence layer for workflow definitions."""

    def __init__(self, prefix: str = _WORKFLOW_PREFIX) -> None:
        self.bucket = get_input_bucket()
        self.prefix = prefix.strip("/ ")

    def _object_name(self, workflow_id: str) -> str:
        return f"{self.prefix}/{workflow_id}.json"

    def _write_record(self, record: StoredWorkflowRecord) -> None:
        client = get_minio_client()
        data = json.dumps(record.model_dump(by_alias=True), ensure_ascii=False).encode("utf-8")
        upload_bytes(
            client,
            self.bucket,
            self._object_name(record.workflow_id),
            data,
            content_type="application/json",
        )

    def _read_object(self, object_name: str) -> Dict[str, Any]:
        client = get_minio_client()
        ensure_bucket(client, self.bucket)
        try:
            response = client.get_object(self.bucket, object_name)
        except S3Error as exc:
            if exc.code == "NoSuchKey":
                raise WorkflowNotFoundError(f"Workflow object '{object_name}' not found") from exc
            raise WorkflowStoreError(f"Failed to load workflow object: {exc}") from exc

        try:
            data = response.read()
        finally:
            response.close()
            response.release_conn()

        try:
            return json.loads(data.decode("utf-8"))
        except json.JSONDecodeError as exc:  # pragma: no cover - corrupt payloads
            raise WorkflowStoreError(f"Workflow data corrupted for {object_name}") from exc

    def get_workflow(self, workflow_id: str) -> StoredWorkflowRecord:
        object_name = self._object_name(workflow_id)
        raw = self._read_object(object_name)
        return StoredWorkflowRecord.model_validate(raw)

    def list_workflows(self) -> List[StoredWorkflowSummary]:
        client = get_minio_client()
        ensure_bucket(client, self.bucket)
        summaries: List[StoredWorkflowSummary] = []
        try:
            objects = client.list_objects(self.bucket, prefix=f"{self.prefix}/", recursive=True)
            for obj in objects:
                if not obj.object_name.endswith(".json"):
                    continue
                workflow_id = obj.object_name.rsplit("/", 1)[-1].removesuffix(".json")
                try:
                    record = self.get_workflow(workflow_id)
                except WorkflowNotFoundError:
                    logger.warning("Workflow object %s disappeared during listing", obj.object_name)
                    continue
                summaries.append(record.summary())
        except S3Error as exc:
            raise WorkflowStoreError(f"Failed to list workflows: {exc}") from exc

        summaries.sort(key=lambda summary: summary.updated_at, reverse=True)
        return summaries

    def save_definition(
        self,
        *,
        workflow_id: Optional[str],
        name: str,
        description: Optional[str],
        session_id: str,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        compiled_manifest: Optional[str],
        manifest_filename: Optional[str],
        node_slug_map: Optional[Dict[str, str]] = None,
    ) -> StoredWorkflowRecord:
        cleaned_name = _clean_text(name)
        if not cleaned_name:
            raise WorkflowStoreValidationError("El nombre del workflow no puede estar vacío.")

        cleaned_session_id = _clean_text(session_id)
        if not cleaned_session_id:
            raise WorkflowStoreValidationError("sessionId es obligatorio.")

        existing: Optional[StoredWorkflowRecord] = None
        resolved_id = workflow_id
        if resolved_id:
            try:
                existing = self.get_workflow(resolved_id)
            except WorkflowNotFoundError as exc:
                raise WorkflowNotFoundError(
                    f"No se encontró el workflow con id '{resolved_id}' para actualizar."
                ) from exc
        else:
            resolved_id = uuid4().hex

        now_iso = _iso_now()
        created_at = existing.created_at if existing else now_iso
        persisted_manifest = (
            compiled_manifest if compiled_manifest else (existing.compiled_manifest if existing else None)
        )
        persisted_manifest_filename = (
            manifest_filename if manifest_filename else (existing.manifest_filename if existing else None)
        )
        compiled_at = now_iso if compiled_manifest else (existing.compiled_at if existing else None)
        persisted_slug_map = (node_slug_map or existing.node_slug_map) if existing else (node_slug_map or {})

        record = StoredWorkflowRecord(
            workflowId=resolved_id,
            name=cleaned_name,
            description=_clean_text(description),
            sessionId=cleaned_session_id,
            nodes=_json_clone(nodes),
            edges=_json_clone(edges),
            compiledManifest=persisted_manifest,
            manifestFilename=persisted_manifest_filename,
            compiledAt=compiled_at,
            nodeSlugMap=persisted_slug_map,
            createdAt=created_at,
            updatedAt=now_iso,
            lastWorkflowName=existing.last_workflow_name if existing else None,
            lastNamespace=existing.last_namespace if existing else None,
            lastBucket=existing.last_bucket if existing else None,
            lastKey=existing.last_key if existing else None,
            lastManifestFilename=existing.last_manifest_filename if existing else None,
            lastCliOutput=existing.last_cli_output if existing else None,
            lastSubmittedAt=existing.last_submitted_at if existing else None,
            version=existing.version if existing else 1,
        )

        self._write_record(record)
        return record

    def record_submission(
        self,
        workflow_id: str,
        *,
        workflow_name: str,
        namespace: str,
        bucket: str,
        key: str,
        manifest_filename: str,
        cli_output: Optional[str],
        node_slug_map: Dict[str, str],
    ) -> StoredWorkflowRecord:
        record = self.get_workflow(workflow_id)
        now_iso = _iso_now()
        updated_record = record.model_copy(
            update={
                "last_workflow_name": workflow_name,
                "last_namespace": namespace,
                "last_bucket": bucket,
                "last_key": key,
                "last_manifest_filename": manifest_filename,
                "last_cli_output": cli_output,
                "last_submitted_at": now_iso,
                "node_slug_map": node_slug_map or record.node_slug_map,
                "updated_at": now_iso,
            }
        )
        self._write_record(updated_record)
        return updated_record
