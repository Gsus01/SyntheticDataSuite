"""Persist workflow definitions and metadata in PostgreSQL."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from db import db_session
from workflow_models import WorkflowRecordDB

logger = logging.getLogger(__name__)


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


def _row_to_record(row: WorkflowRecordDB) -> StoredWorkflowRecord:
    return StoredWorkflowRecord(
        workflowId=row.workflow_id,
        name=row.name,
        description=row.description,
        sessionId=row.session_id,
        nodes=row.nodes,
        edges=row.edges,
        compiledManifest=None,
        manifestFilename=row.manifest_filename,
        compiledAt=row.compiled_at.isoformat() if row.compiled_at else None,
        nodeSlugMap=row.node_slug_map or {},
        createdAt=row.created_at.isoformat(),
        updatedAt=row.updated_at.isoformat(),
        lastWorkflowName=row.last_workflow_name,
        lastNamespace=row.last_namespace,
        lastBucket=row.last_bucket,
        lastKey=row.last_key,
        lastManifestFilename=row.last_manifest_filename,
        lastCliOutput=row.last_cli_output,
        lastSubmittedAt=row.last_submitted_at.isoformat()
        if row.last_submitted_at
        else None,
        version=row.version or 1,
    )


class WorkflowStore:
    """Persistence layer for workflow definitions in PostgreSQL."""

    def get_workflow(self, workflow_id: str) -> StoredWorkflowRecord:
        with db_session() as session:
            row = session.get(WorkflowRecordDB, workflow_id)
            if not row:
                raise WorkflowNotFoundError(
                    f"No se encontró el workflow con id '{workflow_id}'."
                )
            return _row_to_record(row)

    def list_workflows(self) -> List[StoredWorkflowSummary]:
        with db_session() as session:
            rows = (
                session.query(WorkflowRecordDB)
                .order_by(WorkflowRecordDB.updated_at.desc())
                .all()
            )

            return [_row_to_record(row).summary() for row in rows]

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
            raise WorkflowStoreValidationError(
                "El nombre del workflow no puede estar vacío."
            )

        cleaned_session_id = _clean_text(session_id)
        if not cleaned_session_id:
            raise WorkflowStoreValidationError("sessionId es obligatorio.")

        resolved_id = workflow_id or uuid4().hex
        now = datetime.now(timezone.utc)

        with db_session() as session:
            row = session.get(WorkflowRecordDB, resolved_id)
            if workflow_id and not row:
                raise WorkflowNotFoundError(
                    f"No se encontró el workflow con id '{resolved_id}' para actualizar."
                )

            if not row:
                row = WorkflowRecordDB(
                    workflow_id=resolved_id,
                    name=cleaned_name,
                    description=_clean_text(description),
                    session_id=cleaned_session_id,
                    nodes=_json_clone(nodes),
                    edges=_json_clone(edges),
                    node_slug_map=node_slug_map or {},
                    manifest_filename=manifest_filename,
                    compiled_at=now if compiled_manifest else None,
                    version=1,
                )
                session.add(row)
            else:
                row.name = cleaned_name
                row.description = _clean_text(description)
                row.session_id = cleaned_session_id
                row.nodes = _json_clone(nodes)
                row.edges = _json_clone(edges)
                row.node_slug_map = node_slug_map or row.node_slug_map
                if manifest_filename:
                    row.manifest_filename = manifest_filename
                if compiled_manifest:
                    row.compiled_at = now
                row.version = (row.version or 1) + 1

            session.flush()
            return _row_to_record(row)

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
        with db_session() as session:
            row = session.get(WorkflowRecordDB, workflow_id)
            if not row:
                raise WorkflowNotFoundError(
                    f"No se encontró el workflow con id '{workflow_id}'."
                )

            now = datetime.now(timezone.utc)
            row.last_workflow_name = workflow_name
            row.last_namespace = namespace
            row.last_bucket = bucket
            row.last_key = key
            row.last_manifest_filename = manifest_filename
            row.last_cli_output = cli_output
            row.last_submitted_at = now
            row.node_slug_map = node_slug_map or row.node_slug_map
            row.updated_at = now
            row.version = (row.version or 1) + 1

            session.flush()
            return _row_to_record(row)
