import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from uuid import uuid4

from minio.error import S3Error

from minio_helper import (
    build_session_node_prefix,
    ensure_bucket,
    get_input_bucket,
    get_minio_client,
    sanitize_path_segment,
)


CATALOG_PATH = Path(__file__).parent / "catalog" / "nodes.yaml"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


logger = logging.getLogger(__name__)


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
    limits: Optional[Dict] = None
    version: Optional[str] = None
    parameter_defaults: Optional[Dict[str, Any]] = None


class ArtifactUploadResponse(BaseModel):
    bucket: str
    key: str
    size: int
    content_type: Optional[str] = None
    original_filename: Optional[str] = None


def load_catalog() -> List[NodeTemplate]:
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(f"Catalog file not found at {CATALOG_PATH}")
    try:
        with CATALOG_PATH.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except Exception as e:
        raise RuntimeError(f"Failed to read catalog: {e}")

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
            except Exception as exc:
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
        except Exception as exc:
            raise RuntimeError(f"Invalid catalog format for node '{node_data.get('name')}': {exc}")

    return templates


app = FastAPI(title="Synthetic Data Suite Backend", version="0.1.0")

# Allow CORS in dev so the frontend can call the API easily
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/artifacts/upload", response_model=ArtifactUploadResponse)
async def upload_artifact(
    session_id: str = Form(...),
    node_id: str = Form(...),
    file: UploadFile = File(...),
    artifact_name: Optional[str] = Form(None),
) -> ArtifactUploadResponse:
    original_filename = file.filename
    suffix = Path(original_filename or "").suffix.lower()

    prefix = build_session_node_prefix(session_id, node_id)
    raw_base = artifact_name or (Path(original_filename).stem if original_filename else None)
    safe_base = sanitize_path_segment(raw_base or "artifact", "artifact")
    key = f"{prefix}/{safe_base}-{uuid4().hex}{suffix}"

    try:
        client = get_minio_client()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    target_bucket = get_input_bucket()

    try:
        ensure_bucket(client, target_bucket)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    inner_file = file.file
    size: Optional[int] = None
    try:
        current_pos = inner_file.tell()
    except (AttributeError, OSError):
        current_pos = None

    try:
        inner_file.seek(0, os.SEEK_END)
        size = inner_file.tell()
        inner_file.seek(0)
    except (AttributeError, OSError):
        size = None
        if current_pos is not None:
            try:
                inner_file.seek(current_pos)
            except Exception:  # noqa: BLE001 - best-effort reset
                pass

    content_type = file.content_type or "application/octet-stream"
    upload_kwargs = {"content_type": content_type}
    length = size if size is not None else -1
    if size is None:
        upload_kwargs["part_size"] = 5 * 1024 * 1024

    try:
        result = client.put_object(
            target_bucket,
            key,
            inner_file,
            length=length,
            **upload_kwargs,
        )
    except S3Error as exc:
        logger.error("Failed to upload artifact to MinIO: %s", exc)
        raise HTTPException(status_code=500, detail=f"Error subiendo a MinIO: {exc}") from exc
    finally:
        await file.close()

    uploaded_size = getattr(result, "size", None)
    if isinstance(uploaded_size, int):
        size = uploaded_size
    elif size is None:
        size = 0

    return ArtifactUploadResponse(
        bucket=target_bucket,
        key=key,
        size=size or 0,
        content_type=content_type,
        original_filename=original_filename,
    )


@app.get("/workflow-templates", response_model=List[NodeTemplate])
def get_workflow_templates() -> List[NodeTemplate]:
    try:
        return load_catalog()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}
