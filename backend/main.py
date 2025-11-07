import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from uuid import uuid4

from minio.error import S3Error

from minio_helper import (
    build_session_node_prefix,
    ensure_bucket,
    get_input_bucket,
    get_minio_client,
    sanitize_path_segment,
)
from catalog_loader import NodeTemplate, load_catalog
from workflow_builder import (
    WorkflowGraphPayload,
    MissingArtifactError,
    UnknownTemplateError,
    build_workflow_plan,
    render_workflow_yaml,
    suggest_workflow_filename,
)


logger = logging.getLogger(__name__)


class ArtifactUploadResponse(BaseModel):
    bucket: str
    key: str
    size: int
    content_type: Optional[str] = None
    original_filename: Optional[str] = None


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


@app.post("/workflow/render")
def render_workflow(payload: WorkflowGraphPayload):
    try:
        plan = build_workflow_plan(payload)
        yaml_content = render_workflow_yaml(plan)
        filename = suggest_workflow_filename(plan)
    except (UnknownTemplateError, MissingArtifactError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # pragma: no cover - unexpected
        raise HTTPException(status_code=500, detail=f"Failed to render workflow: {exc}")

    return {"filename": filename, "yaml": yaml_content}
