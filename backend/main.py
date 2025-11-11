import base64
import json
import logging
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ConfigDict
from uuid import uuid4

from minio.error import S3Error  # type: ignore[import-not-found]

from minio_helper import (
    build_session_node_prefix,
    ensure_bucket,
    get_input_bucket,
    get_minio_client,
    sanitize_path_segment,
    upload_bytes,
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
from argo_client import ArgoClient, ArgoClientError, ArgoNotFoundError


logger = logging.getLogger(__name__)

_FINAL_WORKFLOW_PHASES = {"Succeeded", "Failed", "Error", "Skipped", "Omitted", "Terminated"}


class ArtifactUploadResponse(BaseModel):
    bucket: str
    key: str
    size: int
    content_type: Optional[str] = None
    original_filename: Optional[str] = None


class WorkflowSubmitResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    workflow_name: str = Field(..., alias="workflowName")
    namespace: str
    node_slug_map: Dict[str, str] = Field(default_factory=dict, alias="nodeSlugMap")
    bucket: str
    key: str
    manifest_filename: str = Field(..., alias="manifestFilename")
    cli_output: Optional[str] = Field(None, alias="cliOutput")


@dataclass
class ArgoSubmissionResult:
    workflow_name: str
    namespace: str
    cli_output: Optional[str]


class OutputArtifactInfo(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    input_name: str = Field(..., alias="inputName")
    source_node_id: Optional[str] = Field(None, alias="sourceNodeId")
    source_artifact_name: str = Field(..., alias="sourceArtifactName")
    bucket: str
    key: str
    workflow_input_name: Optional[str] = Field(None, alias="workflowInputName")
    size: Optional[int] = None
    content_type: Optional[str] = Field(None, alias="contentType")
    exists: bool = False


class ArtifactPreviewResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    content: str
    truncated: bool
    content_type: Optional[str] = Field(None, alias="contentType")
    encoding: str = "utf-8"
    size: Optional[int] = None


class WorkflowLogChunk(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    key: str
    node_slug: str = Field(..., alias="nodeSlug")
    pod_name: str = Field(..., alias="podName")
    content: str
    start_offset: int = Field(..., alias="startOffset")
    end_offset: int = Field(..., alias="endOffset")
    has_more: bool = Field(..., alias="hasMore")
    encoding: str = "utf-8"
    timestamp: float


class WorkflowLogStreamResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    cursor: str
    chunks: List[WorkflowLogChunk]


class WorkflowNodeStatus(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    slug: str
    phase: Optional[str] = None
    type: Optional[str] = None
    id: Optional[str] = None
    display_name: Optional[str] = Field(None, alias="displayName")
    message: Optional[str] = None
    progress: Optional[str] = None
    started_at: Optional[str] = Field(None, alias="startedAt")
    finished_at: Optional[str] = Field(None, alias="finishedAt")


class WorkflowStatusResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    workflow_name: str = Field(..., alias="workflowName")
    namespace: str
    phase: Optional[str] = None
    finished: bool
    nodes: Dict[str, WorkflowNodeStatus]
    updated_at: Optional[str] = Field(None, alias="updatedAt")


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


def _extract_workflow_name(text: str) -> Optional[str]:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("name:"):
            return stripped.split(":", 1)[1].strip()
    return None


def _submit_workflow_via_cli(yaml_content: str) -> ArgoSubmissionResult:
    namespace = os.getenv("ARGO_NAMESPACE", "argo")
    argo_cli = os.getenv("ARGO_CLI_PATH", "argo")
    extra_args = os.getenv("ARGO_SUBMIT_EXTRA_ARGS")

    command = [argo_cli, "submit", "-", "--namespace", namespace, "--output", "json"]
    if extra_args:
        command.extend(shlex.split(extra_args))

    try:
        result = subprocess.run(
            command,
            input=yaml_content.encode("utf-8"),
            capture_output=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Argo CLI no encontrado. Asegúrate de instalar 'argo' o configura ARGO_CLI_PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:  # noqa:  BLE001 - necesitamos capturar errores específicos
        stdout_text = exc.stdout.decode("utf-8", errors="ignore") if exc.stdout else ""
        stderr_text = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else ""
        combined = "\n".join(filter(None, [stdout_text.strip(), stderr_text.strip()]))
        logger.error("Fallo al ejecutar 'argo submit': %s", combined or exc)
        raise RuntimeError(combined or "Fallo al ejecutar 'argo submit'") from exc

    stdout_text = result.stdout.decode("utf-8", errors="ignore").strip()
    stderr_text = result.stderr.decode("utf-8", errors="ignore").strip()

    workflow_name: Optional[str] = None

    if stdout_text:
        try:
            parsed = json.loads(stdout_text)
            workflow_name = parsed.get("metadata", {}).get("name")
        except json.JSONDecodeError:
            workflow_name = _extract_workflow_name(stdout_text)

    if not workflow_name and stderr_text:
        workflow_name = _extract_workflow_name(stderr_text)

    if not workflow_name:
        logger.warning("No se pudo extraer el nombre del workflow de la salida: %s", stdout_text)
        raise RuntimeError(
            "No se pudo determinar el nombre del workflow desde la salida del comando 'argo submit'."
        )

    cli_output = stdout_text or stderr_text or None
    return ArgoSubmissionResult(workflow_name=workflow_name, namespace=namespace, cli_output=cli_output)


def _stat_object(client, bucket: str, key: str):
    try:
        return client.stat_object(bucket, key)
    except S3Error as exc:
        if exc.code in {"NoSuchKey", "NoSuchBucket"}:
            return None
        raise


def _stream_object(client, bucket: str, key: str, chunk_size: int = 1024 * 1024) -> Iterable[bytes]:
    obj = client.get_object(bucket, key)
    try:
        for chunk in iter(lambda: obj.read(chunk_size), b""):
            if chunk:
                yield chunk
            else:
                break
    finally:
        obj.close()
        obj.release_conn()


_CURSOR_VERSION = 1
_MAX_CURSOR_ENTRIES = 512
_MAX_LOG_OBJECTS = 256
_MAX_LOG_CHUNK_BYTES = 512 * 1024
_MIN_LOG_CHUNK_BYTES = 1024


def _decode_cursor(cursor_raw: Optional[str]) -> Dict[str, int]:
    if not cursor_raw:
        return {}

    padding = "=" * (-len(cursor_raw) % 4)
    try:
        data = base64.urlsafe_b64decode((cursor_raw + padding).encode("utf-8"))
        payload = json.loads(data.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - cursor invalid, ignoramos
        logger.warning("Cursor de logs inválido, restableciendo: %s", exc)
        return {}

    if not isinstance(payload, dict) or payload.get("v") != _CURSOR_VERSION:
        return {}

    objects = payload.get("objects", {})
    if not isinstance(objects, dict):
        return {}

    decoded: Dict[str, int] = {}
    for key, value in objects.items():
        try:
            decoded[str(key)] = max(0, int(value))
        except (TypeError, ValueError):
            continue

    return decoded


def _encode_cursor(mapped: Dict[str, int]) -> str:
    if len(mapped) > _MAX_CURSOR_ENTRIES:
        # Preserve the most recent entries deterministically by key ordering
        trimmed_items = sorted(mapped.items(), key=lambda item: item[0])[-_MAX_CURSOR_ENTRIES:]
        mapped = dict(trimmed_items)
    payload = {"v": _CURSOR_VERSION, "objects": {k: max(0, int(v)) for k, v in mapped.items()}}
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


@app.post("/workflow/submit", response_model=WorkflowSubmitResponse)
def submit_workflow(payload: WorkflowGraphPayload) -> WorkflowSubmitResponse:
    try:
        plan = build_workflow_plan(payload)
        yaml_content = render_workflow_yaml(plan)
        filename = suggest_workflow_filename(plan)
    except (UnknownTemplateError, MissingArtifactError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # pragma: no cover - unexpected
        raise HTTPException(status_code=500, detail=f"Failed to render workflow: {exc}")

    session_segment = sanitize_path_segment(payload.session_id, "session")
    manifest_key = f"sessions/{session_segment}/workflow/{filename}"

    try:
        client = get_minio_client()
        upload_bytes(
            client,
            plan.bucket,
            manifest_key,
            yaml_content.encode("utf-8"),
            content_type="application/x-yaml",
        )
    except (S3Error, RuntimeError) as exc:
        logger.error("Fallo al subir el workflow a MinIO: %s", exc)
        raise HTTPException(status_code=500, detail=f"Error subiendo a MinIO: {exc}") from exc

    try:
        submission = _submit_workflow_via_cli(yaml_content)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    node_slug_map = {node_plan.node_id: node_plan.slug for node_plan in plan.tasks()}

    logger.info(
        "Workflow %s enviado a Argo en namespace %s. Manifest almacenado en %s/%s",
        submission.workflow_name,
        submission.namespace,
        plan.bucket,
        manifest_key,
    )

    return WorkflowSubmitResponse(
        workflowName=submission.workflow_name,
        namespace=submission.namespace,
        nodeSlugMap=node_slug_map,
        bucket=plan.bucket,
        key=manifest_key,
        manifestFilename=filename,
        cliOutput=submission.cli_output,
    )


@app.get("/workflow/status", response_model=WorkflowStatusResponse)
def get_workflow_status(
    workflow_name: str = Query(..., alias="workflowName"),
    namespace: Optional[str] = Query(None),
) -> WorkflowStatusResponse:
    resolved_namespace = namespace or os.getenv("ARGO_NAMESPACE", "argo")

    try:
        client = ArgoClient()
        workflow_obj = client.get_workflow(resolved_namespace, workflow_name)
    except ArgoNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ArgoClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    status_obj: Dict[str, Any] = workflow_obj.get("status") or {}
    phase = status_obj.get("phase")
    finished_at = status_obj.get("finishedAt")
    finished = bool(finished_at) or (phase in _FINAL_WORKFLOW_PHASES)

    nodes_raw = status_obj.get("nodes") or {}
    node_statuses: Dict[str, WorkflowNodeStatus] = {}

    if isinstance(nodes_raw, dict):
        for node in nodes_raw.values():
            if not isinstance(node, dict):
                continue

            display_name = node.get("displayName") or node.get("name")
            if not display_name:
                continue

            node_statuses[display_name] = WorkflowNodeStatus(
                slug=display_name,
                phase=node.get("phase"),
                type=node.get("type"),
                id=node.get("id"),
                display_name=display_name,
                message=node.get("message"),
                progress=node.get("progress"),
                started_at=node.get("startedAt"),
                finished_at=node.get("finishedAt"),
            )

    updated_at = status_obj.get("finishedAt") or status_obj.get("startedAt")

    return WorkflowStatusResponse(
        workflow_name=workflow_name,
        namespace=resolved_namespace,
        phase=phase,
        finished=finished,
        nodes=node_statuses,
        updated_at=updated_at,
    )


@app.post("/workflow/output-artifacts", response_model=List[OutputArtifactInfo])
def get_output_artifacts(payload: WorkflowGraphPayload, node_id: str = Query(..., alias="nodeId")) -> List[OutputArtifactInfo]:
    try:
        plan = build_workflow_plan(payload)
    except (UnknownTemplateError, MissingArtifactError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # pragma: no cover - unexpected
        raise HTTPException(status_code=500, detail=f"Failed to build workflow plan: {exc}")

    node_plan = plan.nodes.get(node_id)
    if not node_plan:
        raise HTTPException(status_code=404, detail=f"Nodo '{node_id}' no encontrado en el grafo")
    if node_plan.template.type != "output":
        raise HTTPException(status_code=400, detail="El nodo indicado no es de tipo salida")

    try:
        client = get_minio_client()
    except RuntimeError as exc:
        logger.warning("No se pudo inicializar cliente de MinIO para consultar salidas: %s", exc)
        client = None

    artifacts: List[OutputArtifactInfo] = []

    for binding in node_plan.input_bindings.values():
        info = OutputArtifactInfo(
            inputName=binding.input_name,
            sourceNodeId=binding.source_node_id,
            sourceArtifactName=binding.source_artifact_name,
            bucket=binding.bucket,
            key=binding.key,
            workflowInputName=binding.workflow_input_name,
        )

        if client:
            try:
                stat = _stat_object(client, binding.bucket, binding.key)
            except S3Error as exc:
                logger.warning(
                    "Error consultando metadata de artefacto %s/%s: %s",
                    binding.bucket,
                    binding.key,
                    exc,
                )
                stat = None

            if stat:
                info.size = getattr(stat, "size", None)
                info.content_type = getattr(stat, "content_type", None)
                info.exists = True

        artifacts.append(info)

    return artifacts


@app.get("/artifacts/download")
def download_artifact(bucket: str, key: str):
    try:
        client = get_minio_client()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    stat = _stat_object(client, bucket, key)
    if stat is None:
        raise HTTPException(status_code=404, detail="Artefacto no encontrado en MinIO")

    filename = Path(key).name or "artifact"
    media_type = getattr(stat, "content_type", None) or "application/octet-stream"

    try:
        stream = _stream_object(client, bucket, key)
    except S3Error as exc:
        if exc.code in {"NoSuchKey", "NoSuchBucket"}:
            raise HTTPException(status_code=404, detail="Artefacto no encontrado en MinIO") from exc
        raise HTTPException(status_code=500, detail=f"Error leyendo desde MinIO: {exc}") from exc

    response = StreamingResponse(stream, media_type=media_type)
    response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    if getattr(stat, "size", None) is not None:
        response.headers["Content-Length"] = str(stat.size)
    return response


@app.get("/artifacts/preview", response_model=ArtifactPreviewResponse)
def preview_artifact(
    bucket: str,
    key: str,
    max_bytes: int = Query(65536, ge=128, le=4 * 1024 * 1024, alias="maxBytes"),
) -> ArtifactPreviewResponse:
    try:
        client = get_minio_client()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    stat = _stat_object(client, bucket, key)
    if stat is None:
        raise HTTPException(status_code=404, detail="Artefacto no encontrado en MinIO")

    size = getattr(stat, "size", None)
    if size is None:
        read_length = max_bytes
    elif size <= 0:
        read_length = 0
    else:
        read_length = min(max_bytes, size)

    obj = None
    try:
        if read_length > 0:
            obj = client.get_object(bucket, key, offset=0, length=read_length)
            data = obj.read(read_length)
        else:
            data = b""
    except S3Error as exc:
        if exc.code in {"NoSuchKey", "NoSuchBucket"}:
            raise HTTPException(status_code=404, detail="Artefacto no encontrado en MinIO") from exc
        raise HTTPException(status_code=500, detail=f"Error leyendo desde MinIO: {exc}") from exc
    finally:
        try:
            if obj is not None:
                obj.close()
                obj.release_conn()
        except Exception:  # noqa: BLE001 - best effort cleanup
            pass

    truncated = False
    if size is not None:
        truncated = size > len(data)

    try:
        content = data.decode("utf-8")
        encoding = "utf-8"
    except UnicodeDecodeError:
        content = data.decode("utf-8", errors="replace")
        encoding = "utf-8"

    return ArtifactPreviewResponse(
        content=content,
        truncated=truncated,
        contentType=getattr(stat, "content_type", None),
        encoding=encoding,
        size=getattr(stat, "size", None),
    )


@app.get("/workflow/logs/stream", response_model=WorkflowLogStreamResponse)
def stream_workflow_logs(
    workflow_name: str = Query(..., alias="workflowName"),
    namespace: Optional[str] = Query(None),
    cursor: Optional[str] = None,
    container: Optional[str] = Query(None),
    tail_lines: Optional[int] = Query(2000, ge=1, le=4000, alias="tailLines"),
    since_seconds: Optional[int] = Query(None, ge=1, le=24 * 3600, alias="sinceSeconds"),
) -> WorkflowLogStreamResponse:
    resolved_namespace = namespace or os.getenv("ARGO_NAMESPACE", "argo")

    try:
        client = ArgoClient()
        workflow_obj = client.get_workflow(resolved_namespace, workflow_name)
    except ArgoNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ArgoClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    status_obj: Dict[str, Any] = workflow_obj.get("status") or {}
    nodes_raw = status_obj.get("nodes") or {}

    pod_entries: List[tuple[str, str, Dict[str, Any]]] = []
    if isinstance(nodes_raw, dict):
        for node in nodes_raw.values():
            if not isinstance(node, dict):
                continue
            pod_name = node.get("podName") or node.get("id")
            if not pod_name:
                continue
            display_name = node.get("displayName") or node.get("name") or node.get("id") or pod_name
            pod_entries.append((display_name, pod_name, node))

    cursor_map = _decode_cursor(cursor)
    updated_cursor = dict(cursor_map)
    chunks: List[WorkflowLogChunk] = []

    if not pod_entries:
        encoded_cursor = _encode_cursor(updated_cursor)
        return WorkflowLogStreamResponse(cursor=encoded_cursor, chunks=[])

    pod_entries.sort(key=lambda item: item[0])

    now = time.time()
    effective_container = container or "main"
    tail_lines = tail_lines or 2000

    for display_name, pod_name, node_data in pod_entries:
        try:
            log_text = client.get_workflow_logs(
                resolved_namespace,
                workflow_name,
                pod_name,
                container=effective_container,
                tail_lines=tail_lines,
                since_seconds=since_seconds,
            )
        except ArgoNotFoundError:
            continue
        except ArgoClientError as exc:
            logger.warning("Error obteniendo logs de Argo para pod %s: %s", pod_name, exc)
            continue

        text = log_text or ""
        previous_offset = cursor_map.get(pod_name, 0)
        if previous_offset < 0 or previous_offset > len(text):
            previous_offset = 0

        if len(text) <= previous_offset:
            updated_cursor[pod_name] = len(text)
            continue

        content = text[previous_offset:]
        end_offset = len(text)
        updated_cursor[pod_name] = end_offset

        timestamp_value = now

        phase = node_data.get("phase")
        has_more = phase not in _FINAL_WORKFLOW_PHASES if isinstance(phase, str) else False

        chunks.append(
            WorkflowLogChunk(
                key=pod_name,
                nodeSlug=display_name,
                podName=pod_name,
                content=content,
                startOffset=previous_offset,
                endOffset=end_offset,
                hasMore=has_more,
                encoding="utf-8",
                timestamp=timestamp_value,
            )
        )

    encoded_cursor = _encode_cursor(updated_cursor)
    return WorkflowLogStreamResponse(cursor=encoded_cursor, chunks=chunks)
