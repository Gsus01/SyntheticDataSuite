import json
import logging
import os
import shlex
import subprocess
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
from catalog_loader import NodeTemplate
from catalog_adapter import components_to_catalog
from db import db_session, get_engine, should_run_migrations
from component_registry import ComponentRegistry, ensure_tables
from component_spec import ComponentSpec
from workflow_builder import (
    WorkflowGraphPayload,
    MissingArtifactError,
    UnknownTemplateError,
    build_workflow_plan,
    render_workflow_yaml,
    suggest_workflow_filename,
)
from argo_client import ArgoClient, ArgoClientError, ArgoNotFoundError
from workflow_store import (
    WorkflowStore,
    WorkflowStoreError,
    WorkflowStoreValidationError,
    WorkflowNotFoundError as WorkflowDefinitionNotFoundError,
    StoredWorkflowRecord,
    StoredWorkflowSummary,
)
from image_validator import (
    validate_images,
    validate_all_images,
    get_build_command_for_image,
    ImageValidationResult,
)


logger = logging.getLogger(__name__)
if not logger.handlers:
    # Configura logging básico para emitir trazas de la app en consola
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d | %(message)s",
    )

_FINAL_WORKFLOW_PHASES = {
    "Succeeded",
    "Failed",
    "Error",
    "Skipped",
    "Omitted",
    "Terminated",
}


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


class WorkflowCompileResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    manifest: str
    bucket: str
    manifest_filename: str = Field(..., alias="manifestFilename")
    node_slug_map: Dict[str, str] = Field(default_factory=dict, alias="nodeSlugMap")


class WorkflowSubmitPayload(WorkflowGraphPayload):
    workflow_id: Optional[str] = Field(None, alias="workflowId")


class WorkflowSavePayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    workflow_id: Optional[str] = Field(None, alias="workflowId")
    name: str
    description: Optional[str] = None
    session_id: str = Field(..., alias="sessionId")
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    compiled_manifest: Optional[str] = Field(None, alias="compiledManifest")
    manifest_filename: Optional[str] = Field(None, alias="manifestFilename")
    node_slug_map: Optional[Dict[str, str]] = Field(
        default_factory=dict, alias="nodeSlugMap"
    )


class WorkflowRecordResponse(BaseModel):
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


class WorkflowSummaryResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    workflow_id: str = Field(..., alias="workflowId")
    name: str
    description: Optional[str] = None
    created_at: str = Field(..., alias="createdAt")
    updated_at: str = Field(..., alias="updatedAt")
    last_submitted_at: Optional[str] = Field(None, alias="lastSubmittedAt")
    last_workflow_name: Optional[str] = Field(None, alias="lastWorkflowName")
    last_namespace: Optional[str] = Field(None, alias="lastNamespace")


def _record_to_response(record: StoredWorkflowRecord) -> WorkflowRecordResponse:
    return WorkflowRecordResponse.model_validate(record.model_dump(by_alias=True))


def _summary_to_response(summary: StoredWorkflowSummary) -> WorkflowSummaryResponse:
    return WorkflowSummaryResponse.model_validate(summary.model_dump(by_alias=True))


def _is_final_phase(phase: Optional[str]) -> bool:
    if not phase:
        return False
    return phase.strip().capitalize() in _FINAL_WORKFLOW_PHASES


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
    contentType: Optional[str] = Field(None, alias="contentType")
    exists: bool = False


class ArtifactPreviewResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    content: str
    truncated: bool
    content_type: Optional[str] = Field(None, alias="contentType")
    encoding: str = "utf-8"
    size: Optional[int] = None


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


class WorkflowLogsResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    workflow_name: str = Field(..., alias="workflowName")
    namespace: str
    logs: str


app = FastAPI(title="Synthetic Data Suite Backend", version="0.1.0")


@app.on_event("startup")
def _startup_registry() -> None:
    try:
        engine = get_engine()
    except RuntimeError as exc:
        logger.error("Database not configured: %s", exc)
        return

    if should_run_migrations():
        ensure_tables(engine)

    # Ensure registry tables are accessible on startup.
    try:
        with db_session() as session:
            _ = ComponentRegistry(session)
    except Exception as exc:
        logger.warning("Failed to initialize component registry: %s", exc)


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
    raw_base = artifact_name or (
        Path(original_filename).stem if original_filename else None
    )
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
    upload_kwargs: Dict[str, Any] = {"content_type": content_type}
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
        raise HTTPException(
            status_code=500, detail=f"Error subiendo a MinIO: {exc}"
        ) from exc
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
    """Return the active component catalog.

    This endpoint is consumed by the frontend to populate the node palette.
    Internally we read from the DB-backed registry and keep the response shape
    compatible with the legacy NodeTemplate models.
    """

    try:
        with db_session() as session:
            registry = ComponentRegistry(session)
            active_specs: List[ComponentSpec] = []
            for comp in registry.list_components():
                if not comp.active_version:
                    continue
                spec = registry.resolve_active_spec(comp.name)
                if spec:
                    active_specs.append(spec)
            return components_to_catalog(active_specs)
    except Exception as exc:  # pragma: no cover - surface upstream
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


class ComponentSummary(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str
    active_version: Optional[str] = Field(None, alias="activeVersion")
    created_at: Optional[str] = Field(None, alias="createdAt")
    updated_at: Optional[str] = Field(None, alias="updatedAt")


class ComponentVersionInfo(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str
    version: str
    spec: Dict[str, Any]
    created_at: Optional[str] = Field(None, alias="createdAt")


class ComponentRegisterPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    spec: Dict[str, Any]
    activate: bool = True


@app.get("/components", response_model=List[ComponentSummary])
def list_components() -> List[ComponentSummary]:
    try:
        with db_session() as session:
            registry = ComponentRegistry(session)
            items = []
            for comp in registry.list_components():
                items.append(
                    ComponentSummary(
                        name=comp.name,
                        activeVersion=comp.active_version,
                        createdAt=(
                            comp.created_at.isoformat() if comp.created_at else None
                        ),
                        updatedAt=(
                            comp.updated_at.isoformat() if comp.updated_at else None
                        ),
                    )
                )
            return items
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/components/{name}", response_model=List[ComponentVersionInfo])
def list_component_versions(name: str) -> List[ComponentVersionInfo]:
    try:
        with db_session() as session:
            registry = ComponentRegistry(session)
            versions = registry.list_versions(name)
            items: List[ComponentVersionInfo] = []
            for row in versions:
                items.append(
                    ComponentVersionInfo(
                        name=name,
                        version=row.version,
                        spec=row.spec_json,
                        createdAt=(
                            row.created_at.isoformat() if row.created_at else None
                        ),
                    )
                )
            return items
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/components/{name}/{version}", response_model=ComponentVersionInfo)
def get_component_version(name: str, version: str) -> ComponentVersionInfo:
    try:
        with db_session() as session:
            registry = ComponentRegistry(session)
            row = registry.get_version(name, version)
            if not row:
                raise HTTPException(
                    status_code=404, detail="Component version not found"
                )
            return ComponentVersionInfo(
                name=name,
                version=row.version,
                spec=row.spec_json,
                createdAt=(row.created_at.isoformat() if row.created_at else None),
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/components", response_model=ComponentSummary)
def register_component(payload: ComponentRegisterPayload) -> ComponentSummary:
    try:
        spec = ComponentSpec.model_validate(payload.spec)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Invalid ComponentSpec: {exc}"
        ) from exc

    try:
        with db_session() as session:
            registry = ComponentRegistry(session)
            comp = registry.register(spec, activate=payload.activate)
            return ComponentSummary(
                name=comp.name,
                activeVersion=comp.active_version,
                createdAt=(comp.created_at.isoformat() if comp.created_at else None),
                updatedAt=(comp.updated_at.isoformat() if comp.updated_at else None),
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/components/{name}/{version}/activate", response_model=ComponentSummary)
def activate_component_version(name: str, version: str) -> ComponentSummary:
    try:
        with db_session() as session:
            registry = ComponentRegistry(session)
            comp = registry.activate(name, version)
            if not comp:
                raise HTTPException(
                    status_code=404, detail="Component or version not found"
                )
            return ComponentSummary(
                name=comp.name,
                activeVersion=comp.active_version,
                createdAt=(comp.created_at.isoformat() if comp.created_at else None),
                updatedAt=(comp.updated_at.isoformat() if comp.updated_at else None),
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Image validation endpoints
# ---------------------------------------------------------------------------


class MissingImageInfo(BaseModel):
    image: str
    build_command: Optional[str] = Field(None, alias="buildCommand")

    model_config = ConfigDict(populate_by_name=True)


class ImageValidationResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    all_present: bool = Field(..., alias="allPresent")
    missing_images: List[MissingImageInfo] = Field(
        default_factory=list, alias="missingImages"
    )
    present_images: List[str] = Field(default_factory=list, alias="presentImages")
    error: Optional[str] = None


class ValidateImagesPayload(BaseModel):
    template_names: List[str] = Field(..., alias="templateNames")

    model_config = ConfigDict(populate_by_name=True)


@app.post("/images/validate", response_model=ImageValidationResponse)
def validate_workflow_images(payload: ValidateImagesPayload) -> ImageValidationResponse:
    """Validate that Docker images for the given templates exist in minikube."""
    result = validate_images(payload.template_names)

    missing_info = [
        MissingImageInfo(
            image=img,
            buildCommand=get_build_command_for_image(img),
        )
        for img in result.missing_images
    ]

    return ImageValidationResponse(
        allPresent=result.all_present,
        missingImages=missing_info,
        presentImages=result.present_images,
        error=result.error,
    )


@app.get("/images/validate-all", response_model=ImageValidationResponse)
def validate_all_workflow_images() -> ImageValidationResponse:
    """Validate all Docker images defined in the registry."""
    result = validate_all_images()

    missing_info = [
        MissingImageInfo(
            image=img,
            buildCommand=get_build_command_for_image(img),
        )
        for img in result.missing_images
    ]

    return ImageValidationResponse(
        allPresent=result.all_present,
        missingImages=missing_info,
        presentImages=result.present_images,
        error=result.error,
    )


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


@app.post("/workflow/compile", response_model=WorkflowCompileResponse)
def compile_workflow(payload: WorkflowGraphPayload) -> WorkflowCompileResponse:
    try:
        plan = build_workflow_plan(payload)
        yaml_content = render_workflow_yaml(plan)
        filename = suggest_workflow_filename(plan)
    except (UnknownTemplateError, MissingArtifactError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - unexpected
        raise HTTPException(
            status_code=500, detail=f"Failed to render workflow: {exc}"
        ) from exc

    node_slug_map = {node_plan.node_id: node_plan.slug for node_plan in plan.tasks()}
    return WorkflowCompileResponse(
        manifest=yaml_content,
        manifestFilename=filename,
        nodeSlugMap=node_slug_map,
        bucket=plan.bucket,
    )


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
        logger.warning(
            "No se pudo extraer el nombre del workflow de la salida: %s", stdout_text
        )
        raise RuntimeError(
            "No se pudo determinar el nombre del workflow desde la salida del comando 'argo submit'."
        )

    cli_output = stdout_text or stderr_text or None
    return ArgoSubmissionResult(
        workflow_name=workflow_name, namespace=namespace, cli_output=cli_output
    )


def _stat_object(client, bucket: str, key: str):
    try:
        return client.stat_object(bucket, key)
    except S3Error as exc:
        if exc.code in {"NoSuchKey", "NoSuchBucket"}:
            return None
        raise


def _stream_object(
    client, bucket: str, key: str, chunk_size: int = 1024 * 1024
) -> Iterable[bytes]:
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


def _fetch_workflow_logs_cli(
    workflow_name: str,
    namespace: str,
    follow: bool = False,
    pod_name: Optional[str] = None,
) -> str:
    """Execute `argo logs` and return its output."""

    argo_cli = os.getenv("ARGO_CLI_PATH", "argo")
    cmd = [argo_cli, "-n", namespace, "logs", workflow_name]
    if pod_name:
        cmd.append(pod_name)
        cmd.extend(["-c", "main"])
    if follow:
        cmd.append("--follow")

    cmd_display = " ".join(shlex.quote(part) for part in cmd)
    logger.info("Ejecutando comando de logs: %s", cmd_display)

    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Argo CLI no encontrado. Instala 'argo' o configura ARGO_CLI_PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.output or getattr(exc, "stderr", "") or str(exc)).strip()
        raise RuntimeError(detail or "Fallo al ejecutar 'argo logs'.") from exc


@app.post("/workflow/submit", response_model=WorkflowSubmitResponse)
def submit_workflow(payload: WorkflowSubmitPayload) -> WorkflowSubmitResponse:
    try:
        plan = build_workflow_plan(payload)
        yaml_content = render_workflow_yaml(plan)
        filename = suggest_workflow_filename(plan)
    except (UnknownTemplateError, MissingArtifactError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # pragma: no cover - unexpected
        raise HTTPException(status_code=500, detail=f"Failed to render workflow: {exc}")

    # Validate that all required Docker images exist in minikube
    template_names = [task.template_name for task in plan.tasks()]
    image_result = validate_images(template_names)

    if not image_result.all_present:
        missing_details = []
        for img in image_result.missing_images:
            build_cmd = get_build_command_for_image(img)
            if build_cmd:
                missing_details.append(f"  - {img}\n    Build: {build_cmd}")
            else:
                missing_details.append(f"  - {img}")

        error_msg = (
            f"Faltan imágenes Docker en minikube. El workflow fallaría con ErrImageNeverPull.\n\n"
            f"Imágenes faltantes:\n" + "\n".join(missing_details) + "\n\n"
            f"Para construir las imágenes, primero ejecuta:\n"
            f"  eval $(minikube docker-env)\n\n"
            f"Y luego construye cada imagen con los comandos indicados, o usa:\n"
            f"  ./build_all.sh"
        )
        raise HTTPException(status_code=400, detail=error_msg)

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
        raise HTTPException(
            status_code=500, detail=f"Error subiendo a MinIO: {exc}"
        ) from exc

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

    response = WorkflowSubmitResponse(
        workflowName=submission.workflow_name,
        namespace=submission.namespace,
        nodeSlugMap=node_slug_map,
        bucket=plan.bucket,
        key=manifest_key,
        manifestFilename=filename,
        cliOutput=submission.cli_output,
    )

    if payload.workflow_id:
        store = WorkflowStore()
        try:
            store.record_submission(
                payload.workflow_id,
                workflow_name=submission.workflow_name,
                namespace=submission.namespace,
                bucket=plan.bucket,
                key=manifest_key,
                manifest_filename=filename,
                cli_output=submission.cli_output,
                node_slug_map=node_slug_map,
            )
        except WorkflowDefinitionNotFoundError:
            logger.warning(
                "No se encontró el workflow guardado %s para actualizar metadata tras el envío.",
                payload.workflow_id,
            )
        except WorkflowStoreError as exc:
            logger.error(
                "No se pudo actualizar la metadata del workflow guardado %s: %s",
                payload.workflow_id,
                exc,
            )

    return response


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
                displayName=display_name,
                message=node.get("message"),
                progress=node.get("progress"),
                startedAt=node.get("startedAt"),
                finishedAt=node.get("finishedAt"),
            )

    updated_at = status_obj.get("finishedAt") or status_obj.get("startedAt")

    return WorkflowStatusResponse(
        workflowName=workflow_name,
        namespace=resolved_namespace,
        phase=phase,
        finished=finished,
        nodes=node_statuses,
        updatedAt=updated_at,
    )


@app.post("/workflow/output-artifacts", response_model=List[OutputArtifactInfo])
def get_output_artifacts(
    payload: WorkflowGraphPayload, node_id: str = Query(..., alias="nodeId")
) -> List[OutputArtifactInfo]:
    try:
        plan = build_workflow_plan(payload)
    except (UnknownTemplateError, MissingArtifactError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # pragma: no cover - unexpected
        raise HTTPException(
            status_code=500, detail=f"Failed to build workflow plan: {exc}"
        )

    node_plan = plan.nodes.get(node_id)
    if not node_plan:
        raise HTTPException(
            status_code=404, detail=f"Nodo '{node_id}' no encontrado en el grafo"
        )
    if node_plan.template.type != "output":
        raise HTTPException(
            status_code=400, detail="El nodo indicado no es de tipo salida"
        )

    try:
        client = get_minio_client()
    except RuntimeError as exc:
        logger.warning(
            "No se pudo inicializar cliente de MinIO para consultar salidas: %s", exc
        )
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
            contentType=None,
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
                info.contentType = getattr(stat, "content_type", None)
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
            raise HTTPException(
                status_code=404, detail="Artefacto no encontrado en MinIO"
            ) from exc
        raise HTTPException(
            status_code=500, detail=f"Error leyendo desde MinIO: {exc}"
        ) from exc

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
            raise HTTPException(
                status_code=404, detail="Artefacto no encontrado en MinIO"
            ) from exc
        raise HTTPException(
            status_code=500, detail=f"Error leyendo desde MinIO: {exc}"
        ) from exc
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


@app.get("/workflow/logs/stream", response_model=WorkflowLogsResponse)
def stream_workflow_logs(
    workflow_name: str = Query(..., alias="workflowName"),
    namespace: Optional[str] = Query(None),
    follow: bool = Query(False),
    pod_name: Optional[str] = Query(None, alias="podName"),
) -> WorkflowLogsResponse:
    resolved_namespace = namespace or os.getenv("ARGO_NAMESPACE", "argo")

    logger.info(
        "Logs solicitados (CLI): workflow=%s ns=%s follow=%s pod=%s",
        workflow_name,
        resolved_namespace,
        follow,
        pod_name,
    )

    try:
        logs = _fetch_workflow_logs_cli(
            workflow_name,
            resolved_namespace,
            follow=follow,
            pod_name=pod_name,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return WorkflowLogsResponse(
        workflowName=workflow_name,
        namespace=resolved_namespace,
        logs=logs,
    )


@app.post("/workflows", response_model=WorkflowRecordResponse)
def save_workflow_definition(payload: WorkflowSavePayload) -> WorkflowRecordResponse:
    store = WorkflowStore()
    try:
        record = store.save_definition(
            workflow_id=payload.workflow_id,
            name=payload.name,
            description=payload.description,
            session_id=payload.session_id,
            nodes=payload.nodes,
            edges=payload.edges,
            compiled_manifest=payload.compiled_manifest,
            manifest_filename=payload.manifest_filename,
            node_slug_map=payload.node_slug_map or {},
        )
    except WorkflowDefinitionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except WorkflowStoreValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except WorkflowStoreError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return _record_to_response(record)


@app.get("/workflows", response_model=List[WorkflowSummaryResponse])
def list_workflows() -> List[WorkflowSummaryResponse]:
    store = WorkflowStore()
    try:
        summaries = store.list_workflows()
    except WorkflowStoreError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return [_summary_to_response(summary) for summary in summaries]


@app.get("/workflows/{workflow_id}", response_model=WorkflowRecordResponse)
def get_workflow_definition(workflow_id: str) -> WorkflowRecordResponse:
    store = WorkflowStore()
    try:
        record = store.get_workflow(workflow_id)
    except WorkflowDefinitionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except WorkflowStoreError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return _record_to_response(record)
