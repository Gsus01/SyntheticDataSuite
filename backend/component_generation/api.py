from __future__ import annotations

import json
import mimetypes
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from component_generation.ingest import (
    DEFAULT_MAX_CHARS,
    ingest_paths,
)
from component_generation.run_manager import (
    ComponentGenerationRunManager,
    RunConflictError,
    RunInputFile,
    RunInvalidStateError,
    RunNotFoundError,
)


router = APIRouter(prefix="/component-generation", tags=["component-generation"])
RUN_MANAGER = ComponentGenerationRunManager()


class IngestFileInfo(BaseModel):
    filename: str
    size: int
    truncated: bool = False
    content_type: str | None = Field(default=None, alias="contentType")


class IngestResponse(BaseModel):
    combined: str
    files: List[IngestFileInfo]


class RunNodeState(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    state: str
    started_at: str | None = Field(default=None, alias="startedAt")
    finished_at: str | None = Field(default=None, alias="finishedAt")
    message: str | None = None


class RunResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    run_id: str = Field(..., alias="runId")
    status: str
    created_at: str = Field(..., alias="createdAt")
    updated_at: str = Field(..., alias="updatedAt")
    session_dir: str = Field(..., alias="sessionDir")
    input_files: List[IngestFileInfo] = Field(default_factory=list, alias="inputFiles")
    options: Dict[str, Any] = Field(default_factory=dict)
    node_states: Dict[str, RunNodeState] = Field(default_factory=dict, alias="nodeStates")
    pending_plan: Dict[str, Any] | None = Field(default=None, alias="pendingPlan")
    pending_pretty_plan: str | None = Field(default=None, alias="pendingPrettyPlan")
    generated_index: Dict[str, Dict[str, str]] = Field(
        default_factory=dict, alias="generatedIndex"
    )
    review_report: str | None = Field(default=None, alias="reviewReport")
    review_status: str | None = Field(default=None, alias="reviewStatus")
    integration_report: str | None = Field(default=None, alias="integrationReport")
    log_tail: List[str] = Field(default_factory=list, alias="logTail")
    error: str | None = None
    can_cancel: bool = Field(default=False, alias="canCancel")
    awaiting_decision: bool = Field(default=False, alias="awaitingDecision")
    last_seq: int = Field(default=0, alias="lastSeq")


class DecisionPayload(BaseModel):
    approved: bool
    feedback: str | None = None


class CancelAllRunsResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    canceled_run_ids: List[str] = Field(default_factory=list, alias="canceledRunIds")
    canceled_count: int = Field(default=0, alias="canceledCount")


class RunFilePreviewResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    content: str
    truncated: bool
    content_type: str | None = Field(default=None, alias="contentType")
    encoding: str = "utf-8"
    size: int | None = None
    file_name: str = Field(..., alias="fileName")
    file_path: str = Field(..., alias="filePath")
    is_binary: bool = Field(default=False, alias="isBinary")
    language_hint: str | None = Field(default=None, alias="languageHint")


_LANGUAGE_HINT_BY_EXTENSION: dict[str, str] = {
    ".py": "python",
    ".ipynb": "json",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".md": "markdown",
    ".txt": "text",
    ".csv": "text",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "jsx",
    ".sh": "bash",
}


def _safe_filename(name: str | None, *, fallback: str) -> str:
    if not name:
        return fallback
    return Path(name).name or fallback


def _to_run_response(snapshot: Dict[str, Any]) -> RunResponse:
    return RunResponse.model_validate(snapshot)


def _resolve_run_file_path(run_id: str, raw_path: str) -> Path:
    if not raw_path or not raw_path.strip():
        raise HTTPException(status_code=400, detail="Path is required")

    try:
        snapshot = RUN_MANAGER.get_run_snapshot(run_id)
    except RunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    session_dir_raw = snapshot.get("sessionDir")
    if not isinstance(session_dir_raw, str) or not session_dir_raw.strip():
        raise HTTPException(status_code=500, detail="Run session directory is not available")

    session_dir = Path(session_dir_raw).resolve()
    candidate = Path(raw_path.strip())
    resolved = (session_dir / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()

    try:
        resolved.relative_to(session_dir)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Path is outside the run session directory") from exc

    if not resolved.exists():
        raise HTTPException(status_code=404, detail="File not found")
    if not resolved.is_file():
        raise HTTPException(status_code=400, detail="Path is not a file")

    return resolved


def _guess_language_hint(path: Path, content_type: str | None) -> str | None:
    if content_type:
        normalized = content_type.lower()
        if normalized.startswith("image/"):
            return "image"
        if normalized in {"application/json", "text/json"}:
            return "json"
        if normalized in {"text/markdown"}:
            return "markdown"
        if normalized in {"application/x-yaml", "text/yaml", "text/x-yaml"}:
            return "yaml"

    filename_lower = path.name.lower()
    if filename_lower == "dockerfile":
        return "docker"

    return _LANGUAGE_HINT_BY_EXTENSION.get(path.suffix.lower())


def _decode_preview_bytes(data: bytes) -> tuple[str, bool]:
    if not data:
        return "", False
    if b"\x00" in data:
        return "", True

    try:
        return data.decode("utf-8"), False
    except UnicodeDecodeError:
        decoded = data.decode("utf-8", errors="replace")
        replacement_ratio = decoded.count("\ufffd") / max(len(decoded), 1)
        if replacement_ratio > 0.12:
            return "", True
        return decoded, False


@router.post("/ingest", response_model=IngestResponse)
async def ingest_files(
    files: List[UploadFile] = File(...),
    include_markdown: bool = Query(False, alias="includeMarkdown"),
    max_chars_per_file: int = Query(DEFAULT_MAX_CHARS, alias="maxCharsPerFile", gt=0),
) -> IngestResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    file_infos: List[IngestFileInfo] = []
    temp_paths: List[Path] = []

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        for idx, upload in enumerate(files, start=1):
            filename = _safe_filename(upload.filename, fallback=f"input-{idx}")
            path = temp_root / filename
            try:
                data = await upload.read()
            finally:
                await upload.close()
            path.write_bytes(data)
            temp_paths.append(path)
            file_infos.append(
                IngestFileInfo(
                    filename=filename,
                    size=len(data),
                    contentType=upload.content_type,
                )
            )

        try:
            combined = ingest_paths(
                temp_paths,
                max_chars_per_file=max_chars_per_file,
                include_markdown=include_markdown,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return IngestResponse(combined=combined, files=file_infos)


@router.post("/runs", response_model=RunResponse)
async def create_run(
    files: List[UploadFile] = File(...),
    include_markdown: bool = Form(False, alias="includeMarkdown"),
    max_chars_per_file: int = Form(DEFAULT_MAX_CHARS, alias="maxCharsPerFile"),
    auto_approve: bool = Form(False, alias="autoApprove"),
    run_integration: bool = Form(False, alias="runIntegration"),
    no_llm: bool = Form(False, alias="noLlm"),
    no_structured_output: bool = Form(False, alias="noStructuredOutput"),
    provider: str = Form("ollama"),
    model: str = Form("qwen3:14b"),
    temperature: float = Form(0.0),
    ollama_url: str = Form("http://localhost:11434", alias="ollamaUrl"),
    openrouter_url: str = Form("https://openrouter.ai/api/v1", alias="openrouterUrl"),
    openrouter_key: str | None = Form(None, alias="openrouterKey"),
    openrouter_provider: str | None = Form(None, alias="openrouterProvider"),
) -> RunResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    run_files: list[RunInputFile] = []
    for idx, upload in enumerate(files, start=1):
        filename = _safe_filename(upload.filename, fallback=f"input-{idx}")
        try:
            data = await upload.read()
        finally:
            await upload.close()
        run_files.append(
            RunInputFile(
                filename=filename,
                content=data,
                content_type=upload.content_type,
            )
        )

    try:
        snapshot = RUN_MANAGER.start_run(
            input_files=run_files,
            options={
                "include_markdown": include_markdown,
                "max_chars_per_file": max_chars_per_file,
                "auto_approve": auto_approve,
                "run_integration": run_integration,
                "no_llm": no_llm,
                "no_structured_output": no_structured_output,
                "provider": provider,
                "model": model,
                "temperature": temperature,
                "ollama_url": ollama_url,
                "openrouter_url": openrouter_url,
                "openrouter_key": openrouter_key,
                "openrouter_provider": openrouter_provider,
            },
        )
    except RunConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - unexpected
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return _to_run_response(snapshot)


@router.get("/runs/{run_id}", response_model=RunResponse)
def get_run(run_id: str) -> RunResponse:
    try:
        snapshot = RUN_MANAGER.get_run_snapshot(run_id)
    except RunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _to_run_response(snapshot)


@router.get("/runs/{run_id}/files/download")
def download_run_file(
    run_id: str,
    path: str = Query(..., min_length=1),
):
    file_path = _resolve_run_file_path(run_id, path)
    media_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    response = FileResponse(file_path, media_type=media_type, filename=file_path.name)
    response.headers["Content-Disposition"] = f'attachment; filename="{file_path.name}"'
    return response


@router.get("/runs/{run_id}/files/preview", response_model=RunFilePreviewResponse)
def preview_run_file(
    run_id: str,
    path: str = Query(..., min_length=1),
    max_bytes: int = Query(65536, ge=128, le=4 * 1024 * 1024, alias="maxBytes"),
) -> RunFilePreviewResponse:
    file_path = _resolve_run_file_path(run_id, path)
    file_size = file_path.stat().st_size
    read_size = min(max_bytes, file_size) if file_size > 0 else 0

    if read_size > 0:
        with file_path.open("rb") as handle:
            data = handle.read(read_size)
    else:
        data = b""

    content, is_binary = _decode_preview_bytes(data)
    content_type = mimetypes.guess_type(file_path.name)[0]
    language_hint = _guess_language_hint(file_path, content_type)

    return RunFilePreviewResponse(
        content="" if is_binary else content,
        truncated=file_size > len(data),
        contentType=content_type,
        encoding="utf-8",
        size=file_size,
        fileName=file_path.name,
        filePath=str(file_path),
        isBinary=is_binary,
        languageHint=language_hint,
    )


@router.get("/runs/{run_id}/events")
def stream_run_events(
    run_id: str,
    since_seq: int = Query(0, alias="sinceSeq", ge=0),
) -> StreamingResponse:
    try:
        RUN_MANAGER.get_run_snapshot(run_id)
    except RunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    def event_stream():
        last_seq = since_seq
        next_heartbeat = time.monotonic() + 15.0
        yield "retry: 1500\n\n"
        while True:
            try:
                events, terminal = RUN_MANAGER.get_events_since(run_id, last_seq)
            except RunNotFoundError:
                break

            if events:
                for event in events:
                    seq = int(event.get("seq") or 0)
                    last_seq = max(last_seq, seq)
                    event_type = str(event.get("type") or "message")
                    payload = json.dumps(event, ensure_ascii=False)
                    yield f"id: {seq}\n"
                    yield f"event: {event_type}\n"
                    yield f"data: {payload}\n\n"
                next_heartbeat = time.monotonic() + 15.0
            else:
                now = time.monotonic()
                if now >= next_heartbeat:
                    yield ": ping\n\n"
                    next_heartbeat = now + 15.0

            if terminal and not events:
                break

            time.sleep(0.35)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/runs/{run_id}/decision", response_model=RunResponse)
def submit_run_decision(run_id: str, payload: DecisionPayload) -> RunResponse:
    try:
        snapshot = RUN_MANAGER.submit_decision(
            run_id,
            approved=payload.approved,
            feedback=payload.feedback,
        )
    except RunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RunInvalidStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _to_run_response(snapshot)


@router.post("/runs/{run_id}/cancel", response_model=RunResponse)
def cancel_run(run_id: str) -> RunResponse:
    try:
        snapshot = RUN_MANAGER.cancel_run(run_id)
    except RunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _to_run_response(snapshot)


@router.post("/runs/cancel-all", response_model=CancelAllRunsResponse)
def cancel_all_runs() -> CancelAllRunsResponse:
    result = RUN_MANAGER.cancel_all_active_runs()
    return CancelAllRunsResponse.model_validate(result)
