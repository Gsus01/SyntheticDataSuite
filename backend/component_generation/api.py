from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
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
    generated_index: Dict[str, Dict[str, str]] = Field(
        default_factory=dict, alias="generatedIndex"
    )
    review_report: str | None = Field(default=None, alias="reviewReport")
    review_status: str | None = Field(default=None, alias="reviewStatus")
    integration_report: str | None = Field(default=None, alias="integrationReport")
    error: str | None = None
    can_cancel: bool = Field(default=False, alias="canCancel")
    awaiting_decision: bool = Field(default=False, alias="awaitingDecision")
    last_seq: int = Field(default=0, alias="lastSeq")


class DecisionPayload(BaseModel):
    approved: bool
    feedback: str | None = None


def _safe_filename(name: str | None, *, fallback: str) -> str:
    if not name:
        return fallback
    return Path(name).name or fallback


def _to_run_response(snapshot: Dict[str, Any]) -> RunResponse:
    return RunResponse.model_validate(snapshot)


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
