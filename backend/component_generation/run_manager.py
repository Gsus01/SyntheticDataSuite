from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import queue
import sys
import threading
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable
from uuid import uuid4

from component_generation.context import (
    load_analyst_context,
    load_developer_context,
    load_repair_context,
)
from component_generation.llm import make_llm
from component_generation.state import PipelineState
from component_generation.workflow import build_graph

logger = logging.getLogger(__name__)

ACTIVE_RUN_STATUSES = {"queued", "running", "waiting_decision"}
TERMINAL_RUN_STATUSES = {"succeeded", "failed", "canceled"}
MAX_LOG_LINES_IN_MEMORY = 5000
LOG_TAIL_LINES = 300

KNOWN_GRAPH_NODES = [
    "load",
    "analyst",
    "hitl",
    "developer",
    "tester",
    "repair",
    "integration",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_openrouter_provider(raw: str | None) -> Dict[str, Any] | None:
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    if text.startswith("{"):
        return json.loads(text)
    order = [part.strip() for part in text.split(",") if part.strip()]
    if not order:
        return None
    return {"order": order}


def _default_node_states() -> Dict[str, Dict[str, Any]]:
    return {
        node: {
            "state": "pending",
            "startedAt": None,
            "finishedAt": None,
            "message": None,
        }
        for node in KNOWN_GRAPH_NODES
    }


@dataclass(slots=True)
class RunInputFile:
    filename: str
    content: bytes
    content_type: str | None = None


@dataclass(slots=True)
class RunEvent:
    seq: int
    timestamp: str
    type: str
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seq": self.seq,
            "timestamp": self.timestamp,
            "type": self.type,
            "payload": self.payload,
        }


@dataclass(slots=True)
class RunRecord:
    run_id: str
    created_at: str
    updated_at: str
    status: str
    session_dir: str
    input_files: list[Dict[str, Any]]
    options: Dict[str, Any]
    node_states: Dict[str, Dict[str, Any]]
    pending_plan: Dict[str, Any] | None = None
    generated_index: Dict[str, Dict[str, str]] = field(default_factory=dict)
    review_report: str | None = None
    review_status: str | None = None
    integration_report: str | None = None
    error: str | None = None
    pending_pretty_plan: str | None = None
    cancel_requested: bool = False
    events: list[RunEvent] = field(default_factory=list)
    log_lines: list[str] = field(default_factory=list)
    next_seq: int = 1
    process: mp.Process | None = None
    event_queue: Any | None = None
    decision_queue: Any | None = None
    monitor_thread: threading.Thread | None = None


class RunManagerError(RuntimeError):
    """Base run manager error."""


class RunConflictError(RunManagerError):
    """Raised when an active run already exists."""


class RunNotFoundError(RunManagerError):
    """Raised when a run id does not exist."""


class RunInvalidStateError(RunManagerError):
    """Raised when an action is not valid for the run state."""


class _EventStreamWriter:
    def __init__(
        self,
        emit_line: Callable[[str, str, str], None],
        *,
        level: str,
        source: str,
    ) -> None:
        self._emit_line = emit_line
        self._level = level
        self._source = source
        self._buffer = ""

    def write(self, data: str) -> int:
        if not data:
            return 0
        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._emit_line(line.rstrip("\r"), self._level, self._source)
        return len(data)

    def flush(self) -> None:
        if self._buffer:
            self._emit_line(self._buffer.rstrip("\r"), self._level, self._source)
            self._buffer = ""

    def isatty(self) -> bool:  # pragma: no cover - compatibility
        return False


class _QueueLogHandler(logging.Handler):
    def __init__(self, emit_line: Callable[[str, str, str], None]) -> None:
        super().__init__()
        self._emit_line = emit_line

    def emit(self, record: logging.LogRecord) -> None:
        try:
            rendered = self.format(record)
        except Exception:  # pragma: no cover - defensive
            rendered = record.getMessage()
        lines = rendered.splitlines() or [rendered]
        for line in lines:
            self._emit_line(line, record.levelname.upper(), record.name)


def _worker_main(
    *,
    run_id: str,
    session_dir: str,
    input_paths: list[str],
    options: Dict[str, Any],
    event_queue: Any,
    decision_queue: Any,
) -> None:
    def emit(event_type: str, payload: Dict[str, Any] | None = None) -> None:
        event_queue.put(
            {
                "timestamp": _utc_now_iso(),
                "type": event_type,
                "payload": payload or {},
            }
        )

    def emit_log_line(line: str, level: str = "INFO", source: str = "worker") -> None:
        text = str(line).rstrip("\r")
        emit(
            "log_line",
            {"line": text, "level": level.upper(), "source": source},
        )

    def wait_for_decision(plan: Dict[str, Any]) -> Dict[str, Any]:
        del plan
        while True:
            message = decision_queue.get()
            if not isinstance(message, dict):
                continue
            if message.get("action") != "decision":
                continue
            approved = bool(message.get("approved"))
            feedback = str(message.get("feedback") or "").strip()
            emit(
                "decision_received",
                {"approved": approved, "hasFeedback": bool(feedback)},
            )
            return {"approved": approved, "feedback": feedback}

    emit("run_started", {"runId": run_id})
    emit_log_line(f"run {run_id} started", "INFO", "run_manager")

    root_logger = logging.getLogger()
    previous_handlers = list(root_logger.handlers)
    previous_level = root_logger.level
    queue_handler = _QueueLogHandler(emit_log_line)
    queue_handler.setLevel(logging.DEBUG)
    queue_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root_logger.handlers = [queue_handler]
    root_logger.setLevel(logging.INFO)

    stdout_writer = _EventStreamWriter(
        emit_log_line,
        level="STDOUT",
        source="stdout",
    )
    stderr_writer = _EventStreamWriter(
        emit_log_line,
        level="STDERR",
        source="stderr",
    )
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = stdout_writer  # type: ignore[assignment]
    sys.stderr = stderr_writer  # type: ignore[assignment]

    try:
        llm = None
        if not bool(options.get("no_llm")):
            provider = str(options.get("provider") or "ollama")
            model = str(options.get("model") or "qwen3:14b")
            temperature = float(options.get("temperature") or 0.0)
            if provider.lower() == "openrouter":
                llm = make_llm(
                    provider=provider,
                    model=model,
                    base_url=str(
                        options.get("openrouter_url")
                        or "https://openrouter.ai/api/v1"
                    ),
                    temperature=temperature,
                    api_key=str(options.get("openrouter_key") or "")
                    or os.getenv("OPENROUTER_API_KEY"),
                    openrouter_provider=_parse_openrouter_provider(
                        options.get("openrouter_provider")
                    ),
                )
            else:
                llm = make_llm(
                    provider=provider,
                    model=model,
                    base_url=str(
                        options.get("ollama_url") or "http://localhost:11434"
                    ),
                    temperature=temperature,
                )

        state: PipelineState = {
            "input_paths": input_paths,
            "include_markdown": bool(options.get("include_markdown")),
            "max_chars_per_file": int(options.get("max_chars_per_file") or 200_000),
            "out_dir": str(Path(session_dir).parent),
            "session_dir": session_dir,
            "analyst_context": load_analyst_context(),
            "developer_context": load_developer_context(),
            "repair_context": load_repair_context(),
            "auto_approve": bool(options.get("auto_approve")),
            "trace": True,
            "run_integration": bool(options.get("run_integration")),
            "feedback": "",
            "repair_attempts": 0,
            "llm": llm,
            "llm_provider": str(options.get("provider") or "ollama"),
            "llm_model": str(options.get("model") or "qwen3:14b"),
            "llm_temperature": float(options.get("temperature") or 0.0),
            "structured_output": not bool(options.get("no_structured_output")),
            "disable_llm": bool(options.get("no_llm")),
            "hitl_mode": "api",
            "event_callback": emit,
            "hitl_decision_getter": wait_for_decision,
        }

        graph = build_graph(emit_event=emit)
        final_state = graph.invoke(state)
        stdout_writer.flush()
        stderr_writer.flush()
        emit(
            "run_finished",
            {
                "reviewStatus": final_state.get("review_status"),
                "reviewReport": final_state.get("review_report"),
                "integrationReport": final_state.get("integration_report"),
                "generatedIndex": final_state.get("generated_index") or {},
                "sessionDir": session_dir,
            },
        )
    except Exception as exc:
        traceback_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        for line in traceback_text.splitlines():
            emit_log_line(line, "ERROR", "traceback")
        emit("run_failed", {"error": str(exc)})
        logger.exception("component generation run %s failed", run_id)
        raise
    finally:
        try:
            stdout_writer.flush()
            stderr_writer.flush()
        except Exception:  # pragma: no cover - defensive
            pass
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        root_logger.handlers = previous_handlers
        root_logger.setLevel(previous_level)


class ComponentGenerationRunManager:
    def __init__(self, output_root: Path | None = None) -> None:
        base_root = output_root
        if base_root is None:
            repo_root = Path(__file__).resolve().parents[2]
            base_root = repo_root / "components" / "generated-sessions"
        self._output_root = base_root.resolve()
        self._output_root.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._runs: Dict[str, RunRecord] = {}
        self._active_run_id: str | None = None
        self._mp_ctx = mp.get_context("spawn")

    def start_run(
        self, *, input_files: Iterable[RunInputFile], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        materialized_files = list(input_files)
        if not materialized_files:
            raise ValueError("No files provided")

        normalized_options = self._normalize_options(options)
        public_options = {
            key: value
            for key, value in normalized_options.items()
            if key != "openrouter_key"
        }

        with self._lock:
            self._ensure_no_active_run_locked()

            run_id = uuid4().hex
            now = _utc_now_iso()
            session_dir = (self._output_root / run_id).resolve()
            inputs_dir = session_dir / "inputs"
            session_dir.mkdir(parents=True, exist_ok=True)
            inputs_dir.mkdir(parents=True, exist_ok=True)

            input_paths: list[str] = []
            input_meta: list[Dict[str, Any]] = []
            for idx, file_payload in enumerate(materialized_files, start=1):
                base_name = Path(file_payload.filename or "").name or f"input-{idx}"
                target = inputs_dir / f"{idx:02d}-{base_name}"
                target.write_bytes(file_payload.content)
                input_paths.append(str(target))
                input_meta.append(
                    {
                        "filename": base_name,
                        "storedPath": str(target),
                        "size": len(file_payload.content),
                        "contentType": file_payload.content_type,
                    }
                )

            event_queue: Any = self._mp_ctx.Queue()
            decision_queue: Any = self._mp_ctx.Queue()

            process = self._mp_ctx.Process(
                target=_worker_main,
                kwargs={
                    "run_id": run_id,
                    "session_dir": str(session_dir),
                    "input_paths": input_paths,
                    "options": normalized_options,
                    "event_queue": event_queue,
                    "decision_queue": decision_queue,
                },
                daemon=True,
            )

            record = RunRecord(
                run_id=run_id,
                created_at=now,
                updated_at=now,
                status="queued",
                session_dir=str(session_dir),
                input_files=input_meta,
                options=public_options,
                node_states=_default_node_states(),
                process=process,
                event_queue=event_queue,
                decision_queue=decision_queue,
            )

            self._runs[run_id] = record
            self._active_run_id = run_id
            self._append_event_locked(
                record,
                "run_queued",
                {"runId": run_id, "sessionDir": str(session_dir)},
            )
            self._persist_run_meta_locked(record)

            process.start()
            monitor = threading.Thread(
                target=self._monitor_run,
                args=(run_id,),
                daemon=True,
                name=f"component-gen-monitor-{run_id[:8]}",
            )
            record.monitor_thread = monitor
            monitor.start()

            return self._snapshot_locked(record)

    def get_run_snapshot(self, run_id: str) -> Dict[str, Any]:
        with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                raise RunNotFoundError(f"Run '{run_id}' not found")
            return self._snapshot_locked(record)

    def get_events_since(
        self, run_id: str, since_seq: int
    ) -> tuple[list[Dict[str, Any]], bool]:
        with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                raise RunNotFoundError(f"Run '{run_id}' not found")

            events = [event.to_dict() for event in record.events if event.seq > since_seq]
            terminal = record.status in TERMINAL_RUN_STATUSES
            return events, terminal

    def submit_decision(
        self, run_id: str, *, approved: bool, feedback: str | None
    ) -> Dict[str, Any]:
        clean_feedback = (feedback or "").strip()
        with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                raise RunNotFoundError(f"Run '{run_id}' not found")
            if record.status != "waiting_decision":
                raise RunInvalidStateError(
                    f"Run '{run_id}' is not waiting for decision"
                )
            if not record.decision_queue:
                raise RunInvalidStateError(
                    f"Run '{run_id}' cannot receive decisions right now"
                )

            record.decision_queue.put(
                {
                    "action": "decision",
                    "approved": bool(approved),
                    "feedback": clean_feedback,
                }
            )
            self._append_event_locked(
                record,
                "decision_submitted",
                {"approved": bool(approved), "hasFeedback": bool(clean_feedback)},
            )
            self._persist_run_meta_locked(record)
            return self._snapshot_locked(record)

    def cancel_run(self, run_id: str) -> Dict[str, Any]:
        with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                raise RunNotFoundError(f"Run '{run_id}' not found")
            if record.status in TERMINAL_RUN_STATUSES:
                return self._snapshot_locked(record)

            self._cancel_record_locked(record, reason="user_requested")
            return self._snapshot_locked(record)

    def cancel_all_active_runs(self) -> Dict[str, Any]:
        with self._lock:
            canceled_run_ids: list[str] = []
            for record in self._runs.values():
                if record.status not in ACTIVE_RUN_STATUSES:
                    continue
                self._cancel_record_locked(record, reason="user_requested_batch")
                canceled_run_ids.append(record.run_id)
            return {
                "canceledRunIds": canceled_run_ids,
                "canceledCount": len(canceled_run_ids),
            }

    def _monitor_run(self, run_id: str) -> None:
        while True:
            with self._lock:
                record = self._runs.get(run_id)
                if record is None:
                    return
                event_queue = record.event_queue
                process = record.process

            if event_queue is None or process is None:
                return

            handled_event = False
            try:
                raw_event = event_queue.get(timeout=0.35)
                handled_event = True
                self._handle_worker_event(run_id, raw_event)
            except queue.Empty:
                pass
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Error while consuming run event for %s: %s", run_id, exc)

            if handled_event:
                continue

            if not process.is_alive():
                # Drain remaining queued events before finalizing status.
                while True:
                    try:
                        raw_event = event_queue.get_nowait()
                    except queue.Empty:
                        break
                    self._handle_worker_event(run_id, raw_event)

                with self._lock:
                    record = self._runs.get(run_id)
                    if record is None:
                        return
                    exit_code = process.exitcode
                    if record.status not in TERMINAL_RUN_STATUSES:
                        if record.cancel_requested:
                            record.status = "canceled"
                            record.error = record.error or "Run canceled by user."
                            self._append_event_locked(
                                record,
                                "run_canceled",
                                {"reason": "terminated", "exitCode": exit_code},
                            )
                        elif exit_code == 0:
                            record.status = "succeeded"
                            self._append_event_locked(
                                record, "run_finished", {"exitCode": exit_code}
                            )
                        else:
                            record.status = "failed"
                            record.error = (
                                record.error
                                or f"Component generation worker exited with code {exit_code}"
                            )
                            self._append_event_locked(
                                record,
                                "run_failed",
                                {"error": record.error, "exitCode": exit_code},
                            )
                    self._clear_active_run_locked(run_id)
                    self._persist_run_meta_locked(record)
                return

    def _handle_worker_event(self, run_id: str, raw_event: Any) -> None:
        if not isinstance(raw_event, dict):
            return

        event_type = str(raw_event.get("type") or "log")
        payload_raw = raw_event.get("payload")
        payload = payload_raw if isinstance(payload_raw, dict) else {}
        event_timestamp = raw_event.get("timestamp")
        timestamp = (
            str(event_timestamp).strip()
            if isinstance(event_timestamp, str) and event_timestamp.strip()
            else _utc_now_iso()
        )

        with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                return

            self._append_event_locked(record, event_type, payload, timestamp=timestamp)

            if event_type == "run_started":
                if record.status not in TERMINAL_RUN_STATUSES:
                    record.status = "running"
            elif event_type == "log_line":
                line = payload.get("line")
                if isinstance(line, str):
                    level = str(payload.get("level") or "INFO").upper()
                    source = str(payload.get("source") or "worker")
                    rendered_line = f"[{level}] {source}: {line}" if line else ""
                    self._append_log_line_locked(record, rendered_line)
            elif event_type == "plan_proposed":
                plan = payload.get("plan")
                if isinstance(plan, dict):
                    record.pending_plan = plan
                pretty_plan = payload.get("prettyPlan")
                if isinstance(pretty_plan, str):
                    record.pending_pretty_plan = pretty_plan
            elif event_type == "waiting_decision":
                if record.status not in TERMINAL_RUN_STATUSES:
                    record.status = "waiting_decision"
                plan = payload.get("plan")
                if isinstance(plan, dict):
                    record.pending_plan = plan
                pretty_plan = payload.get("prettyPlan")
                if isinstance(pretty_plan, str):
                    record.pending_pretty_plan = pretty_plan
            elif event_type == "resumed":
                if record.status not in TERMINAL_RUN_STATUSES:
                    record.status = "running"
            elif event_type == "node_started":
                node_name = str(payload.get("node") or "").strip()
                if node_name:
                    node_state = record.node_states.setdefault(
                        node_name,
                        {
                            "state": "pending",
                            "startedAt": None,
                            "finishedAt": None,
                            "message": None,
                        },
                    )
                    node_state["state"] = "running"
                    node_state["startedAt"] = timestamp
                    node_state["message"] = None
            elif event_type == "node_completed":
                node_name = str(payload.get("node") or "").strip()
                if node_name:
                    node_state = record.node_states.setdefault(
                        node_name,
                        {
                            "state": "pending",
                            "startedAt": None,
                            "finishedAt": None,
                            "message": None,
                        },
                    )
                    node_state["state"] = "completed"
                    node_state["finishedAt"] = timestamp
            elif event_type == "node_failed":
                node_name = str(payload.get("node") or "").strip()
                error_msg = str(payload.get("error") or "").strip() or "Node failed"
                if node_name:
                    node_state = record.node_states.setdefault(
                        node_name,
                        {
                            "state": "pending",
                            "startedAt": None,
                            "finishedAt": None,
                            "message": None,
                        },
                    )
                    node_state["state"] = "failed"
                    node_state["finishedAt"] = timestamp
                    node_state["message"] = error_msg
                if record.status not in TERMINAL_RUN_STATUSES:
                    record.status = "failed"
                record.error = error_msg
            elif event_type == "run_finished":
                if record.status not in TERMINAL_RUN_STATUSES:
                    record.status = "succeeded"
                generated_index = payload.get("generatedIndex")
                if isinstance(generated_index, dict):
                    record.generated_index = generated_index
                review_report = payload.get("reviewReport")
                if isinstance(review_report, str):
                    record.review_report = review_report
                review_status = payload.get("reviewStatus")
                if isinstance(review_status, str):
                    record.review_status = review_status
                integration_report = payload.get("integrationReport")
                if isinstance(integration_report, str):
                    record.integration_report = integration_report
                self._clear_active_run_locked(run_id)
            elif event_type == "run_failed":
                if record.status not in TERMINAL_RUN_STATUSES:
                    record.status = "failed"
                record.error = str(payload.get("error") or "Run failed")
                self._clear_active_run_locked(run_id)

            self._persist_run_meta_locked(record)

    def _normalize_options(self, options: Dict[str, Any]) -> Dict[str, Any]:
        normalized = {
            "include_markdown": bool(options.get("include_markdown", False)),
            "max_chars_per_file": int(options.get("max_chars_per_file") or 200_000),
            "auto_approve": bool(options.get("auto_approve", False)),
            "run_integration": bool(options.get("run_integration", False)),
            "no_llm": bool(options.get("no_llm", False)),
            "no_structured_output": bool(options.get("no_structured_output", False)),
            "provider": str(options.get("provider") or "ollama"),
            "model": str(options.get("model") or "qwen3:14b"),
            "temperature": float(options.get("temperature") or 0.0),
            "ollama_url": str(options.get("ollama_url") or "http://localhost:11434"),
            "openrouter_url": str(
                options.get("openrouter_url") or "https://openrouter.ai/api/v1"
            ),
            "openrouter_key": options.get("openrouter_key"),
            "openrouter_provider": options.get("openrouter_provider"),
        }
        if normalized["max_chars_per_file"] <= 0:
            raise ValueError("maxCharsPerFile must be greater than zero")
        if normalized["provider"].lower() not in {"ollama", "openrouter"}:
            raise ValueError("provider must be either 'ollama' or 'openrouter'")
        return normalized

    def _ensure_no_active_run_locked(self) -> None:
        if not self._active_run_id:
            return
        active = self._runs.get(self._active_run_id)
        if not active:
            self._active_run_id = None
            return
        if active.status in ACTIVE_RUN_STATUSES:
            raise RunConflictError(
                f"Run '{active.run_id}' is still active with status '{active.status}'"
            )
        self._active_run_id = None

    def _clear_active_run_locked(self, run_id: str) -> None:
        if self._active_run_id == run_id:
            self._active_run_id = None

    def _cancel_record_locked(self, record: RunRecord, *, reason: str) -> None:
        run_id = record.run_id
        record.cancel_requested = True
        record.status = "canceled"
        record.error = record.error or "Run canceled by user."
        self._append_event_locked(
            record,
            "run_canceled",
            {"reason": reason},
        )
        if record.process and record.process.is_alive():
            try:
                record.process.terminate()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to terminate run %s: %s", run_id, exc)
        self._clear_active_run_locked(run_id)
        self._persist_run_meta_locked(record)

    def _append_event_locked(
        self,
        record: RunRecord,
        event_type: str,
        payload: Dict[str, Any] | None = None,
        *,
        timestamp: str | None = None,
    ) -> None:
        event_time = timestamp or _utc_now_iso()
        event = RunEvent(
            seq=record.next_seq,
            timestamp=event_time,
            type=event_type,
            payload=payload or {},
        )
        record.events.append(event)
        record.next_seq += 1
        record.updated_at = event_time

        events_path = Path(record.session_dir) / "events.jsonl"
        events_path.parent.mkdir(parents=True, exist_ok=True)
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")

    def _append_log_line_locked(self, record: RunRecord, line: str) -> None:
        if line is None:
            return
        record.log_lines.append(line)
        if len(record.log_lines) > MAX_LOG_LINES_IN_MEMORY:
            record.log_lines = record.log_lines[-MAX_LOG_LINES_IN_MEMORY:]

        logs_path = Path(record.session_dir) / "logs.txt"
        logs_path.parent.mkdir(parents=True, exist_ok=True)
        with logs_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def _snapshot_locked(self, record: RunRecord) -> Dict[str, Any]:
        return {
            "runId": record.run_id,
            "status": record.status,
            "createdAt": record.created_at,
            "updatedAt": record.updated_at,
            "sessionDir": record.session_dir,
            "inputFiles": record.input_files,
            "options": record.options,
            "nodeStates": record.node_states,
            "pendingPlan": record.pending_plan,
            "pendingPrettyPlan": record.pending_pretty_plan,
            "generatedIndex": record.generated_index,
            "reviewReport": record.review_report,
            "reviewStatus": record.review_status,
            "integrationReport": record.integration_report,
            "logTail": record.log_lines[-LOG_TAIL_LINES:],
            "error": record.error,
            "canCancel": record.status in ACTIVE_RUN_STATUSES,
            "awaitingDecision": record.status == "waiting_decision",
            "lastSeq": record.next_seq - 1,
        }

    def _persist_run_meta_locked(self, record: RunRecord) -> None:
        snapshot = self._snapshot_locked(record)
        meta_path = Path(record.session_dir) / "run_meta.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(
            json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
