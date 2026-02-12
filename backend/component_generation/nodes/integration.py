from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib import error as url_error
from urllib import request as url_request

from component_generation.state import PipelineState
from component_spec import ComponentSpec

logger = logging.getLogger(__name__)


def _emit_event(state: PipelineState, event_type: str, payload: Dict[str, Any] | None = None) -> None:
    callback = state.get("event_callback")
    if not callable(callback):
        return
    try:
        callback(event_type, payload or {})
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("integration: failed to emit event %s: %s", event_type, exc)


def _load_spec(spec_path: Path) -> ComponentSpec:
    if not spec_path.exists():
        raise RuntimeError(f"Missing ComponentSpec.json: {spec_path}")
    try:
        payload = json.loads(spec_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Invalid JSON in {spec_path}: {exc}") from exc
    try:
        spec = ComponentSpec.model_validate(payload)
    except Exception as exc:
        raise RuntimeError(f"Invalid ComponentSpec at {spec_path}: {exc}") from exc
    if not spec.runtime or not spec.runtime.image.strip():
        raise RuntimeError(f"ComponentSpec runtime.image missing for {spec_path}")
    return spec


def _iter_component_files(meta: Dict[str, str]) -> List[Dict[str, str]]:
    files: List[Dict[str, str]] = []
    for file_name, file_path in sorted(meta.items(), key=lambda item: item[0]):
        if file_name == "folder":
            continue
        files.append({"name": file_name, "path": str(file_path)})
    return files


def _build_summary(
    generated_index: Dict[str, Dict[str, str]],
    session_dir: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    summary_components: List[Dict[str, Any]] = []
    build_targets: List[Dict[str, Any]] = []
    components_root = session_dir / "components"

    for component_name, meta in sorted(generated_index.items(), key=lambda item: item[0]):
        folder_path = Path(meta.get("folder", "")).resolve()
        spec_path = folder_path / "ComponentSpec.json"
        spec = _load_spec(spec_path)

        component_type = spec.metadata.type
        if folder_path.exists():
            try:
                rel_parts = folder_path.relative_to(components_root).parts
                if rel_parts:
                    component_type = rel_parts[0]
            except ValueError:
                pass

        component_entry = {
            "name": spec.metadata.name,
            "title": spec.metadata.title or spec.metadata.name,
            "version": spec.metadata.version,
            "type": component_type,
            "image": spec.runtime.image if spec.runtime else "",
            "description": spec.metadata.description or "",
            "files": _iter_component_files(meta),
        }
        summary_components.append(component_entry)
        build_targets.append(
            {
                "name": component_entry["name"],
                "version": component_entry["version"],
                "image": component_entry["image"],
                "folder": folder_path,
                "spec": spec,
            }
        )

    summary = {
        "componentCount": len(summary_components),
        "components": summary_components,
    }
    return summary, build_targets


def _run_docker_build(component_name: str, image: str, folder: Path) -> None:
    cmd = ["docker", "build", "-t", image, str(folder)]
    logger.info("integration: building %s (%s)", component_name, image)
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    lines: List[str] = []
    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.rstrip("\n")
        lines.append(line)
        logger.info("integration.build[%s]: %s", component_name, line)

    return_code = process.wait()
    if return_code != 0:
        tail = "\n".join(lines[-12:]).strip()
        if tail:
            raise RuntimeError(
                f"docker build failed for {component_name} ({image}) with exit code {return_code}.\n{tail}"
            )
        raise RuntimeError(
            f"docker build failed for {component_name} ({image}) with exit code {return_code}."
        )


def _register_component_spec(backend_url: str, spec: ComponentSpec) -> Dict[str, Any]:
    payload = {
        "spec": spec.model_dump(by_alias=True),
        "activate": True,
    }
    data = json.dumps(payload).encode("utf-8")
    request = url_request.Request(
        f"{backend_url}/components",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with url_request.urlopen(request, timeout=45) as response:
            body = response.read().decode("utf-8", errors="replace")
            if response.status < 200 or response.status >= 300:
                raise RuntimeError(
                    f"registration returned HTTP {response.status}: {body.strip()}"
                )
            if not body.strip():
                return {}
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                return {"raw": body.strip()}
    except url_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"registration HTTP {exc.code}: {detail or exc.reason}"
        ) from exc
    except url_error.URLError as exc:
        raise RuntimeError(f"registration request failed: {exc.reason}") from exc


def _write_integration_report(session_dir: str | None, report: str) -> None:
    if not session_dir:
        return
    path = Path(session_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "integration_report.txt").write_text(report + "\n", encoding="utf-8")


def node_integration(state: PipelineState) -> dict:
    run_integration = bool(state.get("run_integration"))
    session_dir_raw = state.get("session_dir")
    session_dir = Path(session_dir_raw).resolve() if session_dir_raw else Path.cwd()
    generated_index = state.get("generated_index") or {}

    if not generated_index:
        report = "Integration skipped: no generated components."
        logger.info("integration: %s", report)
        _write_integration_report(session_dir_raw, report)
        result = {"components": [], "builtImages": [], "registeredComponents": []}
        return {
            "integration_report": report,
            "integration_status": "completed",
            "integration_result": result,
        }

    summary, build_targets = _build_summary(generated_index, session_dir)
    _emit_event(state, "integration_summary_ready", {"summary": summary})

    if not run_integration:
        report = "Integration skipped by configuration (run_integration=false)."
        logger.info("integration: %s", report)
        _write_integration_report(session_dir_raw, report)
        result = {"components": summary.get("components", []), "builtImages": [], "registeredComponents": []}
        return {
            "integration_report": report,
            "integration_status": "skipped_by_config",
            "integration_result": result,
            "pending_integration": summary,
        }

    decision_getter = state.get("hitl_decision_getter")
    if not callable(decision_getter):
        raise RuntimeError("Integration API mode requires hitl_decision_getter callable")

    _emit_event(
        state,
        "waiting_decision",
        {"stage": "integration", "summary": summary},
    )

    try:
        decision = decision_getter(stage="integration", context={"summary": summary})
    except TypeError:
        # Compatibility fallback for old decision getter signatures.
        decision = decision_getter(summary)
    approved = bool(decision.get("approved"))
    feedback = (decision.get("feedback") or "").strip()

    if not approved:
        report = "Integration canceled by user confirmation."
        result = {
            "components": summary.get("components", []),
            "feedback": feedback,
            "builtImages": [],
            "registeredComponents": [],
        }
        _emit_event(
            state,
            "integration_skipped",
            {"stage": "integration", "reason": "rejected", "result": result},
        )
        logger.info("integration: %s", report)
        _write_integration_report(session_dir_raw, report)
        return {
            "integration_report": report,
            "integration_status": "skipped_by_user",
            "integration_result": result,
            "pending_integration": summary,
        }

    backend_url = os.getenv("BACKEND_URL") or os.getenv(
        "COMPONENT_GENERATION_BACKEND_URL",
        "http://localhost:8000",
    )
    backend_url = backend_url.rstrip("/")

    built_images: List[str] = []
    registered_components: List[Dict[str, Any]] = []

    try:
        _emit_event(
            state,
            "integration_build_started",
            {"stage": "integration", "componentCount": len(build_targets)},
        )
        for target in build_targets:
            _emit_event(
                state,
                "integration_build_component_started",
                {
                    "stage": "integration",
                    "component": target["name"],
                    "image": target["image"],
                },
            )
            _run_docker_build(
                component_name=str(target["name"]),
                image=str(target["image"]),
                folder=Path(target["folder"]),
            )
            built_images.append(str(target["image"]))
            _emit_event(
                state,
                "integration_build_component_succeeded",
                {
                    "stage": "integration",
                    "component": target["name"],
                    "image": target["image"],
                },
            )

        _emit_event(
            state,
            "integration_registration_started",
            {"stage": "integration", "componentCount": len(build_targets)},
        )

        for target in build_targets:
            response_payload = _register_component_spec(
                backend_url=backend_url,
                spec=target["spec"],
            )
            registered_row = {
                "name": target["name"],
                "version": target["version"],
                "response": response_payload,
            }
            registered_components.append(registered_row)
            _emit_event(
                state,
                "integration_registration_component_succeeded",
                {
                    "stage": "integration",
                    "component": target["name"],
                    "version": target["version"],
                    "result": {
                        "builtImages": built_images,
                        "registeredComponents": registered_components,
                    },
                },
            )
    except Exception as exc:
        partial_result = {
            "components": summary.get("components", []),
            "builtImages": built_images,
            "registeredComponents": registered_components,
        }
        _emit_event(
            state,
            "integration_failed",
            {
                "stage": "integration",
                "error": str(exc),
                "result": partial_result,
            },
        )
        raise

    report = (
        f"Integration completed: built {len(built_images)} image(s) and "
        f"registered {len(registered_components)} component(s)."
    )
    result = {
        "components": summary.get("components", []),
        "builtImages": built_images,
        "registeredComponents": registered_components,
    }
    logger.info("integration: %s", report)
    _write_integration_report(session_dir_raw, report)
    return {
        "integration_report": report,
        "integration_status": "completed",
        "integration_result": result,
        "pending_integration": summary,
    }
