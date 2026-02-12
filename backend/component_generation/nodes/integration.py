from __future__ import annotations

import json
import logging
import os
import selectors
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple
from urllib import error as url_error
from urllib import request as url_request

from component_generation.state import PipelineState
from component_spec import ComponentSpec

logger = logging.getLogger(__name__)


def _positive_int_from_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _emit_log_line(
    state: PipelineState,
    line: str,
    *,
    level: str = "INFO",
    source: str = "integration",
) -> None:
    _emit_event(
        state,
        "log_line",
        {"line": line, "level": level.upper(), "source": source},
    )


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


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=5)
        return
    except Exception:
        pass
    try:
        process.kill()
        process.wait(timeout=5)
    except Exception:
        pass


def _apply_shell_export_lines(base_env: Dict[str, str], shell_script: str) -> Dict[str, str]:
    env = dict(base_env)
    for raw_line in shell_script.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            chunk = line[len("export ") :].strip()
            if "=" not in chunk:
                continue
            key, raw_value = chunk.split("=", 1)
            key = key.strip()
            value = raw_value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            env[key] = value
        elif line.startswith("unset "):
            key = line[len("unset ") :].strip()
            if key:
                env.pop(key, None)
    return env


def _resolve_docker_environment(
    *,
    emit_line: Callable[[str, str, str], None] | None = None,
) -> Dict[str, str]:
    base_env = dict(os.environ)
    raw_mode = os.getenv("COMPONENT_GENERATION_USE_MINIKUBE_DOCKER_ENV", "auto")
    mode = raw_mode.strip().lower()
    if mode in {"0", "false", "no", "off", "disabled"}:
        if emit_line:
            emit_line(
                "minikube docker-env disabled by COMPONENT_GENERATION_USE_MINIKUBE_DOCKER_ENV",
                "INFO",
                "integration",
            )
        return base_env

    require_minikube_env = mode in {"1", "true", "yes", "on", "required"}
    profile = os.getenv("MINIKUBE_PROFILE", "minikube").strip() or "minikube"
    cmd = ["minikube", "-p", profile, "docker-env", "--shell", "bash"]

    if emit_line:
        emit_line(
            f"resolving docker environment via: {' '.join(cmd)}",
            "INFO",
            "integration",
        )

    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            check=False,
            env=base_env,
        )
    except FileNotFoundError as exc:
        if require_minikube_env:
            raise RuntimeError(
                "minikube is not available but minikube docker-env is required."
            ) from exc
        if emit_line:
            emit_line(
                "minikube not found; using current docker environment.",
                "WARNING",
                "integration",
            )
        return base_env
    except subprocess.TimeoutExpired as exc:
        if require_minikube_env:
            raise RuntimeError("minikube docker-env command timed out.") from exc
        if emit_line:
            emit_line(
                "minikube docker-env timed out; using current docker environment.",
                "WARNING",
                "integration",
            )
        return base_env

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        if require_minikube_env:
            raise RuntimeError(
                f"minikube docker-env failed with exit code {completed.returncode}: {detail}"
            )
        if emit_line:
            emit_line(
                (
                    "minikube docker-env failed; using current docker environment. "
                    f"exit={completed.returncode}"
                ),
                "WARNING",
                "integration",
            )
            if detail:
                emit_line(detail, "WARNING", "integration")
        return base_env

    resolved_env = _apply_shell_export_lines(base_env, completed.stdout or "")
    docker_host = resolved_env.get("DOCKER_HOST")
    if emit_line:
        if docker_host:
            emit_line(
                f"docker environment configured for minikube host: {docker_host}",
                "INFO",
                "integration",
            )
        else:
            emit_line(
                "minikube docker-env resolved without explicit DOCKER_HOST; using resolved env.",
                "INFO",
                "integration",
            )
    return resolved_env


def _run_docker_build(
    component_name: str,
    image: str,
    folder: Path,
    *,
    emit_line: Callable[[str, str, str], None] | None = None,
    timeout_seconds: int = 1_200,
    idle_timeout_seconds: int = 180,
    heartbeat_seconds: int = 15,
    process_env: Dict[str, str] | None = None,
) -> None:
    cmd = ["docker", "build", "--progress=plain", "-t", image, str(folder)]
    logger.info("integration: building %s (%s)", component_name, image)
    if emit_line:
        emit_line(f"$ {' '.join(cmd)}", "INFO", "integration.build")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=process_env,
    )
    lines: List[str] = []
    assert process.stdout is not None
    stdout_handle = process.stdout
    selector = selectors.DefaultSelector()
    selector.register(stdout_handle, selectors.EVENT_READ)
    start_time = time.monotonic()
    last_output_at = start_time
    last_heartbeat_at = start_time
    stdout_open = True

    def _record_output_line(raw_line: str) -> None:
        nonlocal last_output_at, lines
        line = raw_line.rstrip("\r\n")
        last_output_at = time.monotonic()
        lines.append(line)
        if len(lines) > 4000:
            lines = lines[-4000:]
        if emit_line:
            emit_line(line, "INFO", "integration.build")

        # Fast-fail on explicit Docker build errors.
        lowered = line.lower()
        if "error:" in lowered and "failed to solve" in lowered:
            _terminate_process(process)
            tail = "\n".join(lines[-18:]).strip()
            raise RuntimeError(
                f"docker build failed for {component_name} ({image}).\n{tail}"
            )

    def _drain_stdout() -> None:
        if not stdout_open:
            return
        while True:
            raw_line = stdout_handle.readline()
            if raw_line == "":
                break
            _record_output_line(raw_line)

    try:
        while True:
            now = time.monotonic()
            if process.poll() is not None:
                _drain_stdout()
                break

            if timeout_seconds > 0 and (now - start_time) > timeout_seconds:
                _terminate_process(process)
                tail = "\n".join(lines[-18:]).strip()
                message = (
                    f"docker build timed out for {component_name} ({image}) "
                    f"after {timeout_seconds}s"
                )
                if tail:
                    message += f". Last logs:\n{tail}"
                raise RuntimeError(message)

            if idle_timeout_seconds > 0 and (now - last_output_at) > idle_timeout_seconds:
                _terminate_process(process)
                tail = "\n".join(lines[-18:]).strip()
                message = (
                    f"docker build stalled for {component_name} ({image}) "
                    f"with no output for {idle_timeout_seconds}s"
                )
                if tail:
                    message += f". Last logs:\n{tail}"
                raise RuntimeError(message)

            if stdout_open and selector.get_map():
                events = selector.select(timeout=1.0)
            else:
                time.sleep(1.0)
                events = []
            now = time.monotonic()
            if not events:
                if (
                    emit_line
                    and heartbeat_seconds > 0
                    and (now - last_heartbeat_at) >= heartbeat_seconds
                ):
                    elapsed = int(now - start_time)
                    emit_line(
                        f"build still running for {component_name} ({elapsed}s elapsed)",
                        "INFO",
                        "integration.build",
                    )
                    last_heartbeat_at = now
                if process.poll() is not None:
                    break
                continue

            for selected, _ in events:
                raw_line = selected.fileobj.readline()
                if raw_line == "":
                    stdout_open = False
                    try:
                        selector.unregister(selected.fileobj)
                    except Exception:
                        pass
                    continue
                _record_output_line(raw_line)
    finally:
        try:
            selector.close()
        except Exception:
            pass
        try:
            stdout_handle.close()
        except Exception:
            pass

    return_code = process.wait()
    if return_code != 0:
        tail = "\n".join(lines[-18:]).strip()
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
        _emit_log_line(state, report, source="integration")
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
        _emit_log_line(state, report, source="integration")
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
        _emit_log_line(state, report, source="integration")
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
    build_timeout_seconds = _positive_int_from_env(
        "COMPONENT_GENERATION_BUILD_TIMEOUT_SECONDS",
        1_200,
    )
    build_idle_timeout_seconds = _positive_int_from_env(
        "COMPONENT_GENERATION_BUILD_IDLE_TIMEOUT_SECONDS",
        180,
    )
    build_heartbeat_seconds = _positive_int_from_env(
        "COMPONENT_GENERATION_BUILD_HEARTBEAT_SECONDS",
        15,
    )

    built_images: List[str] = []
    registered_components: List[Dict[str, Any]] = []

    try:
        docker_env = _resolve_docker_environment(
            emit_line=lambda line, level, source: _emit_log_line(
                state, line, level=level, source=source
            )
        )
        _emit_log_line(
            state,
            (
                f"integration: starting docker build for {len(build_targets)} component(s). "
                f"timeout={build_timeout_seconds}s idle-timeout={build_idle_timeout_seconds}s "
                f"heartbeat={build_heartbeat_seconds}s"
            ),
            source="integration",
        )
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
            _emit_log_line(
                state,
                f"building image for component '{target['name']}' -> {target['image']}",
                source="integration",
            )
            _run_docker_build(
                component_name=str(target["name"]),
                image=str(target["image"]),
                folder=Path(target["folder"]),
                emit_line=lambda line, level, source: _emit_log_line(
                    state, line, level=level, source=source
                ),
                timeout_seconds=build_timeout_seconds,
                idle_timeout_seconds=build_idle_timeout_seconds,
                heartbeat_seconds=build_heartbeat_seconds,
                process_env=docker_env,
            )
            built_images.append(str(target["image"]))
            _emit_log_line(
                state,
                f"image built successfully for '{target['name']}'",
                source="integration",
            )
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
        _emit_log_line(
            state,
            f"integration: registering {len(build_targets)} component(s) in platform",
            source="integration",
        )

        for target in build_targets:
            _emit_log_line(
                state,
                f"registering component '{target['name']}' ({target['version']})",
                source="integration",
            )
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
            _emit_log_line(
                state,
                f"component registered: {target['name']} ({target['version']})",
                source="integration",
            )
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
        _emit_log_line(
            state,
            f"integration failed: {exc}",
            level="ERROR",
            source="integration",
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
    _emit_log_line(state, report, source="integration")
    _write_integration_report(session_dir_raw, report)
    return {
        "integration_report": report,
        "integration_status": "completed",
        "integration_result": result,
        "pending_integration": summary,
    }
