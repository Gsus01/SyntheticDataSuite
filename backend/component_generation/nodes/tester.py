from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

from component_generation.state import PipelineState
from component_spec import ComponentSpec

logger = logging.getLogger(__name__)


def _is_under(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix + "/")


def _validate_ports(spec: ComponentSpec) -> List[str]:
    issues: List[str] = []
    for port in spec.io.inputs:
        if port.path is None:
            continue
        if port.role == "config":
            if not _is_under(port.path, "/data/config"):
                issues.append(
                    f"Input {port.name} role=config not under /data/config: {port.path}"
                )
        else:
            if not _is_under(port.path, "/data/inputs"):
                issues.append(
                    f"Input {port.name} not under /data/inputs: {port.path}"
                )
    for port in spec.io.outputs:
        if port.path is None:
            continue
        if not _is_under(port.path, "/data/outputs"):
            issues.append(
                f"Output {port.name} not under /data/outputs: {port.path}"
            )
    return issues


def _check_dockerfile(path: Path) -> List[str]:
    if not path.exists():
        return ["Dockerfile missing"]
    text = path.read_text(encoding="utf-8", errors="replace")
    if "ENTRYPOINT" in text and "/data" in text:
        return ["Dockerfile uses ENTRYPOINT with /data paths"]
    return []


def _check_main_py(path: Path) -> List[str]:
    if not path.exists():
        return ["main.py missing"]
    text = path.read_text(encoding="utf-8", errors="replace")
    issues = []
    if "--input-dir" not in text or "--output-dir" not in text:
        issues.append("main.py missing --input-dir/--output-dir args")
    return issues


def node_tester(state: PipelineState) -> Dict[str, object]:
    generated = state.get("generated_index") or {}
    logger.info("tester: reviewing %s component(s)", len(generated))

    issues: List[str] = []
    structured: List[Dict[str, str]] = []
    for name, meta in generated.items():
        spec_path = Path(meta.get("ComponentSpec.json", ""))
        main_path = Path(meta.get("main.py", ""))
        docker_path = Path(meta.get("Dockerfile", ""))

        if not spec_path.exists():
            msg = "ComponentSpec.json missing"
            issues.append(f"[{name}] {msg}")
            structured.append({"component": name, "message": msg})
            continue

        try:
            payload = json.loads(spec_path.read_text(encoding="utf-8"))
            spec = ComponentSpec.model_validate(payload)
        except Exception as exc:
            msg = f"invalid ComponentSpec: {exc}"
            issues.append(f"[{name}] {msg}")
            structured.append({"component": name, "message": msg})
            continue

        image = spec.runtime.image if spec.runtime else None
        if image and image != image.lower():
            msg = f"runtime.image not lowercase: {image}"
            issues.append(f"[{name}] {msg}")
            structured.append({"component": name, "message": msg})

        for msg in _validate_ports(spec):
            issues.append(f"[{name}] {msg}")
            structured.append({"component": name, "message": msg})
        for msg in _check_main_py(main_path):
            issues.append(f"[{name}] {msg}")
            structured.append({"component": name, "message": msg})
        for msg in _check_dockerfile(docker_path):
            issues.append(f"[{name}] {msg}")
            structured.append({"component": name, "message": msg})

    status = "OK" if not issues else "NEEDS_FIX"
    report_lines = [status]
    if issues:
        report_lines.append("")
        report_lines.append("Issues:")
        report_lines.extend([f"- {item}" for item in issues])

    report = "\n".join(report_lines)

    session_dir = state.get("session_dir")
    if session_dir:
        Path(session_dir).mkdir(parents=True, exist_ok=True)
        (Path(session_dir) / "review_report.txt").write_text(
            report, encoding="utf-8"
        )

    logger.info("tester: status=%s issues=%s", status, len(issues))
    return {
        "review_report": report,
        "review_status": status,
        "review_issues": structured,
    }
