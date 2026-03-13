from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

from component_generation.context import load_prompt
from component_generation.llm import LLMClient
from component_generation.llm_trace import write_llm_request, write_llm_response
from component_generation.schemas import GeneratedComponentFiles
from component_generation.state import PipelineState
from component_generation.validation import (
    validate_generated_component_plan,
    validate_generated_componentspec,
)
from component_spec import ComponentSpec
from component_generation.nodes.developer import _write_session_scripts

logger = logging.getLogger(__name__)


def _read_file(path: Path, max_chars: int = 60_000) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        text = path.read_text(errors="replace")
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n[TRUNCATED]\n"
    return text


def node_repair(state: PipelineState) -> Dict[str, object]:
    attempts = int(state.get("repair_attempts") or 0)
    if attempts >= 1:
        raise RuntimeError("Repair attempts exceeded (max 1).")

    issues = state.get("review_issues") or []
    if not issues:
        logger.info("repair: no issues to fix")
        return {"repair_attempts": attempts + 1}

    llm: LLMClient | None = state.get("llm")
    if not llm:
        raise RuntimeError("LLM not configured; cannot run repair.")

    plan = state.get("plan") or {}
    plan_components = {c.get("name"): c for c in plan.get("components") or []}
    generated = state.get("generated_index") or {}
    session_dir = state.get("session_dir")
    repair_context = state.get("repair_context") or ""
    use_structured_output = bool(state.get("structured_output", True))

    # group issues by component
    issues_by_component: Dict[str, List[str]] = {}
    for item in issues:
        comp = item.get("component") or "unknown"
        msg = item.get("message") or ""
        issues_by_component.setdefault(comp, []).append(msg)

    for comp_name, comp_issues in issues_by_component.items():
        meta = generated.get(comp_name)
        if not meta:
            raise RuntimeError(f"Repair: missing generated component {comp_name}")

        folder = Path(meta["folder"])
        current_files = {
            "main.py": _read_file(folder / "main.py"),
            "Dockerfile": _read_file(folder / "Dockerfile"),
            "ComponentSpec.json": _read_file(folder / "ComponentSpec.json"),
            "requirements.txt": _read_file(folder / "requirements.txt"),
            "README.md": _read_file(folder / "README.md"),
        }

        comp_plan = plan_components.get(comp_name) or {}
        validate_generated_component_plan(comp_plan)

        system = load_prompt("repair_system.md")
        schema_hint = ""
        if not use_structured_output:
            schema_hint = (
                "\n\nOUTPUT JSON SCHEMA:\n"
                + json.dumps(
                    GeneratedComponentFiles.model_json_schema(),
                    indent=2,
                    ensure_ascii=False,
                )
            )
        user = (
            "REPAIR CONTEXT:\n"
            + repair_context
            + "\n\nCOMPONENT PLAN:\n"
            + json.dumps(comp_plan, indent=2, ensure_ascii=False)
            + "\n\nREVIEW ISSUES:\n- "
            + "\n- ".join(comp_issues)
            + "\n\nCURRENT FILES:\n"
            + json.dumps(current_files, indent=2, ensure_ascii=False)
        )
        if schema_hint:
            user += schema_hint

        format_schema = (
            GeneratedComponentFiles.model_json_schema() if use_structured_output else None
        )
        write_llm_request(
            state.get("session_dir"),
            "repair",
            comp_name,
            {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "format_schema": format_schema,
                "structured_output": use_structured_output,
                "provider": state.get("llm_provider"),
                "model": state.get("llm_model"),
            },
        )
        raw = llm.invoke(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            format_schema=format_schema,
        )
        write_llm_response(state.get("session_dir"), "repair", comp_name, raw)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("repair: LLM raw response for %s:\n%s", comp_name, raw.strip())

        files = GeneratedComponentFiles.model_validate_json(raw)
        # validate ComponentSpec strictly
        spec = ComponentSpec.model_validate(files.componentspec)
        validate_generated_componentspec(spec, component_name=comp_name)

        (folder / "main.py").write_text(files.main_py, encoding="utf-8")
        (folder / "Dockerfile").write_text(files.dockerfile, encoding="utf-8")
        (folder / "ComponentSpec.json").write_text(
            json.dumps(files.componentspec, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (folder / "requirements.txt").write_text(
            (files.requirements_txt or "").strip() + "\n", encoding="utf-8"
        )
        if (files.readme_md or "").strip():
            (folder / "README.md").write_text(files.readme_md, encoding="utf-8")

        logger.info("repair: updated component %s", comp_name)

    # refresh scripts and index
    if session_dir:
        session_path = Path(session_dir)
        (session_path / "generated_index.json").write_text(
            json.dumps(generated, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        _write_session_scripts(session_path, generated)

    return {"repair_attempts": attempts + 1}
