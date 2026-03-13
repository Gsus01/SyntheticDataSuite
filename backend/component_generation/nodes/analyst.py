from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from component_generation.llm import LLMClient
from component_generation.context import load_prompt
from component_generation.llm_trace import write_llm_request, write_llm_response
from component_generation.schemas import ExtractionPlan
from component_generation.schemas import ComponentType
from component_generation.state import PipelineState
from component_generation.validation import validate_generated_component_plan

logger = logging.getLogger(__name__)


def _is_under(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix + "/")


def _validate_plan_paths(plan: Dict[str, Any]) -> None:
    errors: List[str] = []
    components = plan.get("components") or []
    for comp in components:
        name = comp.get("name", "component")
        try:
            validate_generated_component_plan(comp)
        except ValueError as exc:
            errors.append(str(exc))
        for port in comp.get("inputs") or []:
            path = port.get("path")
            role = port.get("role")
            if not isinstance(path, str):
                errors.append(f"{name}: input port missing path")
                continue
            if role == "config":
                if not _is_under(path, "/data/config"):
                    errors.append(
                        f"{name}: config input path must be under /data/config: {path}"
                    )
            else:
                if not _is_under(path, "/data/inputs"):
                    errors.append(
                        f"{name}: input path must be under /data/inputs: {path}"
                    )
        for port in comp.get("outputs") or []:
            path = port.get("path")
            if not isinstance(path, str):
                errors.append(f"{name}: output port missing path")
                continue
            if not _is_under(path, "/data/outputs"):
                errors.append(
                    f"{name}: output path must be under /data/outputs: {path}"
                )
    if errors:
        raise ValueError("Invalid port paths:\n" + "\n".join(errors))


def _sanitize_kebab(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\-]+", "-", value.strip().lower())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned or "component"


def _infer_type(source: str, paths: List[str]) -> ComponentType:
    haystack = (source or "").lower()
    for path in paths:
        haystack += f" {path.lower()}"

    if any(k in haystack for k in ["preprocess", "normalize", "clean", "feature"]):
        return "preprocessing"
    if any(k in haystack for k in ["train", "fit", "epoch", "loss"]):
        return "training"
    if any(k in haystack for k in ["generate", "synthesize", "sample"]):
        return "generation"
    return "other"


def _default_component_name(paths: List[str]) -> str:
    if not paths:
        return "component"
    name = Path(paths[0]).stem
    return _sanitize_kebab(name)


def _heuristic_plan(state: PipelineState) -> Dict[str, Any]:
    source = state.get("combined_source") or ""
    input_paths = state.get("input_paths") or []
    ctype = _infer_type(source, input_paths)
    name = _default_component_name(input_paths)
    secondary_type: ComponentType = (
        "training"
        if ctype == "preprocessing"
        else "generation"
        if ctype == "training"
        else "other"
        if ctype == "generation"
        else "preprocessing"
    )

    primary_component = {
        "name": name,
        "title": name.replace("-", " ").title(),
        "type": ctype,
        "description": "Auto-generated primary component (heuristic step 1).",
        "inputs": [
            {
                "name": "input-data",
                "path": "/data/inputs/input",
                "role": "data",
            }
        ],
        "outputs": [
            {
                "name": "prepared-data",
                "path": "/data/outputs/prepared",
                "role": "data",
            }
        ],
        "parameters_defaults": {},
        "notes": ["Generated without LLM. Adjust IO/params in HITL."],
    }

    secondary_name = _sanitize_kebab(f"{name}-aux")
    secondary_component = {
        "name": secondary_name,
        "title": secondary_name.replace("-", " ").title(),
        "type": secondary_type,
        "description": "Auto-generated complementary component (heuristic step 2).",
        "inputs": [
            {
                "name": "prepared-data",
                "path": "/data/inputs/prepared",
                "role": "data",
            },
            {
                "name": "settings",
                "path": "/data/config/settings.json",
                "role": "config",
            },
        ],
        "outputs": [
            {
                "name": "result-data",
                "path": "/data/outputs/result",
                "role": "data",
            }
        ],
        "parameters_defaults": {"emit_summary": True},
        "notes": ["Fallback component added to make the review plan more explicit."],
    }

    return {
        "components": [primary_component, secondary_component],
        "rationale": "Heuristic plan derived from input filenames and content.",
        "assumptions": ["LLM not configured; using minimal defaults with a two-step fallback plan."],
    }


def node_analyst(state: PipelineState) -> Dict[str, Any]:
    logger.info("analyst: building plan")
    llm: LLMClient | None = state.get("llm")
    combined = state.get("combined_source") or ""
    context = state.get("analyst_context") or ""
    feedback = (state.get("feedback") or "").strip()
    disable_llm = bool(state.get("disable_llm"))
    use_structured_output = bool(state.get("structured_output", True))

    if llm and not disable_llm:
        system = load_prompt("analyst_system.md")
        schema_hint = ""
        if not use_structured_output:
            schema_hint = (
                "\n\nOUTPUT JSON SCHEMA:\n"
                + json.dumps(ExtractionPlan.model_json_schema(), indent=2, ensure_ascii=False)
            )
        user_parts = [
            "ANALYST CONTEXT:\n",
            context,
            "\n\nSOURCE (may be truncated):\n",
            combined,
        ]
        if schema_hint:
            user_parts.append(schema_hint)
        if feedback:
            user_parts += ["\n\nUSER FEEDBACK (must incorporate):\n", feedback]
        user = "".join(user_parts)

        try:
            format_schema = (
                ExtractionPlan.model_json_schema() if use_structured_output else None
            )
            write_llm_request(
                state.get("session_dir"),
                "analyst",
                "plan",
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
            write_llm_response(state.get("session_dir"), "analyst", "plan", raw)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("analyst: LLM raw response:\n%s", raw.strip())
            plan_obj = ExtractionPlan.model_validate_json(raw)
            plan = plan_obj.model_dump()
            _validate_plan_paths(plan)
            logger.info("analyst: components=%s", len(plan.get("components", [])))
        except Exception as exc:
            logger.error("analyst: LLM failed: %s", exc)
            raise RuntimeError("Analyst LLM failed.") from exc
    else:
        if not disable_llm:
            raise RuntimeError(
                "LLM not configured; rerun with --no-llm to skip LLM."
            )
        plan = _heuristic_plan(state)
        logger.info("analyst: LLM disabled, using heuristic plan.")

    session_dir = state.get("session_dir")
    if session_dir:
        Path(session_dir).mkdir(parents=True, exist_ok=True)
        (Path(session_dir) / "plan.json").write_text(
            json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    return {"plan": plan, "approved": False}
