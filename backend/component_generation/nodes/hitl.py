from __future__ import annotations

import logging
import sys
from typing import Any, Dict, List

from component_generation.state import PipelineState

logger = logging.getLogger(__name__)


def _pretty_plan(plan: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("\n=== Proposed plan ===")
    rationale = (plan.get("rationale") or "").strip() or "(no rationale)"
    lines.append(rationale)
    lines.append("")
    for idx, comp in enumerate(plan.get("components", []), start=1):
        lines.append(f"[{idx}] {comp.get('name','')} ({comp.get('type','')})")
        desc = comp.get("description", "")
        if desc:
            lines.append(f"    {desc}")
        inputs = comp.get("inputs") or []
        if inputs:
            lines.append("    Inputs:")
            for port in inputs:
                lines.append(
                    f"      - {port.get('name','')} ({port.get('role','data')}): {port.get('path','')}"
                )
        outputs = comp.get("outputs") or []
        if outputs:
            lines.append("    Outputs:")
            for port in outputs:
                lines.append(
                    f"      - {port.get('name','')} ({port.get('role','data')}): {port.get('path','')}"
                )
        params = comp.get("parameters_defaults") or {}
        if params:
            lines.append("    Params defaults:")
            for key, value in params.items():
                lines.append(f"      - {key}: {value!r}")
        notes = comp.get("notes") or []
        if notes:
            lines.append("    Notes:")
            for note in notes:
                lines.append(f"      - {note}")
        lines.append("")
    return "\n".join(lines)


def _emit_event(state: PipelineState, event_type: str, payload: Dict[str, Any] | None = None) -> None:
    callback = state.get("event_callback")
    if not callable(callback):
        return
    try:
        callback(event_type, payload or {})
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("hitl: failed to emit event %s: %s", event_type, exc)


def node_hitl(state: PipelineState) -> Dict[str, Any]:
    plan = state.get("plan") or {}
    hitl_mode = str(state.get("hitl_mode") or "cli").strip().lower()

    if hitl_mode == "api":
        _emit_event(state, "plan_proposed", {"plan": plan})
        if state.get("auto_approve"):
            logger.info("hitl: auto-approving plan (api mode)")
            _emit_event(state, "resumed", {"approved": True})
            return {"approved": True, "feedback": ""}

        decision_getter = state.get("hitl_decision_getter")
        if not callable(decision_getter):
            raise RuntimeError("HITL API mode requires a callable hitl_decision_getter")

        _emit_event(state, "waiting_decision", {"plan": plan})
        decision = decision_getter(plan)
        approved = bool(decision.get("approved"))
        feedback = (decision.get("feedback") or "").strip()
        if not approved and not feedback:
            feedback = "Revise the plan: improve component boundaries and IO/params."
        _emit_event(
            state,
            "resumed",
            {"approved": approved, "hasFeedback": bool(feedback)},
        )
        if approved:
            return {"approved": True, "feedback": ""}
        return {"approved": False, "feedback": feedback}

    print(_pretty_plan(plan))

    if state.get("auto_approve") or not sys.stdin.isatty():
        logger.info("hitl: auto-approving plan")
        print("Auto-approving plan (non-interactive or --auto-approve).")
        return {"approved": True, "feedback": ""}

    logger.info("hitl: waiting for approval")
    while True:
        ans = input("Approve this plan? [y/n]: ").strip().lower()
        if ans in {"y", "n"}:
            break

    if ans == "y":
        return {"approved": True, "feedback": ""}

    feedback = input(
        "OK. Briefly describe what to change (split/merge components, rename ports, add params):\n> "
    ).strip()
    if not feedback:
        feedback = "Revise the plan: improve component boundaries and IO/params."
    return {"approved": False, "feedback": feedback}


def route_after_hitl(state: PipelineState) -> str:
    return "continue" if state.get("approved") else "revise"
