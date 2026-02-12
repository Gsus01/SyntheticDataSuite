from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from langgraph.graph import END, START, StateGraph

from component_generation.nodes import (
    node_analyst,
    node_developer,
    node_hitl,
    node_integration,
    node_load,
    node_repair,
    node_tester,
    route_after_hitl,
)
from component_generation.state import PipelineState


EventEmitter = Callable[[str, Dict[str, Any]], None]


def _wrap_node(
    name: str,
    node_fn: Callable[[PipelineState], Dict[str, Any]],
    emit_event: Optional[EventEmitter],
) -> Callable[[PipelineState], Dict[str, Any]]:
    if emit_event is None:
        return node_fn

    def wrapped(state: PipelineState) -> Dict[str, Any]:
        emit_event("node_started", {"node": name})
        try:
            result = node_fn(state)
        except Exception as exc:
            emit_event("node_failed", {"node": name, "error": str(exc)})
            raise
        emit_event("node_completed", {"node": name})
        return result

    return wrapped


def build_graph(*, emit_event: Optional[EventEmitter] = None) -> StateGraph:
    builder = StateGraph(PipelineState)
    builder.add_node("load", _wrap_node("load", node_load, emit_event))
    builder.add_node("analyst", _wrap_node("analyst", node_analyst, emit_event))
    builder.add_node("hitl", _wrap_node("hitl", node_hitl, emit_event))
    builder.add_node("developer", _wrap_node("developer", node_developer, emit_event))
    builder.add_node("tester", _wrap_node("tester", node_tester, emit_event))
    builder.add_node("repair", _wrap_node("repair", node_repair, emit_event))
    builder.add_node("integration", _wrap_node("integration", node_integration, emit_event))

    builder.add_edge(START, "load")
    builder.add_edge("load", "analyst")
    builder.add_edge("analyst", "hitl")
    builder.add_conditional_edges(
        "hitl",
        route_after_hitl,
        {"revise": "analyst", "continue": "developer"},
    )
    builder.add_edge("developer", "tester")
    builder.add_conditional_edges(
        "tester",
        route_after_tester,
        {"continue": "integration", "repair": "repair", "stop": END},
    )
    builder.add_edge("repair", "tester")
    builder.add_edge("integration", END)

    return builder.compile()


def route_after_tester(state: PipelineState) -> str:
    status = state.get("review_status")
    if status == "OK":
        return "continue"
    attempts = int(state.get("repair_attempts") or 0)
    if attempts >= 1:
        return "stop"
    return "repair"
