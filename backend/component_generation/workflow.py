from __future__ import annotations

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


def build_graph() -> StateGraph:
    builder = StateGraph(PipelineState)
    builder.add_node("load", node_load)
    builder.add_node("analyst", node_analyst)
    builder.add_node("hitl", node_hitl)
    builder.add_node("developer", node_developer)
    builder.add_node("tester", node_tester)
    builder.add_node("repair", node_repair)
    builder.add_node("integration", node_integration)

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
