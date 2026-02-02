from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class PipelineState(TypedDict, total=False):
    input_paths: List[str]
    combined_source: str
    include_markdown: bool
    max_chars_per_file: int
    analyst_context: str
    developer_context: str

    # runtime + controls
    auto_approve: bool
    trace: bool
    run_integration: bool

    # planning + HITL
    plan: Dict[str, Any]
    approved: bool
    feedback: str

    # outputs
    out_dir: str
    session_dir: str
    generated_index: Dict[str, Dict[str, str]]
    review_report: str
    review_status: str
    review_issues: List[Dict[str, str]]
    repair_attempts: int
    integration_report: str

    # optional LLM placeholder
    llm: Optional[Any]
    llm_provider: Optional[str]
    llm_model: Optional[str]
    llm_temperature: Optional[float]
    structured_output: bool
    disable_llm: bool
    repair_context: str
