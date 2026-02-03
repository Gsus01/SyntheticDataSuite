from __future__ import annotations

import logging
from pathlib import Path

from component_generation.state import PipelineState

logger = logging.getLogger(__name__)


def node_integration(state: PipelineState) -> dict:
    run_integration = bool(state.get("run_integration"))
    message = (
        "Integration step skipped (dry-run)."
        if not run_integration
        else "Integration step not implemented yet."
    )

    logger.info("integration: %s", message)

    session_dir = state.get("session_dir")
    if session_dir:
        Path(session_dir).mkdir(parents=True, exist_ok=True)
        (Path(session_dir) / "integration_report.txt").write_text(
            message + "\n", encoding="utf-8"
        )

    return {"integration_report": message}
