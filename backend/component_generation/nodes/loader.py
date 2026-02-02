from __future__ import annotations

import json
import logging
from pathlib import Path

from component_generation.ingest import ingest_paths
from component_generation.state import PipelineState

logger = logging.getLogger(__name__)


def node_load(state: PipelineState) -> dict:
    input_paths = state.get("input_paths") or []
    include_markdown = bool(state.get("include_markdown"))
    max_chars = int(state.get("max_chars_per_file") or 200_000)

    logger.info("load: ingesting %s input files", len(input_paths))
    combined = ingest_paths(
        input_paths,
        max_chars_per_file=max_chars,
        include_markdown=include_markdown,
    )
    logger.info("load: combined source chars=%s", len(combined))

    session_dir = state.get("session_dir")
    if session_dir:
        session_path = Path(session_dir)
        session_path.mkdir(parents=True, exist_ok=True)
        (session_path / "combined_source.txt").write_text(
            combined, encoding="utf-8"
        )
        (session_path / "ingest_meta.json").write_text(
            json.dumps(
                {
                    "inputs": input_paths,
                    "include_markdown": include_markdown,
                    "max_chars_per_file": max_chars,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    return {"combined_source": combined}
