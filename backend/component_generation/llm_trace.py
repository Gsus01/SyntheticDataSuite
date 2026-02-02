from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def _trace_dir(session_dir: Optional[str], node: str, label: Optional[str]) -> Optional[Path]:
    if not session_dir:
        return None
    safe_node = node.strip() or "node"
    safe_label = (label or "default").strip() or "default"
    path = Path(session_dir) / "llm" / safe_node / safe_label
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_llm_request(
    session_dir: Optional[str],
    node: str,
    label: Optional[str],
    payload: Dict[str, Any],
) -> None:
    path = _trace_dir(session_dir, node, label)
    if not path:
        return
    (path / "request.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_llm_response(
    session_dir: Optional[str],
    node: str,
    label: Optional[str],
    response_text: str,
) -> None:
    path = _trace_dir(session_dir, node, label)
    if not path:
        return
    (path / "response.txt").write_text(response_text, encoding="utf-8")
