from __future__ import annotations

from pathlib import Path
from typing import List


def _read_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return path.read_text(errors="replace")


def load_prompt(prompt_name: str) -> str:
    root = Path(__file__).resolve().parents[2]
    prompt_path = root / "backend" / "component_generation" / "prompts" / prompt_name
    if not prompt_path.exists():
        raise FileNotFoundError(f"Missing prompt file: {prompt_path}")
    return _read_file(prompt_path)


def load_analyst_context() -> str:
    root = Path(__file__).resolve().parents[2]
    analyst_path = root / "backend" / "component_generation" / "prompts" / "analyst_context.md"
    if not analyst_path.exists():
        return ""
    text = _read_file(analyst_path)
    return f"# Analyst Context: {analyst_path.relative_to(root)}\n\n{text}"


def load_developer_context() -> str:
    root = Path(__file__).resolve().parents[2]
    developer_path = root / "backend" / "component_generation" / "prompts" / "developer_context.md"
    if developer_path.exists():
        text = _read_file(developer_path)
        return f"# Developer Context: {developer_path.relative_to(root)}\n\n{text}"

    targets = [
        root / "docs" / "new-component.md",
        root / "backend" / "component_spec.py",
        root / "backend" / "catalog_adapter.py",
    ]

    parts: List[str] = []
    for path in targets:
        if not path.exists():
            continue
        text = _read_file(path)
        parts.append(f"# Context: {path.relative_to(root)}\n\n{text}")

    return "\n\n" + ("\n\n" + ("-" * 80) + "\n\n").join(parts) if parts else ""


def load_repair_context() -> str:
    root = Path(__file__).resolve().parents[2]
    repair_path = root / "backend" / "component_generation" / "prompts" / "repair_context.md"
    if not repair_path.exists():
        return ""
    text = _read_file(repair_path)
    return f"# Repair Context: {repair_path.relative_to(root)}\n\n{text}"
