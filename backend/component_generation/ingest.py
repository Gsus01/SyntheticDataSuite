from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

DEFAULT_MAX_CHARS = 200_000
_NOTEBOOK_TRUNCATION_MARK = "\n\n[TRUNCATED]\n"


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) > max_chars:
        return text[:max_chars] + _NOTEBOOK_TRUNCATION_MARK
    return text


def read_text_file(path: Path, *, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        text = path.read_text(errors="replace")
    return _truncate(text, max_chars)


def _detect_notebook_language(payload: dict) -> Optional[str]:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return None

    lang = None
    language_info = metadata.get("language_info")
    if isinstance(language_info, dict):
        lang = language_info.get("name")

    if not lang:
        kernelspec = metadata.get("kernelspec")
        if isinstance(kernelspec, dict):
            lang = kernelspec.get("language")

    if isinstance(lang, str):
        return lang.strip().lower()
    return None


def _cell_source_to_text(source: object) -> str:
    if source is None:
        return ""
    if isinstance(source, list):
        return "".join(str(item) for item in source)
    return str(source)


def notebook_to_text(
    path: Path,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    include_markdown: bool = False,
) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    language = _detect_notebook_language(payload)
    if language and language != "python":
        raise ValueError(
            f"Notebook language not supported: {language!r} (only python)"
        )

    cells = payload.get("cells", [])
    if not isinstance(cells, list):
        cells = []

    parts = [f"# Notebook: {path.name}"]
    for idx, cell in enumerate(cells):
        if not isinstance(cell, dict):
            continue
        cell_type = cell.get("cell_type")
        if cell_type == "code":
            source_text = _cell_source_to_text(cell.get("source"))
            if not source_text.strip():
                continue
            parts.append(f"\n# --- code cell {idx:03d} ---\n")
            parts.append(source_text)
        elif include_markdown and cell_type == "markdown":
            source_text = _cell_source_to_text(cell.get("source"))
            if not source_text.strip():
                continue
            parts.append(f"\n# --- markdown cell {idx:03d} ---\n")
            for line in source_text.splitlines():
                parts.append(f"# {line}")
            parts.append("")

    combined = "\n".join(parts)
    return _truncate(combined, max_chars)


def ingest_paths(
    paths: Iterable[str | Path],
    *,
    max_chars_per_file: int = DEFAULT_MAX_CHARS,
    include_markdown: bool = False,
) -> str:
    chunks = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Input not found: {path}")

        if path.suffix.lower() == ".ipynb":
            text = notebook_to_text(
                path, max_chars=max_chars_per_file, include_markdown=include_markdown
            )
            chunks.append(text)
        else:
            text = read_text_file(path, max_chars=max_chars_per_file)
            chunks.append(f"# File: {path.name}\n\n{text}")

    if not chunks:
        return ""
    separator = "\n\n" + ("-" * 80) + "\n\n"
    return "\n\n" + separator.join(chunks)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preview ingest output for code files and Python notebooks."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input files (.ipynb, .py, .md, etc.)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help="Maximum characters per file before truncation.",
    )
    parser.add_argument(
        "--include-markdown",
        action="store_true",
        help="Include markdown cells from notebooks (as comments).",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    combined = ingest_paths(
        args.inputs,
        max_chars_per_file=args.max_chars,
        include_markdown=args.include_markdown,
    )
    print(combined)


if __name__ == "__main__":
    main()
