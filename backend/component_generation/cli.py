from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from component_generation.context import (
    load_analyst_context,
    load_developer_context,
    load_repair_context,
)
from component_generation.llm import make_llm
from component_generation.logging_utils import setup_logging
from component_generation.state import PipelineState
from component_generation.workflow import build_graph


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Component generation pipeline (offline, LangGraph)."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input files (.ipynb, .py, .md, etc.)",
    )
    parser.add_argument(
        "--out",
        default="components/generated-sessions",
        help="Base output directory for per-session artifacts.",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Optional session id (defaults to UUID).",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Skip HITL prompt and auto-approve plan.",
    )
    parser.add_argument(
        "--include-markdown",
        action="store_true",
        help="Include markdown cells from notebooks (as comments).",
    )
    parser.add_argument(
        "--max-chars-per-file",
        type=int,
        default=200_000,
        help="Maximum characters per file before truncation.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    parser.add_argument(
        "--log-json",
        action="store_true",
        help="Emit logs as JSON lines.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional log file path.",
    )
    parser.add_argument(
        "--run-integration",
        action="store_true",
        help="Run integration step (build/push/register).",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM and run heuristic plan + stub outputs.",
    )
    parser.add_argument(
        "--no-structured-output",
        action="store_true",
        help=(
            "Disable structured output (json_schema). "
            "Uses prompt-only JSON with strict validation."
        ),
    )
    parser.add_argument(
        "--provider",
        default="ollama",
        help="LLM provider (default: ollama).",
    )
    parser.add_argument(
        "--model",
        default="qwen3:14b",
        help="LLM model identifier (default: qwen3:14b).",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama base URL.",
    )
    parser.add_argument(
        "--openrouter-url",
        default="https://openrouter.ai/api/v1",
        help="OpenRouter base URL.",
    )
    parser.add_argument(
        "--openrouter-key",
        default=None,
        help="OpenRouter API key (or set OPENROUTER_API_KEY).",
    )
    parser.add_argument(
        "--openrouter-provider",
        default=None,
        help=(
            "OpenRouter provider routing (comma-separated order or JSON). "
            "Examples: 'openai' or 'openai,anthropic' or '{\"order\":[\"openai\"]}'."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature.",
    )
    parser.add_argument(
        "--no-trace",
        dest="trace",
        action="store_false",
        help="Disable verbose trace logs.",
    )
    parser.set_defaults(trace=True)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Load .env files if available (local dev convenience)
    try:
        from dotenv import load_dotenv  # type: ignore[import-not-found]
    except Exception:
        load_dotenv = None

    if load_dotenv is not None:
        # Load from CWD first, then repo root (.env / .env.local) if present.
        load_dotenv()
        for name in (".env", ".env.local"):
            dotenv_path = REPO_ROOT / name
            if dotenv_path.exists():
                load_dotenv(dotenv_path, override=False)

    setup_logging(args.log_level, json_logs=args.log_json, log_file=args.log_file)

    session_id = args.session_id or str(uuid4())
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir = out_dir.resolve()
    session_dir = out_dir / session_id

    analyst_context = load_analyst_context()
    developer_context = load_developer_context()
    repair_context = load_repair_context()
    llm = None
    if not args.no_llm:
        if args.provider.lower() == "openrouter":
            api_key = args.openrouter_key or os.getenv("OPENROUTER_API_KEY")
            provider_cfg = None
            if args.openrouter_provider:
                raw = args.openrouter_provider.strip()
                if raw.startswith("{"):
                    import json

                    provider_cfg = json.loads(raw)
                else:
                    order = [p.strip() for p in raw.split(",") if p.strip()]
                    if order:
                        provider_cfg = {"order": order}
            llm = make_llm(
                provider=args.provider,
                model=args.model,
                base_url=args.openrouter_url,
                temperature=args.temperature,
                api_key=api_key,
                openrouter_provider=provider_cfg,
            )
        else:
            llm = make_llm(
                provider=args.provider,
                model=args.model,
                base_url=args.ollama_url,
                temperature=args.temperature,
            )

    init_state: PipelineState = {
        "input_paths": args.inputs,
        "include_markdown": args.include_markdown,
        "max_chars_per_file": args.max_chars_per_file,
        "out_dir": str(out_dir),
        "session_dir": str(session_dir),
        "analyst_context": analyst_context,
        "developer_context": developer_context,
        "repair_context": repair_context,
        "auto_approve": args.auto_approve,
        "trace": args.trace,
        "run_integration": args.run_integration,
        "feedback": "",
        "repair_attempts": 0,
        "llm": llm,
        "llm_provider": args.provider,
        "llm_model": args.model,
        "llm_temperature": args.temperature,
        "structured_output": not args.no_structured_output,
        "disable_llm": args.no_llm,
    }

    graph = build_graph()
    final_state = graph.invoke(init_state)

    print("\n=== DONE ===")
    print(f"Output dir: {out_dir}")
    print(f"Session dir: {session_dir}")
    print("\nReviewer report:\n")
    print(final_state.get("review_report", "(no report)"))


if __name__ == "__main__":
    main()
