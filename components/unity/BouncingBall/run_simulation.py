#!/usr/bin/env python3
"""Wrapper to launch the Unity bouncing ball simulation from a JSON config."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Tuple


DEFAULT_CONFIG_PATH = Path("/tmp/config/bouncing_ball.json")
DEFAULT_OUTPUT_PATH = Path("/tmp/output/bouncing_ball.json")
DEFAULT_BINARY_PATH = Path("/app/BouncingBallSimulation.x86_64")
DEFAULT_START_HEIGHT = 15.0
DEFAULT_GRAVITY = -9.8


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def resolve_parameters(data: Dict[str, Any], cli_output: Path) -> Tuple[float, float, Path]:
    start_height = data.get("startHeight", DEFAULT_START_HEIGHT)
    gravity = data.get("gravity", DEFAULT_GRAVITY)

    try:
        start_height = float(start_height)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid startHeight value: {start_height}") from exc

    try:
        gravity = float(gravity)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid gravity value: {gravity}") from exc

    return start_height, gravity, cli_output


def run_simulation(binary_path: Path, start_height: float, gravity: float, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(binary_path),
        "-batchmode",
        "-nographics",
        "-startHeight",
        str(start_height),
        "-gravity",
        str(gravity),
        "-outputFile",
        str(output_path),
    ]
    completed = subprocess.run(cmd, check=False)
    return completed.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Unity bouncing ball simulation.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to JSON config (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output file path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=DEFAULT_BINARY_PATH,
        help=f"Path to Unity binary (default: {DEFAULT_BINARY_PATH})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    start_height, gravity, output_path = resolve_parameters(config, args.output)
    return run_simulation(args.binary, start_height, gravity, output_path)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[bouncing-ball] failed: {exc}", file=sys.stderr)
        sys.exit(1)
