from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from component_generation.nodes.analyst import _validate_plan_paths
from component_generation.nodes.developer import node_developer
from component_generation.nodes.tester import node_tester
from component_generation.validation import validate_generated_componentspec
from component_spec import ComponentSpec


def _output_plan_component() -> dict[str, object]:
    return {
        "name": "model-evaluator",
        "title": "Model Evaluator",
        "type": "output",
        "description": "Evaluates a trained model and writes a metrics report.",
        "inputs": [
            {"name": "model", "path": "/data/inputs/model.pkl", "role": "model"},
        ],
        "outputs": [
            {"name": "metrics", "path": "/data/outputs/metrics.json", "role": "metrics"},
        ],
        "parameters_defaults": {},
        "notes": [],
    }


def _write_generated_component(tmp_path: Path, *, component_type: str) -> dict[str, dict[str, str]]:
    folder = tmp_path / "components" / component_type / "demo-component"
    folder.mkdir(parents=True, exist_ok=True)

    spec_payload = {
        "apiVersion": "sds/v1",
        "kind": "Component",
        "metadata": {
            "name": "demo-component",
            "version": "v1.0.0",
            "type": component_type,
            "title": "Demo Component",
            "description": "Generated test component",
        },
        "io": {
            "inputs": [
                {"name": "input-data", "path": "/data/inputs/data.csv", "role": "data"}
            ],
            "outputs": [
                {"name": "result-data", "path": "/data/outputs/result.csv", "role": "data"}
            ],
        },
        "runtime": {"image": "sds/demo-component:dev"},
    }
    (folder / "ComponentSpec.json").write_text(
        json.dumps(spec_payload, indent=2),
        encoding="utf-8",
    )
    (folder / "main.py").write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--input-dir')\n"
        "parser.add_argument('--output-dir')\n",
        encoding="utf-8",
    )
    (folder / "Dockerfile").write_text("FROM python:3.12-slim\n", encoding="utf-8")

    return {
        "demo-component": {
            "folder": str(folder),
            "ComponentSpec.json": str(folder / "ComponentSpec.json"),
            "main.py": str(folder / "main.py"),
            "Dockerfile": str(folder / "Dockerfile"),
        }
    }


def test_analyst_rejects_output_component_type() -> None:
    plan = {"components": [_output_plan_component()]}

    with pytest.raises(ValueError, match="built-in output node"):
        _validate_plan_paths(plan)


def test_developer_rejects_output_component_plan(tmp_path: Path) -> None:
    state = {
        "plan": {"components": [_output_plan_component()]},
        "out_dir": str(tmp_path / "out"),
        "session_dir": str(tmp_path / "session"),
        "disable_llm": True,
    }

    with pytest.raises(ValueError, match="built-in output node"):
        node_developer(state)


def test_generated_componentspec_validation_rejects_output_type() -> None:
    spec = ComponentSpec.model_validate(
        {
            "apiVersion": "sds/v1",
            "kind": "Component",
            "metadata": {
                "name": "demo-component",
                "version": "v1.0.0",
                "type": "output",
            },
            "io": {"inputs": [], "outputs": []},
            "runtime": {"image": "sds/demo-component:dev"},
        }
    )

    with pytest.raises(ValueError, match="built-in output node"):
        validate_generated_componentspec(spec)


def test_tester_rejects_generated_output_component(tmp_path: Path) -> None:
    generated_index = _write_generated_component(tmp_path, component_type="output")

    result = node_tester(
        {
            "generated_index": generated_index,
            "session_dir": str(tmp_path),
        }
    )

    assert result["review_status"] == "NEEDS_FIX"
    assert any(
        "built-in output node" in issue["message"]
        for issue in result["review_issues"]
    )
