from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from component_generation.nodes.integration import node_integration


def _write_component(session_dir: Path, *, name: str = "demo-comp") -> dict[str, dict[str, str]]:
    folder = session_dir / "components" / "preprocessing" / name
    folder.mkdir(parents=True, exist_ok=True)

    spec_path = folder / "ComponentSpec.json"
    spec_payload = {
        "apiVersion": "sds/v1",
        "kind": "Component",
        "metadata": {
            "name": name,
            "version": "v1.0.0",
            "type": "preprocessing",
            "title": "Demo Component",
            "description": "Demo integration test component",
        },
        "io": {"inputs": [], "outputs": []},
        "runtime": {"image": f"sds/{name}:dev"},
    }
    spec_path.write_text(json.dumps(spec_payload, indent=2), encoding="utf-8")
    main_path = folder / "main.py"
    main_path.write_text("print('ok')\n", encoding="utf-8")

    return {
        name: {
            "folder": str(folder),
            "ComponentSpec.json": str(spec_path),
            "main.py": str(main_path),
        }
    }


def test_integration_rejected_skips_build_and_register(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    generated_index = _write_component(tmp_path)
    events: list[tuple[str, dict[str, Any]]] = []

    def _fail_build(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("build should not run when integration is rejected")

    def _fail_register(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("register should not run when integration is rejected")

    monkeypatch.setattr(
        "component_generation.nodes.integration._run_docker_build",
        _fail_build,
    )
    monkeypatch.setattr(
        "component_generation.nodes.integration._register_component_spec",
        _fail_register,
    )

    state = {
        "run_integration": True,
        "session_dir": str(tmp_path),
        "generated_index": generated_index,
        "event_callback": lambda event_type, payload: events.append((event_type, payload)),
        "hitl_decision_getter": lambda **_: {"approved": False, "feedback": "not now"},
    }

    result = node_integration(state)

    assert result["integration_status"] == "skipped_by_user"
    assert "canceled" in result["integration_report"].lower()
    event_types = [event_type for event_type, _ in events]
    assert "waiting_decision" in event_types
    assert "integration_skipped" in event_types


def test_integration_success_builds_and_registers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    generated_index = _write_component(tmp_path)
    events: list[tuple[str, dict[str, Any]]] = []
    build_calls: list[tuple[str, str]] = []
    register_calls: list[tuple[str, str]] = []

    def _fake_build(component_name: str, image: str, folder: Path) -> None:
        build_calls.append((component_name, image))
        assert folder.exists()

    def _fake_register(backend_url: str, spec: Any) -> dict[str, Any]:
        register_calls.append((backend_url, spec.metadata.name))
        return {"name": spec.metadata.name, "activeVersion": spec.metadata.version}

    monkeypatch.setattr(
        "component_generation.nodes.integration._run_docker_build",
        _fake_build,
    )
    monkeypatch.setattr(
        "component_generation.nodes.integration._register_component_spec",
        _fake_register,
    )

    state = {
        "run_integration": True,
        "session_dir": str(tmp_path),
        "generated_index": generated_index,
        "event_callback": lambda event_type, payload: events.append((event_type, payload)),
        "hitl_decision_getter": lambda **_: {"approved": True, "feedback": ""},
    }

    result = node_integration(state)

    assert result["integration_status"] == "completed"
    assert len(build_calls) == 1
    assert len(register_calls) == 1
    assert result["integration_result"]["builtImages"]
    assert result["integration_result"]["registeredComponents"]
    event_types = [event_type for event_type, _ in events]
    assert "integration_build_started" in event_types
    assert "integration_registration_started" in event_types
    assert "integration_registration_component_succeeded" in event_types


def test_integration_build_failure_emits_failed_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated_index = _write_component(tmp_path)
    events: list[tuple[str, dict[str, Any]]] = []

    def _failing_build(component_name: str, image: str, folder: Path) -> None:
        del component_name, image, folder
        raise RuntimeError("docker build failed")

    monkeypatch.setattr(
        "component_generation.nodes.integration._run_docker_build",
        _failing_build,
    )

    state = {
        "run_integration": True,
        "session_dir": str(tmp_path),
        "generated_index": generated_index,
        "event_callback": lambda event_type, payload: events.append((event_type, payload)),
        "hitl_decision_getter": lambda **_: {"approved": True, "feedback": ""},
    }

    with pytest.raises(RuntimeError, match="docker build failed"):
        node_integration(state)

    event_types = [event_type for event_type, _ in events]
    assert "integration_failed" in event_types
