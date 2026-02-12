from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from component_generation.run_manager import (  # noqa: E402
    ComponentGenerationRunManager,
    RunInputFile,
    RunInvalidStateError,
)


def _base_options(auto_approve: bool) -> dict:
    return {
        "include_markdown": False,
        "max_chars_per_file": 50000,
        "auto_approve": auto_approve,
        "run_integration": False,
        "no_llm": True,
        "no_structured_output": False,
        "provider": "ollama",
        "model": "qwen3:14b",
        "temperature": 0.0,
        "ollama_url": "http://localhost:11434",
        "openrouter_url": "https://openrouter.ai/api/v1",
    }


def _wait_until(
    manager: ComponentGenerationRunManager,
    run_id: str,
    predicate: Callable[[dict], bool],
    *,
    timeout: float = 12.0,
) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        snapshot = manager.get_run_snapshot(run_id)
        if predicate(snapshot):
            return snapshot
        time.sleep(0.15)
    raise AssertionError(f"Timed out waiting for run state. Last run id={run_id}")


def test_run_manager_autoapprove_finishes(tmp_path: Path) -> None:
    manager = ComponentGenerationRunManager(output_root=tmp_path)
    snapshot = manager.start_run(
        input_files=[
            RunInputFile(filename="sample.py", content=b"print('hello')\n"),
        ],
        options=_base_options(auto_approve=True),
    )
    run_id = snapshot["runId"]

    final = _wait_until(
        manager,
        run_id,
        lambda item: item["status"] in {"succeeded", "failed", "canceled"},
    )

    assert final["status"] == "succeeded"
    assert final["reviewStatus"] == "OK"
    assert len(final["generatedIndex"]) >= 1
    assert isinstance(final.get("logTail"), list)
    assert any("run_manager" in line for line in final["logTail"])

    events, terminal = manager.get_events_since(run_id, 0)
    event_types = {event["type"] for event in events}
    assert terminal is True
    assert "run_started" in event_types
    assert "run_finished" in event_types
    assert "log_line" in event_types


def test_run_manager_requires_hitl_decision(tmp_path: Path) -> None:
    manager = ComponentGenerationRunManager(output_root=tmp_path)
    snapshot = manager.start_run(
        input_files=[
            RunInputFile(filename="sample.py", content=b"print('hello')\n"),
        ],
        options=_base_options(auto_approve=False),
    )
    run_id = snapshot["runId"]

    waiting = _wait_until(manager, run_id, lambda item: item["status"] == "waiting_decision")
    assert waiting["awaitingDecision"] is True
    assert waiting["pendingPlan"] is not None
    assert isinstance(waiting.get("pendingPrettyPlan"), str)
    assert "Proposed plan" in waiting["pendingPrettyPlan"]

    manager.submit_decision(run_id, approved=False, feedback="Refina el plan")
    waiting_again = _wait_until(
        manager,
        run_id,
        lambda item: item["status"] == "waiting_decision" and item["lastSeq"] > waiting["lastSeq"],
    )
    assert waiting_again["awaitingDecision"] is True

    manager.submit_decision(run_id, approved=True, feedback="")
    final = _wait_until(
        manager,
        run_id,
        lambda item: item["status"] in {"succeeded", "failed", "canceled"},
    )
    assert final["status"] == "succeeded"


def test_cancel_all_active_runs_cancels_waiting_run(tmp_path: Path) -> None:
    manager = ComponentGenerationRunManager(output_root=tmp_path)
    snapshot = manager.start_run(
        input_files=[
            RunInputFile(filename="sample.py", content=b"print('hello')\n"),
        ],
        options=_base_options(auto_approve=False),
    )
    run_id = snapshot["runId"]

    _wait_until(manager, run_id, lambda item: item["status"] == "waiting_decision")

    result = manager.cancel_all_active_runs()
    assert result["canceledCount"] == 1
    assert run_id in result["canceledRunIds"]

    final = _wait_until(
        manager,
        run_id,
        lambda item: item["status"] in {"succeeded", "failed", "canceled"},
    )
    assert final["status"] == "canceled"


def test_submit_decision_stage_mismatch_raises(tmp_path: Path) -> None:
    manager = ComponentGenerationRunManager(output_root=tmp_path)
    snapshot = manager.start_run(
        input_files=[
            RunInputFile(filename="sample.py", content=b"print('hello')\n"),
        ],
        options=_base_options(auto_approve=False),
    )
    run_id = snapshot["runId"]

    waiting = _wait_until(manager, run_id, lambda item: item["status"] == "waiting_decision")
    assert waiting["decisionStage"] == "plan"

    try:
        manager.submit_decision(
            run_id,
            approved=True,
            feedback="",
            stage="integration",
        )
    except RunInvalidStateError:
        pass
    else:
        raise AssertionError("Expected RunInvalidStateError for stage mismatch")
