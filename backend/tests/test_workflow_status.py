import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

sys.path.append(str(Path(__file__).resolve().parents[1]))

import main  # noqa: E402


def test_workflow_status_maps_nodes_and_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "status": {
            "phase": "Succeeded",
            "startedAt": "2024-01-01T00:01:00Z",
            "nodes": {
                "node-1": {
                    "displayName": "task-a",
                    "phase": "Succeeded",
                    "type": "Pod",
                    "id": "123",
                    "message": "ok",
                    "progress": "1/1",
                    "startedAt": "2024-01-01T00:01:10Z",
                    "finishedAt": "2024-01-01T00:01:20Z",
                },
                "node-2": {
                    "name": "task-b",
                    "phase": "Running",
                    "type": "Pod",
                    "id": "456",
                },
            },
        }
    }

    class StubArgoClient:
        def get_workflow(self, namespace: str, workflow_name: str):
            assert namespace == "test-ns"
            assert workflow_name == "wf-1"
            return payload

    monkeypatch.setattr(main, "ArgoClient", lambda: StubArgoClient())

    response = main.get_workflow_status(workflow_name="wf-1", namespace="test-ns")

    assert response.workflow_name == "wf-1"
    assert response.namespace == "test-ns"
    assert response.phase == "Succeeded"
    assert response.finished is True
    assert response.updated_at == "2024-01-01T00:01:00Z"

    assert "task-a" in response.nodes
    assert response.nodes["task-a"].slug == "task-a"
    assert response.nodes["task-a"].phase == "Succeeded"
    assert response.nodes["task-a"].display_name == "task-a"

    assert "task-b" in response.nodes
    assert response.nodes["task-b"].slug == "task-b"
    assert response.nodes["task-b"].phase == "Running"
    assert response.nodes["task-b"].display_name == "task-b"


def test_workflow_status_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubArgoClient:
        def get_workflow(self, namespace: str, workflow_name: str):
            raise main.ArgoNotFoundError("not found")

    monkeypatch.setattr(main, "ArgoClient", lambda: StubArgoClient())

    with pytest.raises(HTTPException) as excinfo:
        main.get_workflow_status(workflow_name="wf-404", namespace="test-ns")

    assert excinfo.value.status_code == 404


def test_workflow_status_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubArgoClient:
        def get_workflow(self, namespace: str, workflow_name: str):
            raise main.ArgoClientError("bad gateway")

    monkeypatch.setattr(main, "ArgoClient", lambda: StubArgoClient())

    with pytest.raises(HTTPException) as excinfo:
        main.get_workflow_status(workflow_name="wf-500", namespace="test-ns")

    assert excinfo.value.status_code == 502
