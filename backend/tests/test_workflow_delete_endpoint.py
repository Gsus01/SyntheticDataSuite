import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

sys.path.append(str(Path(__file__).resolve().parents[1]))

import main  # noqa: E402


def test_delete_workflow_endpoint_returns_success(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubStore:
        def delete_workflow(self, workflow_id: str) -> str:
            assert workflow_id == "wf-1"
            return workflow_id

    monkeypatch.setattr(main, "WorkflowStore", lambda: StubStore())

    response = main.delete_workflow_definition("wf-1")

    assert response.workflow_id == "wf-1"
    assert response.deleted is True


def test_delete_workflow_endpoint_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubStore:
        def delete_workflow(self, workflow_id: str) -> str:
            raise main.WorkflowDefinitionNotFoundError(
                f"No se encontró el workflow con id '{workflow_id}'."
            )

    monkeypatch.setattr(main, "WorkflowStore", lambda: StubStore())

    with pytest.raises(HTTPException) as excinfo:
        main.delete_workflow_definition("wf-404")

    assert excinfo.value.status_code == 404


def test_delete_workflow_endpoint_returns_500(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubStore:
        def delete_workflow(self, workflow_id: str) -> str:
            raise main.WorkflowStoreError("db failure")

    monkeypatch.setattr(main, "WorkflowStore", lambda: StubStore())

    with pytest.raises(HTTPException) as excinfo:
        main.delete_workflow_definition("wf-500")

    assert excinfo.value.status_code == 500

