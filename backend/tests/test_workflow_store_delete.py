import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

import workflow_store  # noqa: E402


class _DummyRow:
    def __init__(self, workflow_id: str) -> None:
        self.workflow_id = workflow_id


class _FakeSession:
    def __init__(self, row: _DummyRow | None) -> None:
        self._row = row
        self.deleted_row = None
        self.flushed = False

    def get(self, _model, workflow_id: str):
        if self._row and self._row.workflow_id == workflow_id:
            return self._row
        return None

    def delete(self, row) -> None:
        self.deleted_row = row

    def flush(self) -> None:
        self.flushed = True


class _FakeDbSession:
    def __init__(self, session: _FakeSession) -> None:
        self._session = session

    def __enter__(self) -> _FakeSession:
        return self._session

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        return None


def test_delete_workflow_removes_existing_row(monkeypatch: pytest.MonkeyPatch) -> None:
    row = _DummyRow("wf-1")
    session = _FakeSession(row)
    monkeypatch.setattr(workflow_store, "db_session", lambda: _FakeDbSession(session))

    store = workflow_store.WorkflowStore()
    deleted_id = store.delete_workflow("wf-1")

    assert deleted_id == "wf-1"
    assert session.deleted_row is row
    assert session.flushed is True


def test_delete_workflow_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _FakeSession(None)
    monkeypatch.setattr(workflow_store, "db_session", lambda: _FakeDbSession(session))

    store = workflow_store.WorkflowStore()
    with pytest.raises(workflow_store.WorkflowNotFoundError):
        store.delete_workflow("wf-404")

