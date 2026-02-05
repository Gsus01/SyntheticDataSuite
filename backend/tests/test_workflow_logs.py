import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import main


def test_stream_workflow_logs_aggregates_pods(monkeypatch) -> None:
    class StubArgoClient:
        def get_workflow(self, namespace: str, workflow_name: str):
            return {
                "status": {
                    "nodes": {
                        "node-b": {
                            "podName": "pod-b",
                            "startedAt": "2024-01-01T00:00:02Z",
                        },
                        "node-a": {
                            "podName": "pod-a",
                            "startedAt": "2024-01-01T00:00:01Z",
                        },
                    }
                }
            }

        def get_workflow_logs(self, namespace: str, workflow_name: str, pod_name: str, *, container=None, follow=None):
            if pod_name is None:
                return ""
            return f"logs-{pod_name}"

    monkeypatch.setattr(main, "ArgoClient", lambda: StubArgoClient())

    response = main.stream_workflow_logs(
        workflow_name="wf-1",
        namespace="argo",
        follow=False,
        pod_name=None,
    )

    expected = "\n".join(
        [
            "pod-a: logs-pod-a",
            "pod-b: logs-pod-b",
        ]
    )
    assert response.logs == expected


def test_stream_workflow_logs_normalizes_json_lines(monkeypatch) -> None:
    class StubArgoClient:
        def get_workflow(self, namespace: str, workflow_name: str):
            return {"status": {"nodes": {}}}

        def get_workflow_logs(self, namespace: str, workflow_name: str, pod_name: str, *, container=None, follow=None):
            if pod_name is None:
                return (
                    '{"result":{"content":"hello","podName":"pod-a"}}\n'
                    '{"result":{"content":"world","podName":"pod-b"}}\n'
                    'time="2024-01-01T00:00:00Z" level=info msg="ignored" argo=true\n'
                )
            return ""

    monkeypatch.setattr(main, "ArgoClient", lambda: StubArgoClient())

    response = main.stream_workflow_logs(
        workflow_name="wf-1",
        namespace="argo",
        follow=False,
        pod_name=None,
    )

    assert response.logs == "pod-a: hello\npod-b: world"
