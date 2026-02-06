import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import main


def test_stream_workflow_logs_normalizes_json_lines(monkeypatch) -> None:
    class StubArgoClient:
        def get_workflow(self, namespace: str, workflow_name: str):
            return {"status": {"nodes": {}}}

        def get_workflow_logs(self, namespace: str, workflow_name: str, pod_name: str, *, container=None, follow=None):
            return (
                '{"result":{"content":"hello","podName":"pod-a"}}\n'
                '{"level":"info","msg":"structured"}\n'
                '{"level":"info","content":"structured-with-content","msg":"keep-me"}\n'
                '{"result":{"content":"world","podName":"pod-b"}}\n'
                'time="2024-01-01T00:00:00Z" level=info msg="plain" argo=true\n'
            )

    monkeypatch.setattr(main, "ArgoClient", lambda: StubArgoClient())

    response = main.stream_workflow_logs(
        workflow_name="wf-1",
        namespace="argo",
        follow=False,
        pod_name=None,
    )

    assert response.logs == "\n".join(
        [
            "pod-a: hello",
            '{"level":"info","msg":"structured"}',
            '{"level":"info","content":"structured-with-content","msg":"keep-me"}',
            "pod-b: world",
            'time="2024-01-01T00:00:00Z" level=info msg="plain" argo=true',
        ]
    )
