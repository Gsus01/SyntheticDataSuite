import sys
from pathlib import Path

import httpx

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argo_client


def test_create_workflow_posts_wrapper(monkeypatch) -> None:
    captured = {}

    def fake_request(method, url, headers=None, params=None, json=None, timeout=None, verify=None):
        captured["method"] = method
        captured["url"] = url
        captured["json"] = json
        return httpx.Response(200, json={"metadata": {"name": "wf-1"}})

    monkeypatch.setattr(argo_client.httpx, "request", fake_request)

    client = argo_client.ArgoClient(base_url="https://argo.example")
    result = client.create_workflow("argo", {"kind": "Workflow"})

    assert captured["method"] == "POST"
    assert captured["url"] == "https://argo.example/api/v1/workflows/argo"
    assert captured["json"]["namespace"] == "argo"
    assert captured["json"]["workflow"]["kind"] == "Workflow"
    assert result["metadata"]["name"] == "wf-1"
