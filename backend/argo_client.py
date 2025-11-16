from __future__ import annotations

import os
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import httpx  # type: ignore[import-not-found]


class ArgoClientError(RuntimeError):
    """Generic error raised when interacting with the Argo Server API."""


class ArgoNotFoundError(ArgoClientError):
    """Raised when the requested workflow was not found."""


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class ArgoClient:
    """Minimal HTTP client for the Argo Server REST API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
        timeout: Optional[float] = None,
        verify: Optional[bool] = None,
    ) -> None:
        resolved_base = base_url or os.getenv("ARGO_SERVER_BASE_URL") or "https://localhost:2746"
        self._base_url = resolved_base.rstrip("/")

        parsed = urlparse(self._base_url)

        self._token = token or os.getenv("ARGO_SERVER_AUTH_TOKEN") or None

        if timeout is None:
            timeout_env = os.getenv("ARGO_SERVER_TIMEOUT_SECONDS")
            if timeout_env:
                try:
                    timeout = float(timeout_env)
                except ValueError as exc:  # pragma: no cover - defensive
                    raise ArgoClientError(f"Invalid ARGO_SERVER_TIMEOUT_SECONDS value: {timeout_env}") from exc
            else:
                timeout = 10.0
        self._timeout = timeout

        if verify is None:
            default_skip = parsed.hostname in {"localhost", "127.0.0.1"}
            skip_verify = _env_flag("ARGO_SERVER_INSECURE_SKIP_VERIFY", default_skip)
            verify = not skip_verify
        self._verify = verify

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        accept: str = "application/json",
    ) -> httpx.Response:
        url = f"{self._base_url}{path}"
        headers = {
            "Accept": accept,
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        try:
            response = httpx.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                timeout=self._timeout,
                verify=self._verify,
            )
        except httpx.HTTPError as exc:  # pragma: no cover - network layer
            raise ArgoClientError(f"Failed to reach Argo server at {url}: {exc}") from exc

        return response

    def get_workflow(self, namespace: str, workflow_name: str) -> Dict[str, Any]:
        """Fetch a workflow resource by namespace and name."""
        response = self._request("GET", f"/api/v1/workflows/{namespace}/{workflow_name}")

        if response.status_code == 404:
            raise ArgoNotFoundError(f"Workflow '{workflow_name}' not found in namespace '{namespace}'.")

        if response.status_code == 401:
            raise ArgoClientError("Unauthorized when querying Argo server (401).")

        if response.status_code == 403:
            raise ArgoClientError("Forbidden when querying Argo server (403).")

        if not response.is_success:
            text = response.text.strip()
            detail = f"{response.status_code} {response.reason_phrase}"
            if text:
                detail = f"{detail}: {text}"
            raise ArgoClientError(f"Argo server responded with error: {detail}")

        try:
            return response.json()
        except ValueError as exc:
            raise ArgoClientError("Argo server returned invalid JSON payload.") from exc

    def get_workflow_logs(
        self,
        namespace: str,
        workflow_name: str,
        pod_name: str,
        *,
        container: Optional[str] = None,
        tail_lines: Optional[int] = None,
        since_seconds: Optional[int] = None,
    ) -> str:
        """Fetch logs for a specific workflow pod."""

        params: Dict[str, Any] = {
            "podName": pod_name,
        }
        if container:
            params["logOptions.container"] = container
        if tail_lines is not None:
            params["logOptions.tailLines"] = str(tail_lines)
        if since_seconds is not None:
            params["logOptions.sinceSeconds"] = str(since_seconds)

        response = self._request(
            "GET",
            f"/api/v1/workflows/{namespace}/{workflow_name}/log",
            params=params,
            accept="text/plain, */*",
        )

        if response.status_code == 404:
            raise ArgoNotFoundError(
                f"Logs for pod '{pod_name}' in workflow '{workflow_name}' not found in namespace '{namespace}'."
            )

        if response.status_code in {401, 403}:
            detail = "Unauthorized" if response.status_code == 401 else "Forbidden"
            raise ArgoClientError(f"{detail} when retrieving logs from Argo server ({response.status_code}).")

        if not response.is_success:
            text = response.text.strip()
            detail = f"{response.status_code} {response.reason_phrase}"
            if text:
                detail = f"{detail}: {text}"
            raise ArgoClientError(f"Argo server responded with error when fetching logs: {detail}")

        return response.text or ""


