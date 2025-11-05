"""Utilities for working with MinIO from the backend service."""

from __future__ import annotations

import os
import threading
import re
from functools import lru_cache
from typing import Final

from minio import Minio
from minio.error import S3Error

_client_lock = threading.Lock()
_DEFAULT_ENDPOINT: Final[str] = "localhost:9000"
_DEFAULT_BUCKET: Final[str] = "artifacts-input"


def _get_env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "t", "yes", "y"}


def _get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def get_input_bucket() -> str:
    """Return the bucket to store uploaded artifacts, defaulting when unset."""

    return os.getenv("MINIO_INPUT_BUCKET", _DEFAULT_BUCKET)


@lru_cache(maxsize=1)
def _build_client() -> Minio:
    endpoint = os.getenv("MINIO_ENDPOINT", _DEFAULT_ENDPOINT)
    access_key = _get_required_env("MINIO_ACCESS_KEY")
    secret_key = _get_required_env("MINIO_SECRET_KEY")
    secure = _get_env_bool("MINIO_SECURE", False)
    region = os.getenv("MINIO_REGION")

    return Minio(
        endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure,
        region=region,
    )


def get_minio_client() -> Minio:
    """Return a cached MinIO client instance."""

    # The cached builder handles the heavy lifting, but we wrap in a lock to
    # guard initial creation under heavy concurrency.
    if _build_client.cache_info().currsize:
        return _build_client()

    with _client_lock:
        return _build_client()


def ensure_bucket(client: Minio, bucket_name: str | None = None) -> None:
    """Ensure the target bucket exists, creating it if necessary."""

    target = bucket_name or get_input_bucket()
    try:
        exists = client.bucket_exists(target)
        if not exists:
            client.make_bucket(target)
    except S3Error as exc:
        raise RuntimeError(f"Failed to ensure bucket '{target}': {exc}") from exc


def ensure_default_bucket() -> str:
    """Ensure the configured default bucket exists and return its name."""

    bucket = get_input_bucket()
    client = get_minio_client()
    ensure_bucket(client, bucket)
    return bucket


def sanitize_path_segment(value: str, fallback: str) -> str:
    """Sanitize user-provided identifiers for inclusion in object keys."""

    candidate = re.sub(r"[^0-9A-Za-z._-]", "-", value.strip())
    candidate = re.sub(r"-+", "-", candidate).strip("-._")
    return candidate or fallback


def build_session_node_prefix(session_id: str, node_id: str) -> str:
    session_part = sanitize_path_segment(session_id, "session")
    node_part = sanitize_path_segment(node_id, "node")
    return f"sessions/{session_part}/nodes/{node_part}"


