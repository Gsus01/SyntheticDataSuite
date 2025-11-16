from __future__ import annotations

import threading
from typing import Optional

from kubernetes import client, config
from kubernetes.config.config_exception import ConfigException

_core_v1_lock = threading.Lock()
_cached_core_v1: Optional[client.CoreV1Api] = None


def get_core_v1_client() -> client.CoreV1Api:
    """Return a cached CoreV1Api client, loading Kubernetes config on demand."""
    global _cached_core_v1

    with _core_v1_lock:
        if _cached_core_v1 is not None:
            return _cached_core_v1

        try:
            config.load_incluster_config()
        except ConfigException:
            config.load_kube_config()

        _cached_core_v1 = client.CoreV1Api()
        return _cached_core_v1


