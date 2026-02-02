from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


Message = Dict[str, str]


class LLMClient:
    def invoke(
        self, messages: List[Message], format_schema: Optional[Dict[str, Any]] = None
    ) -> str:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class OllamaClient(LLMClient):
    model: str
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    timeout_seconds: float = 120.0

    def invoke(
        self, messages: List[Message], format_schema: Optional[Dict[str, Any]] = None
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if format_schema is not None:
            payload["format"] = format_schema
        url = self.base_url.rstrip("/") + "/api/chat"
        logger.info("llm: calling ollama model=%s url=%s", self.model, url)
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        message = data.get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("Unexpected Ollama response shape.")
        return content


@dataclass
class OpenRouterClient(LLMClient):
    model: str
    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"
    temperature: float = 0.0
    timeout_seconds: float = 120.0
    provider: Optional[Dict[str, Any]] = None

    def invoke(
        self, messages: List[Message], format_schema: Optional[Dict[str, Any]] = None
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.provider:
            payload["provider"] = self.provider
        if format_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": format_schema},
            }

        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        logger.info("llm: calling openrouter model=%s url=%s", self.model, url)
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        choices = data.get("choices") or []
        if not choices:
            raise ValueError("OpenRouter response missing choices.")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("Unexpected OpenRouter response shape.")
        return content


def make_llm(
    *,
    provider: str,
    model: str,
    base_url: Optional[str] = None,
    temperature: float = 0.0,
    api_key: Optional[str] = None,
    openrouter_provider: Optional[Dict[str, Any]] = None,
) -> LLMClient:
    provider_key = (provider or "").strip().lower()
    if provider_key == "ollama":
        return OllamaClient(
            model=model,
            base_url=base_url or "http://localhost:11434",
            temperature=temperature,
        )
    if provider_key == "openrouter":
        if not api_key:
            raise ValueError("Missing OPENROUTER_API_KEY (or --openrouter-key).")
        return OpenRouterClient(
            model=model,
            api_key=api_key,
            base_url=base_url or "https://openrouter.ai/api/v1",
            temperature=temperature,
            provider=openrouter_provider,
        )
    raise ValueError(f"Unknown provider: {provider!r}")


def extract_json_block(text: str) -> Any:
    """Best-effort JSON extraction (supports object or array)."""
    text = text.strip()
    if text.startswith("{") or text.startswith("["):
        return json.loads(text)

    start_obj = text.find("{")
    start_arr = text.find("[")
    starts = [s for s in [start_obj, start_arr] if s != -1]
    if not starts:
        raise ValueError("No JSON found in response.")

    start = min(starts)
    end = max(text.rfind("}"), text.rfind("]"))
    if end == -1 or end <= start:
        raise ValueError("Incomplete JSON in response.")

    snippet = text[start : end + 1]
    return json.loads(snippet)
