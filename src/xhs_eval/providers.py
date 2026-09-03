from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import httpx

from xhs_eval.io import read_jsonl


class ProviderError(RuntimeError):
    """Raised when a model provider cannot return a usable response."""


class TextProvider(ABC):
    name: str

    @abstractmethod
    def generate(self, messages: list[dict[str, str]], sample_id: str) -> str:
        """Return one text completion for a chat-style request."""


class ReplayProvider(TextProvider):
    """Deterministic offline provider used for reproducible demos and tests."""

    def __init__(self, responses_path: str | Path, name: str = "replay") -> None:
        self.name = name
        rows = read_jsonl(responses_path)
        self._responses: dict[str, Any] = {}
        for row in rows:
            if "id" not in row or "output" not in row:
                raise ProviderError("replay rows require 'id' and 'output'")
            self._responses[str(row["id"])] = row["output"]

    def generate(self, messages: list[dict[str, str]], sample_id: str) -> str:
        del messages
        if sample_id not in self._responses:
            raise ProviderError(f"no replay response for sample {sample_id!r}")
        output = self._responses[sample_id]
        if isinstance(output, str):
            return output
        return json.dumps(output, ensure_ascii=False)


class OpenAICompatibleProvider(TextProvider):
    """Minimal client for OpenAI-compatible chat-completions endpoints."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key_env: str = "LLM_API_KEY",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        response_format_json: bool = False,
    ) -> None:
        self.model = os.path.expandvars(model)
        self.name = self.model
        self.base_url = os.path.expandvars(base_url).rstrip("/")
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.response_format_json = response_format_json

    def generate(self, messages: list[dict[str, str]], sample_id: str) -> str:
        del sample_id
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise ProviderError(f"missing API key environment variable: {self.api_key_env}")

        endpoint = self.base_url
        if not endpoint.endswith("/chat/completions"):
            endpoint = f"{endpoint}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.response_format_json:
            payload["response_format"] = {"type": "json_object"}

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    response = client.post(
                        endpoint,
                        headers={"Authorization": f"Bearer {api_key}"},
                        json=payload,
                    )
                    response.raise_for_status()
                    body = response.json()
                return str(body["choices"][0]["message"]["content"])
            except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(2**attempt)
        raise ProviderError(f"provider request failed: {last_error}") from last_error


def build_provider(config: dict[str, Any], *, base_dir: Path) -> TextProvider:
    provider_type = config.get("type")
    if provider_type == "replay":
        response_path = resolve_path(config["responses_path"], base_dir)
        return ReplayProvider(response_path, name=config.get("name", "replay"))
    if provider_type == "openai_compatible":
        return OpenAICompatibleProvider(
            model=config["model"],
            base_url=config["base_url"],
            api_key_env=config.get("api_key_env", "LLM_API_KEY"),
            temperature=float(config.get("temperature", 0.0)),
            max_tokens=int(config.get("max_tokens", 1024)),
            timeout_seconds=float(config.get("timeout_seconds", 60.0)),
            max_retries=int(config.get("max_retries", 2)),
            response_format_json=bool(config.get("response_format_json", False)),
        )
    raise ProviderError(f"unsupported provider type: {provider_type!r}")


def resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    if path.is_absolute():
        return path
    candidate = (base_dir / path).resolve()
    if candidate.exists() or not (Path.cwd() / path).exists():
        return candidate
    return (Path.cwd() / path).resolve()
