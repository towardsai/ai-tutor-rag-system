"""Provider clients: one tiny interface, three backends.

This lesson's experiments sit deliberately outside the course's three-provider
pattern: context-engineering economics are cache-discount economics, and those
live on specific providers. The tested path is Gemini (``google-genai``). The
DeepSeek and Ollama backends speak the OpenAI-compatible chat API over plain
HTTP; they are configured and ready but NOT covered by the course's executed
runs. Treat them as starting points, not verified paths.

Every backend implements one method::

    generate(system, messages, max_output_tokens) -> GenerateResult

where ``messages`` is the lab's neutral history format:
``[{"role": "user"|"assistant", "text": str}, ...]``.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field


@dataclass
class GenerateResult:
    text: str
    usage: dict = field(default_factory=dict)  # input/output/cached_input tokens
    latency_ms: int = 0
    model: str = ""


class GeminiProvider:
    """Google Gemini via the ``google-genai`` SDK. The tested backend.

    Reads ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``) from the environment.

    ``cached_input_tokens`` comes from ``usage_metadata.cached_content_token_count``.
    That field is ``None``, not ``0``, when the API returns no cache accounting
    at all, and the difference matters: ``0`` means "measured, no hits" while
    ``None`` means "not measured". We preserve it, because a report that prints
    "0% cache hits" for unreported data is asserting something the
    instrumentation never observed. Verified on 2026-07-30: three back-to-back
    calls sharing an identical 8,108-token prefix on ``gemini-3.5-flash-lite``
    all returned ``None`` here.
    """

    name = "gemini"

    def __init__(self, model: str = "gemini-3.5-flash-lite"):
        from google import genai

        self.model = model
        self._client = genai.Client()

    def generate(
        self,
        system: str,
        messages: list[dict],
        max_output_tokens: int = 1_024,
    ) -> GenerateResult:
        from google.genai import types

        contents = [
            types.Content(
                role="user" if m["role"] == "user" else "model",
                parts=[types.Part(text=m["text"])],
            )
            for m in messages
        ]
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.3,
            max_output_tokens=max_output_tokens,
        )
        start = time.monotonic()
        response = self._client.models.generate_content(
            model=self.model, contents=contents, config=config
        )
        latency_ms = int((time.monotonic() - start) * 1000)
        meta = response.usage_metadata
        cached = meta.cached_content_token_count
        usage = {
            "input_tokens": int(meta.prompt_token_count or 0),
            "output_tokens": int(meta.candidates_token_count or 0)
            + int(getattr(meta, "thoughts_token_count", 0) or 0),
            # None is preserved on purpose. See the class docstring.
            "cached_input_tokens": None if cached is None else int(cached),
        }
        return GenerateResult(
            text=response.text or "",
            usage=usage,
            latency_ms=latency_ms,
            model=self.model,
        )


class OpenAICompatProvider:
    """Any OpenAI-compatible ``/chat/completions`` endpoint over plain HTTP.

    Used for the DeepSeek and Ollama configurations. UNTESTED in the course's
    executed runs; the request/usage shapes follow each provider's API docs.
    DeepSeek reports prefix-cache hits as ``prompt_cache_hit_tokens``; Ollama
    reports no cache field (and costs $0 anyway).
    """

    def __init__(self, base_url: str, model: str, api_key: str | None, name: str):
        import httpx

        self.name = name
        self.model = model
        self._base_url = base_url.rstrip("/")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(headers=headers, timeout=120.0)

    def generate(
        self,
        system: str,
        messages: list[dict],
        max_output_tokens: int = 1_024,
    ) -> GenerateResult:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}]
            + [{"role": m["role"], "content": m["text"]} for m in messages],
            "temperature": 0.3,
            "max_tokens": max_output_tokens,
        }
        start = time.monotonic()
        response = self._client.post(
            f"{self._base_url}/chat/completions", json=payload
        )
        response.raise_for_status()
        latency_ms = int((time.monotonic() - start) * 1000)
        data = response.json()
        usage = data.get("usage") or {}
        cached = usage.get("prompt_cache_hit_tokens")
        return GenerateResult(
            text=data["choices"][0]["message"]["content"] or "",
            usage={
                "input_tokens": int(usage.get("prompt_tokens") or 0),
                "output_tokens": int(usage.get("completion_tokens") or 0),
                # None when the endpoint reports no cache accounting (Ollama).
                "cached_input_tokens": None if cached is None else int(cached),
            },
            latency_ms=latency_ms,
            model=str(data.get("model") or self.model),
        )


def get_provider(name: str, model: str | None = None):
    """Build a provider by name: ``gemini`` (tested), ``deepseek``, ``ollama``."""
    if name == "gemini":
        return GeminiProvider(model or "gemini-3.5-flash-lite")
    if name == "deepseek":
        return OpenAICompatProvider(
            base_url="https://api.deepseek.com",
            # Use the explicit model id, not the legacy "deepseek-chat" alias,
            # which DeepSeek deprecated on 2026-07-24.
            model=model or "deepseek-v4-flash",
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            name="deepseek",
        )
    if name == "ollama":
        return OpenAICompatProvider(
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            model=model or "qwen3:8b",
            api_key="ollama",  # any non-empty string; Ollama ignores it
            name="ollama",
        )
    raise ValueError(f"Unknown provider {name!r}; expected gemini, deepseek, ollama")
