"""Streaming chat backends over plain HTTP (no vendor SDKs).

ollama:<model>     local, default. e.g. ollama:qwen2.5-coder:7b
openai:<model>     any OpenAI-compatible endpoint (OpenAI, Groq, OpenRouter, vLLM, LM Studio, llama.cpp server)
                   set OPENAI_BASE_URL and OPENAI_API_KEY
anthropic:<model>  e.g. anthropic:claude-sonnet-5-5, set ANTHROPIC_API_KEY
"""
from __future__ import annotations

import json
from typing import Iterator

import httpx

from .config import Settings

Message = dict[str, str]


class LLMError(RuntimeError):
    pass


class LLM:
    name: str

    def stream(self, system: str, messages: list[Message]) -> Iterator[str]: ...


def _raise(r: httpx.Response, what: str):
    if r.status_code >= 400:
        r.read()
        raise LLMError(f"{what} returned HTTP {r.status_code}: {r.text[:500]}")


class Ollama(LLM):
    def __init__(self, model: str, url: str, num_ctx: int = 12288):
        self.name, self.model, self.url, self.num_ctx = f"ollama:{model}", model, url.rstrip("/"), num_ctx

    def stream(self, system, messages):
        body = {"model": self.model, "stream": True,
                "messages": [{"role": "system", "content": system}, *messages],
                "options": {"temperature": 0.1, "num_ctx": self.num_ctx}}
        try:
            with httpx.stream("POST", f"{self.url}/api/chat", json=body, timeout=httpx.Timeout(600, connect=5)) as r:
                _raise(r, "Ollama")
                for line in r.iter_lines():
                    if not line:
                        continue
                    ev = json.loads(line)
                    if ev.get("error"):
                        raise LLMError(f"Ollama: {ev['error']}")
                    if piece := ev.get("message", {}).get("content"):
                        yield piece
                    if ev.get("done"):
                        break
        except httpx.ConnectError as e:
            raise LLMError(f"Can't reach Ollama at {self.url}. Is `ollama serve` running? ({e})") from e


class OpenAICompat(LLM):
    def __init__(self, model: str, base_url: str, api_key: str):
        self.name, self.model, self.base, self.key = f"openai:{model}", model, base_url.rstrip("/"), api_key

    def stream(self, system, messages):
        body = {"model": self.model, "stream": True, "temperature": 0.1,
                "messages": [{"role": "system", "content": system}, *messages]}
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        with httpx.stream("POST", f"{self.base}/chat/completions", json=body, headers=headers,
                          timeout=httpx.Timeout(600, connect=10)) as r:
            _raise(r, self.base)
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                choices = json.loads(data).get("choices") or [{}]
                if piece := (choices[0].get("delta") or {}).get("content"):
                    yield piece


class Anthropic(LLM):
    def __init__(self, model: str, api_key: str, max_tokens: int = 2048):
        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        self.name, self.model, self.key, self.max_tokens = f"anthropic:{model}", model, api_key, max_tokens

    def stream(self, system, messages):
        body = {"model": self.model, "max_tokens": self.max_tokens, "system": system,
                "messages": messages, "stream": True, "temperature": 0.1}
        headers = {"x-api-key": self.key, "anthropic-version": "2023-06-01"}
        with httpx.stream("POST", "https://api.anthropic.com/v1/messages", json=body, headers=headers,
                          timeout=httpx.Timeout(600, connect=10)) as r:
            _raise(r, "Anthropic API")
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                ev = json.loads(line[5:])
                if ev.get("type") == "content_block_delta" and (t := ev.get("delta", {}).get("text")):
                    yield t
                elif ev.get("type") == "error":
                    raise LLMError(f"Anthropic: {ev.get('error')}")


def get_llm(spec: str, settings: Settings) -> LLM:
    kind, _, model = spec.partition(":")
    if not model:
        raise ValueError(f"LLM spec {spec!r} needs a model, e.g. ollama:qwen2.5-coder:7b")
    if kind == "ollama":
        return Ollama(model, settings.ollama_url)
    if kind == "openai":
        return OpenAICompat(model, settings.openai_base_url, settings.openai_api_key)
    if kind == "anthropic":
        return Anthropic(model, settings.anthropic_api_key)
    raise ValueError(f"unknown LLM backend {kind!r}; use ollama:, openai: or anthropic:")
