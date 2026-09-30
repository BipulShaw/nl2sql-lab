"""Ollama's native /api/chat with thinking off, checking on every call that the model read the whole prompt.

Ollama cuts an over-long prompt to fit num_ctx and reports no error (the probe sent 15,921 tokens; 4,098 were
read). So the client counts the prompt with the model's own tokenizer before sending, refuses prompts that
cannot fit, and requires the server's prompt_eval_count to equal its own count afterwards (ADR-012).

A reply Ollama did not finish (no `done`, no prompt_eval_count) is retried like a server error; one that stays
unfinished is kept as an incomplete reply, like one cut off at max_new_tokens (ADR-019)."""

import logging
import time
from collections.abc import Callable
from dataclasses import asdict

import httpx

from nl2sql.llm.base import Generation
from nl2sql.llm.cache import ResponseCache, cache_key

logger = logging.getLogger(__name__)


class PromptTooLongError(ValueError):
    """Prompt plus reply budget does not fit the context window; nothing was sent."""


class PromptCountMismatchError(RuntimeError):
    """The server read a different number of prompt tokens than the client counted: truncation, or a chat
    template that differs from the tokenizer's. Either way the run's prompts are not what we think."""


class OllamaLLM:
    backend = "ollama"

    def __init__(
        self,
        model: str,
        count_chat_tokens: Callable[[list[dict[str, str]]], int],
        base_url: str = "http://localhost:11434",
        num_ctx: int = 8192,
        max_new_tokens: int = 256,
        seed: int = 42,
        cache: ResponseCache | None = None,
        timeout_s: float = 300.0,  # per read; a 256-token reply takes seconds, a cold 9B load under a minute
        retries: int = 3,
        backoff_s: float = 1.0,  # before retry n (from 0): backoff_s * 2**n
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self.count_chat_tokens = count_chat_tokens
        self.num_ctx, self.max_new_tokens, self.seed = num_ctx, max_new_tokens, seed
        self.cache, self.retries, self.backoff_s = cache, retries, backoff_s
        self.http = httpx.Client(base_url=base_url, timeout=timeout_s, transport=transport)
        self.info = self.describe()  # fails fast if the server is down or the model is not pulled

    def options(self) -> dict:
        # temperature 0 = greedy; presence_penalty 0 overrides the model's Modelfile default (ADR-007)
        return {
            "num_ctx": self.num_ctx,
            "num_predict": self.max_new_tokens,
            "temperature": 0,
            "presence_penalty": 0,
            "seed": self.seed,
        }

    def describe(self) -> dict:
        version = self.http.get("/api/version").json()["version"]
        models = self.http.get("/api/tags").json()["models"]
        entry = next((m for m in models if m["name"].lower() == self.model.lower()), None)
        if entry is None:
            raise RuntimeError(f"{self.model!r} is not pulled in Ollama; run: ollama pull {self.model}")
        details = entry.get("details") or {}
        return {
            "backend": self.backend,
            "ollama_version": version,
            "model": self.model,
            "digest": entry["digest"],
            "quantization": details.get("quantization_level"),
            "parameter_size": details.get("parameter_size"),
            "options": self.options(),
            "think": False,
        }

    def request(self, messages: list[dict[str, str]]) -> tuple[dict, str]:
        """The /api/chat payload for these messages, and its response-cache key."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": self.options(),
        }
        key = cache_key(
            backend=self.backend,
            digest=self.info["digest"],
            quantization=self.info["quantization"],
            adapter=None,
            payload=payload,
        )
        return payload, key

    def generate(self, messages: list[dict[str, str]]) -> Generation:
        prompt_tokens = self.count_chat_tokens(messages)
        if prompt_tokens + self.max_new_tokens > self.num_ctx:
            raise PromptTooLongError(
                f"{prompt_tokens} prompt + {self.max_new_tokens} new > num_ctx {self.num_ctx}"
            )
        payload, key = self.request(messages)
        if self.cache is not None and (hit := self.cache.get(key)) is not None:
            return Generation(**hit, cached=True)

        start = time.perf_counter()
        data = self.post_chat(payload)
        latency_ms = (time.perf_counter() - start) * 1000
        message = data["message"]
        if message.get("thinking"):
            raise RuntimeError("the model returned a thinking block although think=false was sent")
        if data.get("done") is not True:
            # Ollama ended the reply early on every attempt, with no counts (ADR-019). Kept like a reply cut
            # off at max_new_tokens: the SQL it holds is scored. Output tokens are unknown, recorded as 0.
            generation = Generation(message.get("content", ""), prompt_tokens, 0, latency_ms, "incomplete")
        elif data.get("prompt_eval_count") != prompt_tokens:
            raise PromptCountMismatchError(
                f"client counted {prompt_tokens} prompt tokens, server read {data.get('prompt_eval_count')}"
            )
        else:
            generation = Generation(
                text=message.get("content", ""),
                prompt_tokens=prompt_tokens,
                output_tokens=data.get("eval_count", 0),
                latency_ms=latency_ms,
                finish_reason=data.get("done_reason", "stop"),
            )
        if self.cache is not None:
            self.cache.put(key, {k: v for k, v in asdict(generation).items() if k != "cached"})
        return generation

    def post_chat(self, payload: dict) -> dict:
        """POST, retrying with backoff on connection errors, 5xx and unfinished replies; 4xx fails at once. A
        reply still unfinished after the last attempt is returned as it is."""
        error = ""
        for attempt in range(self.retries + 1):
            unfinished = None
            try:
                response = self.http.post("/api/chat", json=payload)
                if response.status_code < 500:
                    response.raise_for_status()
                    data = response.json()
                    if data.get("done") is True and "prompt_eval_count" in data:
                        return data
                    # Seen with the 4B: llama-server stopped partway through the prompt and Ollama still
                    # answered 200, without the counts (ADR-019).
                    unfinished = data
                    error = f"unfinished reply: done={data.get('done')!r}, fields {sorted(data)}"
                else:
                    error = f"HTTP {response.status_code}: {response.text[:300]}"
            except httpx.TransportError as exc:
                error = f"{type(exc).__name__}: {exc}"
            if attempt < self.retries:
                logger.warning("Ollama /api/chat attempt %d failed, retrying: %s", attempt + 1, error)
                time.sleep(self.backoff_s * 2**attempt)
        if unfinished is not None and "message" in unfinished:
            logger.warning(
                "Ollama /api/chat: still unfinished after %d attempts, kept as incomplete", attempt + 1
            )
            return unfinished
        raise RuntimeError(f"Ollama /api/chat failed {self.retries + 1} times; last error: {error}")

    def close(self) -> None:
        self.http.close()
