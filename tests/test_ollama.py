import json
from pathlib import Path

import httpx
import pytest

from nl2sql.llm.cache import ResponseCache
from nl2sql.llm.ollama import OllamaLLM, PromptCountMismatchError, PromptTooLongError

MESSAGES = [{"role": "user", "content": "How many singers?"}]


class FakeOllama:
    """Answers /api/version, /api/tags and /api/chat the way Ollama does, and records chat payloads."""

    def __init__(self, prompt_eval_count: int, thinking: str = "", unfinished: int = 0) -> None:
        self.prompt_eval_count, self.thinking = prompt_eval_count, thinking
        self.unfinished = unfinished  # the first this many chats get a 200 without done or counts
        self.chats: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.34.0"})
        if request.url.path == "/api/tags":
            details = {"quantization_level": "Q4_K_M", "parameter_size": "9.7B"}
            return httpx.Response(
                200, json={"models": [{"name": "qwen3.5:9b", "digest": "abc", "details": details}]}
            )
        self.chats.append(json.loads(request.content))
        if len(self.chats) <= self.unfinished:
            return httpx.Response(200, json={"message": {"role": "assistant", "content": ""}, "done": False})
        message = {"role": "assistant", "content": "```sql\nSELECT 1\n```", "thinking": self.thinking}
        body = {
            "message": message,
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": self.prompt_eval_count,
            "eval_count": 9,
        }
        return httpx.Response(200, json=body)


def make_llm(server: FakeOllama, count: int, **kwargs: object) -> OllamaLLM:
    return OllamaLLM("qwen3.5:9B", lambda _: count, transport=httpx.MockTransport(server), **kwargs)


def test_request_turns_thinking_off_and_pins_decoding() -> None:
    server = FakeOllama(prompt_eval_count=100)

    generation = make_llm(server, 100, num_ctx=8192, max_new_tokens=256).generate(MESSAGES)

    [payload] = server.chats
    assert payload["think"] is False and payload["stream"] is False
    assert payload["options"] == {
        "num_ctx": 8192,
        "num_predict": 256,
        "temperature": 0,
        "presence_penalty": 0,
        "seed": 42,
    }
    assert generation.text == "```sql\nSELECT 1\n```" and generation.prompt_tokens == 100


def test_server_reading_fewer_tokens_than_counted_is_an_error() -> None:
    llm = make_llm(FakeOllama(prompt_eval_count=4098), 15921, num_ctx=32768)

    with pytest.raises(PromptCountMismatchError):
        llm.generate(MESSAGES)


def test_prompt_that_cannot_fit_is_never_sent() -> None:
    server = FakeOllama(prompt_eval_count=8000)

    with pytest.raises(PromptTooLongError):
        make_llm(server, 8000, num_ctx=8192, max_new_tokens=256).generate(MESSAGES)
    assert server.chats == []


def test_a_thinking_block_is_an_error() -> None:
    with pytest.raises(RuntimeError, match="thinking"):
        make_llm(FakeOllama(prompt_eval_count=10, thinking="hmm"), 10).generate(MESSAGES)


def test_cache_serves_repeat_requests(tmp_path: Path) -> None:
    server = FakeOllama(prompt_eval_count=10)
    llm = make_llm(server, 10, cache=ResponseCache(tmp_path / "cache.sqlite"))

    first, second = llm.generate(MESSAGES), llm.generate(MESSAGES)

    assert len(server.chats) == 1
    assert not first.cached and second.cached and second.text == first.text


def test_a_reply_ollama_did_not_finish_is_retried() -> None:
    server = FakeOllama(prompt_eval_count=10, unfinished=2)

    generation = make_llm(server, 10, backoff_s=0).generate(MESSAGES)

    assert len(server.chats) == 3 and generation.prompt_tokens == 10


def test_unfinished_replies_that_persist_fail_loudly() -> None:
    server = FakeOllama(prompt_eval_count=10, unfinished=99)

    with pytest.raises(RuntimeError, match="unfinished reply"):
        make_llm(server, 10, retries=2, backoff_s=0).generate(MESSAGES)
    assert len(server.chats) == 3


def test_unknown_model_fails_at_construction() -> None:
    with pytest.raises(RuntimeError, match="not pulled"):
        OllamaLLM("qwen3.5:4b", lambda _: 1, transport=httpx.MockTransport(FakeOllama(1)))
