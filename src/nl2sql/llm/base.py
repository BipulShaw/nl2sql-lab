"""What a generation backend returns."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Generation:
    text: str
    prompt_tokens: int
    output_tokens: int
    latency_ms: float  # of the original request, also when served from the cache
    finish_reason: str  # "stop", or "length" when max_new_tokens cut the reply off
    cached: bool = False
