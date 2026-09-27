"""Client-side token counts with the served model's own tokenizer and chat template."""

from typing import Any


class ChatTokenizer:
    """Counts what Ollama counts: the rendered chat template with thinking off, with no extra special tokens.
    Measured equal to Ollama's prompt_eval_count for the 9B (scripts/probe_ollama_context.py, ADR-012)."""

    def __init__(self, model_id: str, revision: str) -> None:
        try:
            from transformers import AutoTokenizer  # slow to import, and only in the `ml` extra
        except ImportError as exc:
            raise RuntimeError("token counting needs the ml extra: uv sync --extra ml") from exc
        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_id, revision=revision)

    def count(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def render_chat(self, messages: list[dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    def count_chat(self, messages: list[dict[str, str]]) -> int:
        return self.count(self.render_chat(messages))
