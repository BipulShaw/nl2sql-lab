"""Probe how Ollama counts and cuts prompts, so the eval client's truncation guard rests on measured facts.

Checks three things on the native /api/chat endpoint (thinking off, num_ctx fixed):
1. Does `prompt_eval_count` report the whole prompt when Ollama reuses its KV cache for a repeated prefix?
2. What happens to a prompt longer than num_ctx: an error, or a silent cut, and what count is reported then?
3. Does the Hugging Face chat template (thinking off) count the same prompt tokens as Ollama? This is what
   lets the eval client detect truncation by comparing its own count with the server's.

Usage: uv run python scripts/probe_ollama_context.py [--model qwen3.5:9B] [--hf-model Qwen/Qwen3.5-9B]
       [--num-ctx 8192]
"""

import argparse
import json

import httpx
from transformers import AutoTokenizer

BASE_URL = "http://localhost:11434"


def chat(http: httpx.Client, tokenizer, model: str, user: str, num_ctx: int) -> dict:
    messages = [{"role": "user", "content": user}]
    body = {
        "model": model,
        "messages": messages,
        "think": False,
        "stream": False,
        "options": {"num_ctx": num_ctx, "temperature": 0, "num_predict": 8, "seed": 42},
    }
    res = http.post("/api/chat", json=body).json()
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return {
        "error": res.get("error"),
        "prompt_eval_count": res.get("prompt_eval_count"),
        "hf_template_count": len(tokenizer(text, add_special_tokens=False)["input_ids"]),
        "reply": (res.get("message") or {}).get("content"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3.5:9B")
    parser.add_argument("--hf-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--num-ctx", type=int, default=8192)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model)  # tokenizer files only (a few MB)

    # Numbered lines make the prompt long, cheap to generate, and let the reply show which lines survived.
    def numbered(n: int) -> str:
        return "\n".join(f"line {i}: the quick brown fox jumps over the lazy dog" for i in range(1, n + 1))

    ask = "\nWhat is the number of the FIRST line above? Answer with the number only."
    report: dict = {"model": args.model, "hf_model": args.hf_model, "num_ctx": args.num_ctx}
    with httpx.Client(base_url=BASE_URL, timeout=600) as http:
        http.post("/api/generate", json={"model": args.model, "keep_alive": 0})  # start from an empty cache

        def run(user: str) -> dict:
            return chat(http, tokenizer, args.model, user, args.num_ctx)

        short = numbered(100) + ask
        report["same_prompt_first_call"] = run(short)
        report["same_prompt_second_call"] = run(short)
        report["shared_prefix_new_suffix"] = run(short + " Be brief.")
        # ~14 tokens per line: 1,000 lines is well past an 8192-token window.
        report["over_length_prompt"] = run(numbered(1000) + ask)
        http.post("/api/generate", json={"model": args.model, "keep_alive": 0})  # free the GPU
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
