"""Phase 0 smoke test: one short chat request to the local 9B served by Ollama, thinking off.

Compares the native /api/chat endpoint (think=false, num_ctx via options) with the OpenAI-compatible
/v1 endpoint, and reports which context window each call actually loads the model with.

Usage: uv run python scripts/smoke_ollama.py [--model qwen3.5:9B]
"""

import argparse
import json
import subprocess
import time

import httpx
from openai import OpenAI

BASE_URL = "http://localhost:11434"
NUM_CTX = 8192
SYSTEM = (
    "You are an expert SQL assistant. Given a database schema and a question, write a single SQLite SELECT "
    "query that answers the question. Use only tables and columns from the schema. Output only the SQL in a "
    "```sql code block."
)
USER = (
    "### Database: concert_singer\n### Schema\n"
    "CREATE TABLE singer (\n  singer_id INTEGER PRIMARY KEY,\n  name TEXT,\n  country TEXT\n);\n"
    "### Evidence\n(none)\n### Question\nHow many singers are from France?"
)
MESSAGES = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}]


def gpu_used_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(out.stdout.strip())


def loaded_model(http: httpx.Client, model: str) -> dict | None:
    running = http.get("/api/ps").json()["models"]
    return next((m for m in running if m["name"].lower() == model.lower()), None)


def unload(http: httpx.Client, model: str) -> None:
    http.post("/api/generate", json={"model": model, "keep_alive": 0})
    time.sleep(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3.5:9B")
    args = parser.parse_args()
    report: dict = {"model": args.model}

    with httpx.Client(base_url=BASE_URL, timeout=300) as http:
        report["ollama_version"] = http.get("/api/version").json()["version"]
        unload(http, args.model)
        report["gpu_used_mib_before"] = gpu_used_mib()

        body = {
            "model": args.model,
            "messages": MESSAGES,
            "think": False,
            "stream": False,
            "options": {
                "num_ctx": NUM_CTX,
                "temperature": 0,
                "presence_penalty": 0,
                "num_predict": 50,
                "seed": 42,
            },
        }
        res = http.post("/api/chat", json=body).json()
        if "error" in res:
            raise SystemExit(f"/api/chat failed: {res['error']}")
        ps = loaded_model(http, args.model) or {}
        report["native"] = {
            "content": res["message"]["content"],
            "thinking_returned": bool(res["message"].get("thinking")),
            "prompt_tokens": res["prompt_eval_count"],
            "output_tokens": res["eval_count"],
            "gen_tok_per_s": round(res["eval_count"] / (res["eval_duration"] / 1e9), 1),
            "load_s": round(res["load_duration"] / 1e9, 1),
            "context_length": ps.get("context_length"),
            "size_gib": round(ps.get("size", 0) / 2**30, 2),
            "size_vram_gib": round(ps.get("size_vram", 0) / 2**30, 2),
            "gpu_used_mib": gpu_used_mib(),
        }

        client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="ollama")
        variants = {
            "v1_default": {},
            "v1_extra_body_think_false": {"extra_body": {"think": False}},
            "v1_reasoning_effort_none": {"reasoning_effort": "none"},
        }
        for name, extra in variants.items():
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=MESSAGES,
                    temperature=0,
                    presence_penalty=0,
                    max_tokens=50,
                    seed=42,
                    **extra,
                )
                msg = resp.choices[0].message
                extras = msg.model_extra or {}
                report[name] = {
                    "content": msg.content,
                    "reasoning_returned": bool(extras.get("reasoning") or extras.get("reasoning_content")),
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "context_length": (loaded_model(http, args.model) or {}).get("context_length"),
                }
            except Exception as exc:  # report every variant, even the ones the server rejects
                report[name] = {"error": f"{type(exc).__name__}: {exc}"}

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
