"""Does Ollama give the same greedy reply to the same request? Regenerates a config's prompts for the first
N questions of a split with the response cache off, and compares them with the replies the cache recorded
(ADR-020).

Pass 1 sends each prompt once, in dataset order. Pass 2 sends each prompt twice back to back, so the second
copy reuses the server's cached prompt; the two copies are also compared with each other. Nothing is written
to the cache. Questions without a recorded reply are skipped.

Usage: uv run --extra ml python scripts/determinism_probe.py [--config configs/base_4b_local.yaml]
       [--dataset bird-mini] [--n 60]
"""

import argparse
import json
from pathlib import Path

from nl2sql.config import load_config
from nl2sql.data import load_examples
from nl2sql.eval.harness import RESPONSE_CACHE
from nl2sql.llm.base import Generation
from nl2sql.llm.cache import ResponseCache
from nl2sql.llm.ollama import OllamaLLM
from nl2sql.llm.tokens import ChatTokenizer
from nl2sql.prompting.builder import build_messages, extract_sql
from nl2sql.schema.introspect import load_schema
from nl2sql.schema.serialize import serialize_schema


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/base_4b_local.yaml"))
    parser.add_argument("--dataset", default="bird-mini")
    parser.add_argument("--n", type=int, default=60)
    args = parser.parse_args()
    config = load_config(args.config)
    if config.linking.enabled:
        raise SystemExit("full-schema configs only: the probe doesn't run the linker")
    tokenizer = ChatTokenizer(config.tokenizer.id, config.tokenizer.revision)

    def llm(cache: ResponseCache | None) -> OllamaLLM:
        return OllamaLLM(
            config.model_id,
            tokenizer.count_chat,
            base_url=config.base_url,
            num_ctx=config.context_window,
            max_new_tokens=config.decoding.max_new_tokens,
            seed=config.seed,
            cache=cache,
        )

    recorder, fresh = llm(ResponseCache(RESPONSE_CACHE)), llm(None)
    schemas = {}
    prompts = []
    for e in load_examples(args.dataset, "dev")[: args.n]:
        if e.db_path not in schemas:
            schema = load_schema(e.db_path, e.db_id)
            schemas[e.db_path] = serialize_schema(
                schema,
                tokenizer.count,
                config.prompt.schema_token_budget,
                include_samples=config.prompt.include_samples,
            )
        prompts.append((e.id, build_messages(e.db_id, schemas[e.db_path].text, e.question, e.evidence)))

    recorded: dict[str, Generation] = {}
    for qid, messages in prompts:
        if recorder.cache.get(recorder.request(messages)[1]) is not None:
            recorded[qid] = recorder.generate(messages)  # a cache hit: nothing is sent
    print(f"{len(recorded)} of {len(prompts)} prompts have a recorded reply")

    def compare(label: str, replies: dict[str, Generation]) -> None:
        same = sum(replies[q].text == recorded[q].text for q in replies)
        same_sql = sum(extract_sql(replies[q].text).sql == extract_sql(recorded[q].text).sql for q in replies)
        print(f"{label}: identical to the recorded reply {same}/{len(replies)}, identical SQL {same_sql}")
        for q in replies:
            if replies[q].text != recorded[q].text:
                print(f"  {q}: recorded {json.dumps(recorded[q].text)[:150]}")
                print(f"  {' ' * len(q)}  now      {json.dumps(replies[q].text)[:150]}")

    pass1 = {q: fresh.generate(m) for q, m in prompts if q in recorded}
    compare("pass 1, dataset order", pass1)
    first, second = {}, {}
    for q, m in prompts:
        if q in recorded:
            first[q], second[q] = fresh.generate(m), fresh.generate(m)
    compare("pass 2, first copy", first)
    compare("pass 2, second copy (prompt cached by the server)", second)
    same_copies = sum(first[q].text == second[q].text for q in first)
    print(f"pass 2 copies identical to each other: {same_copies}/{len(first)}")
    print(f"pass 1 identical to pass 2: {sum(pass1[q].text == first[q].text for q in pass1)}/{len(pass1)}")


if __name__ == "__main__":
    main()
