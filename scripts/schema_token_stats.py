"""How many tokens the schema and the whole prompt take on each dev split, to choose a config's token budgets.

For every database: schema tokens at each shedding level with no linker (every table kept). For every example:
prompt tokens plus max_new_tokens, with the schema serialized at each candidate budget (ADR-015).

Usage: uv run --extra ml python scripts/schema_token_stats.py [--config configs/base_9b_local.yaml]
       [--budgets 1536 3072]
"""

import argparse
import statistics
from collections import Counter
from pathlib import Path

from nl2sql.config import load_config
from nl2sql.data import load_examples
from nl2sql.llm.tokens import ChatTokenizer
from nl2sql.prompting.builder import build_messages
from nl2sql.schema.introspect import load_schema
from nl2sql.schema.serialize import FULL_SCHEMA_LEVELS, render, serialize_schema

SPLITS = [("spider", "dev"), ("bird", "dev"), ("bird-mini", "dev")]
LIMITS = (2048, 4096, 8192)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/base_9b_local.yaml"))
    parser.add_argument("--budgets", type=int, nargs="+", default=[1536, 3072])
    args = parser.parse_args()
    config = load_config(args.config)
    tokenizer = ChatTokenizer(config.tokenizer.id, config.tokenizer.revision)
    new_tokens = config.decoding.max_new_tokens

    for dataset, split in SPLITS:
        examples = load_examples(dataset, split)
        schemas = {e.db_path: load_schema(e.db_path, e.db_id) for e in examples}
        print(f"\n{dataset}-{split}: {len(schemas)} databases, {len(examples)} examples")
        for level in FULL_SCHEMA_LEVELS:
            counts = {
                path.parent.name: tokenizer.count(render(s, {t.name.lower() for t in s.tables}, level))
                for path, s in schemas.items()
            }
            largest = sorted(counts.items(), key=lambda kv: -kv[1])[:3]
            print(
                f"  schema at {level.name:<15} median {statistics.median(counts.values()):>6.0f}"
                f"  max {max(counts.values()):>6}  largest: {', '.join(f'{k} {v}' for k, v in largest)}"
            )
        for budget in args.budgets:
            serialized = {
                path: serialize_schema(
                    s, tokenizer.count, budget, include_samples=config.prompt.include_samples
                )
                for path, s in schemas.items()
            }
            totals = sorted(
                tokenizer.count_chat(
                    build_messages(e.db_id, serialized[e.db_path].text, e.question, e.evidence)
                )
                + new_tokens
                for e in examples
            )
            levels = Counter(s.level for s in serialized.values())
            over = ", ".join(f"over {limit}: {sum(t > limit for t in totals)}" for limit in LIMITS)
            print(
                f"  budget {budget}: databases by level {dict(levels)}\n"
                f"    prompt + {new_tokens} new tokens: median {statistics.median(totals):.0f}, "
                f"p95 {totals[int(0.95 * (len(totals) - 1))]}, max {totals[-1]}; {over}"
            )


if __name__ == "__main__":
    main()
