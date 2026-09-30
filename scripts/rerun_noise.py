"""Rerun noise (ADR-020): regenerate a config's latest recorded full-split run from scratch, with the response
cache off and nothing written to it. Score it, and compare it with the recorded run question by question. The
config and prompts are the same and decoding is greedy, so every difference comes from the server.

The rerun's manifest goes to --out (default /tmp/nl2sql-rerun), not results/runs, so the tables keep the
recorded runs.

Usage: uv run --extra ml python scripts/rerun_noise.py --config configs/base_9b_local.yaml --dataset bird-mini
"""

import argparse
import json
from pathlib import Path

import nl2sql.eval.harness as harness
from nl2sql.config import load_config
from nl2sql.eval.paired import compare_runs


def latest_recorded(config_name: str, dataset: str) -> Path:
    runs = []
    for manifest in harness.RUNS_DIR.glob(f"*_{dataset}-dev_{config_name}_n*/manifest.json"):
        m = json.loads(manifest.read_text(encoding="utf-8"))
        if m["n"] == m["n_split"] and m.get("latency_scope") == "pipeline":
            runs.append((m["started_at"], manifest.parent))
    if not runs:
        raise SystemExit(f"no full-split Phase 2 run of {config_name} on {dataset} dev")
    return max(runs)[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", type=Path, default=Path("/tmp/nl2sql-rerun"))
    args = parser.parse_args()
    config = load_config(args.config)
    original = latest_recorded(config.name, args.dataset)

    harness.RUNS_DIR = args.out
    manifest = harness.run_eval(args.dataset, "dev", config, use_cache=False)
    rerun = args.out / manifest["run_id"]

    def preds(run: Path) -> list[str | None]:  # in order: ids can repeat (BIRD mini-dev)
        lines = (run / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        return [r["pred"] for r in map(json.loads, lines)]

    before = json.loads((original / "manifest.json").read_text(encoding="utf-8"))
    r = compare_runs(original, rerun)
    a, b = preds(original), preds(rerun)
    print(f"recorded {original.name}: EX {before['metrics']['ex']:.4f}")
    print(f"rerun    {rerun.name}: EX {manifest['metrics']['ex']:.4f}")
    same = sum(x == y for x, y in zip(a, b, strict=True))
    print(
        f"identical SQL {same}/{len(a)}; correct in both {r['both']}, only recorded {r['only_a']}, "
        f"only rerun {r['only_b']}, neither {r['neither']}; McNemar p = {r['p_value']:.3f}"
    )


if __name__ == "__main__":
    main()
