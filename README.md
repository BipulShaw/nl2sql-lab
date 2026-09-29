# nl2sql-lab

Text-to-SQL pipeline (schema linking, SQL guard, execution-repair loop), a QLoRA fine-tune of Qwen3.5-4B,
and an execution-accuracy evaluation harness on Spider and BIRD. Work in progress; see `docs/STATUS.md`
for where it stands and `docs/DECISIONS.md` for why it is built this way.

## Setup (WSL2 Ubuntu, NVIDIA GPU)

```bash
uv sync --extra ml   # CUDA 13.0 PyTorch + training stack
uv run pytest -q
```

Smoke tests (Phase 0; results in `docs/STATUS.md`):

```bash
uv run python scripts/smoke_ollama.py   # 9B via Ollama at localhost:11434, thinking off
uv run python scripts/smoke_hf.py --quant nf4 --use-kernels --mem-fraction 0.85   # 4B in transformers
```

## Data and evaluation

```bash
scripts/download_data.sh                  # Spider 1.0, BIRD dev and mini-dev; every archive SHA-256-checked
uv run nl2sql data verify                 # runs every gold query; lists any that cannot be scored
uv run nl2sql eval --dataset spider --split dev --gold          # sanity check: gold scored as the prediction
uv run --extra ml nl2sql eval --dataset spider --split dev --config configs/base_9b_local.yaml [--limit 200]
uv run nl2sql results table               # regenerate results/RESULTS.md from the run manifests
uv run python scripts/failure_breakdown.py results/runs/<run_id>   # sort a run's misses by kind
uv run nl2sql results compare results/runs/<run_a> results/runs/<run_b>   # paired comparison, McNemar's p
```

Each run writes `results/runs/<run_id>/manifest.json`: the config, the model digest, the git commit, the
metrics and counts. Model replies are cached, so rerunning a run re-scores it without calling the model again.

**About the numbers.** EX here is strict execution accuracy. Rows are compared as multisets, and as ordered
lists when the gold query has a top-level `ORDER BY`; columns must come in the gold's order. That is not
exactly what either official evaluator computes. BIRD's compares sets of rows. Spider's test-suite evaluator
accepts any column order, so we also report EX with any column order next to the strict number. Details and
the reasons are in `docs/DECISIONS.md` (ADR-013). Results: `results/RESULTS.md`.

## Learning notes

This project is also a study log. `docs/LEARNING_LOG.md` explains each LLM concept as it comes up, with
numbers measured on this machine. `docs/DECISIONS.md` records why things are built the way they are, and
`docs/INTERVIEW_QA.md` collects questions about the project with answers.
