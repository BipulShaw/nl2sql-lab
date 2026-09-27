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
