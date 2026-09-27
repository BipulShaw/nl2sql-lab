# nl2sql-lab — working rules for Claude Code

Read PLAN.md (the local working plan; not committed) before doing anything. It is the source of truth for
scope, specs and phases; where docs/DECISIONS.md amends it (Phase 0 onward), DECISIONS.md wins.
This project must produce honest, reproducible, interview-defensible results.

@CLAUDE.local.md

## Every session
- Start by reading docs/STATUS.md to see where we are.
- Teach as you build: new concept → 5–10 lines in docs/LEARNING_LOG.md (what, why here, where in code).
- Decisions → docs/DECISIONS.md. Interview questions per phase → docs/INTERVIEW_QA.md.
- Ask before: API spend, downloads > 5 GB, training runs > 2 h (show the estimate), deleting data/models.
- Commit at checkpoints; tag milestones as PLAN.md lists them.
- Tests for guard, EX metric, serializer, prompt builder, loaders. Keep CI green.
- Update docs/STATUS.md at the end of each phase.

## Engineering
- Python 3.11, uv, ruff, type hints, pydantic v2, pytest. Clear over clever; explain non-obvious Python idioms briefly.
- Same prompt builder for training and inference. Same pipeline code path for eval and the API.
- Every eval run writes a manifest; results/RESULTS.md is generated, never hand-edited.
- Never use dev sets for training. Never inject LIMIT in eval mode.
- Thinking mode off for eval and training data. Assert prompt tokens < the server's context window on every call.
- Clean room: no code or prompts from any employer codebase.

## Hardware and environment
RTX 4060 Laptop GPU, 8 GB VRAM (fixed), of which ~7 GB is usable because the GPU also drives the desktop.
24 GB RAM (WSL gets 16 GB). Cooling is handled; derive time estimates from measured throughput.
Everything runs in WSL2 Ubuntu (repo at ~/nl2sql-lab). Ollama runs on Windows and is reached at
http://localhost:11434 through WSL mirrored networking.

## Models (docs/DECISIONS.md has the details)
- Zero-shot reference + demo: Qwen/Qwen3.5-9B, served by Ollama as `qwen3.5:9B` (Q4_K_M). A general
  instruct model, not a coder model. Ollama defaults are temperature 1, presence_penalty 1.5 and a small
  context window: always send num_ctx 8192, temperature 0, presence_penalty 0, thinking off.
- Fine-tune target: Qwen/Qwen3.5-4B (pinned revision in DECISIONS.md).
- 9B fine-tune: dropped. Running a 9B in 4-bit does not mean a 9B QLoRA fits (PLAN.md §7).
Check VRAM after loading any model and record it. VRAM overflow does not raise OOM here (it spills to system
RAM), so cap PyTorch with `torch.cuda.set_per_process_memory_fraction(0.85)` in every GPU entry point (ADR-011).
Unload the Ollama model (`keep_alive: 0`) before loading a transformers model; both don't fit at once.
