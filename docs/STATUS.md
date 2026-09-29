# Status

**Current phase:** Phase 0 (environment) is done. Next: Phase 1 (baseline 9B).
**Last updated:** 2026-09-27

---

## Environment (measured 2026-09-27)

| | |
|---|---|
| Host | Windows 11 Home 10.0.26200, 24 GB RAM |
| Linux | WSL2, kernel 6.6.87.2-microsoft-standard-WSL2, Ubuntu 24.04.2 LTS; distro on D: (122 GB free after Phase 0) |
| WSL limits | `memory=16GB` (15 GiB visible), `networkingMode=mirrored` (see `C:\Users\shawb\.wslconfig`) |
| GPU | RTX 4060 Laptop GPU, 8,188 MiB. Windows driver 616.92; CUDA 13.4 available to WSL. ~0.5–0.7 GiB used by the desktop at idle |
| Toolchain | uv 0.12.19, Python 3.11.16 (uv-managed), git 2.43.0, gcc 13.3.0 |
| GPU stack | torch 2.14.0+cu130, torchvision 0.29.0+cu130, triton 3.8.0, transformers 5.17.0, peft 0.21.0, trl 1.14.0, bitsandbytes 0.50.2, accelerate 1.15.0, liger-kernel 0.8.3, kernels 0.16.2 |
| Other | datasets 5.0.1, sentence-transformers 6.1.0, sqlglot 30.19.0, pydantic 2.13.5. Everything else is in `uv.lock` |
| Serving | Ollama 0.34.0 on Windows, reachable from WSL at `http://localhost:11434` |
| Disk used | `.venv` 6.2 GB; HF cache 9.3 GB (4B weights + ~10 MB of Hub kernels) |

Checks that passed:
- `torch.cuda.is_available()` and bf16 support.
- A bitsandbytes nf4 matmul on the GPU.
- `apply_liger_kernel_to_qwen3_5` imports.
- transformers has native `qwen3_5` modeling code.

## Models (details and licenses in DECISIONS ADR-002/003)

| Role | Model | Where |
|---|---|---|
| Zero-shot reference + demo | `Qwen/Qwen3.5-9B` (Apache-2.0) | Ollama `qwen3.5:9B`, Q4_K_M, 6.14 GiB, digest `6488c96fa5fa` |
| Fine-tune target | `Qwen/Qwen3.5-4B` (Apache-2.0) @ `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | HF cache in WSL, 9.32 GB |
| 9B fine-tune | dropped (ADR-005) | — |

## Smoke tests

All smoke tests use the same prompt, in PLAN §6.3 format: 114 tokens, thinking off, greedy decoding, at most 50
new tokens. Every configuration returned the same correct answer:
`SELECT COUNT(*) FROM singer WHERE country = 'France';`

**9B via Ollama** (`scripts/smoke_ollama.py`, native `/api/chat`)

| context_length | model in VRAM | GPU used, incl. desktop | cold load | decode |
|---|---|---|---|---|
| 8192 | 5.24 GiB (100% on GPU) | 7,202 of 8,188 MiB | 47.4 s | 6.9 tok/s* |

\* First call, 17 tokens. This is not a throughput figure; Phase 1 measures it on real eval prompts.

**4B in transformers** (`scripts/smoke_hf.py --quant <q> [--use-kernels]`). "Peak" is
`torch.cuda.max_memory_allocated()`. tok/s is measured on a warm second call for a 19-token answer, so it's a
rough figure.

| Config | Weights | Peak | Device used, incl. desktop | Decode | Notes |
|---|---|---|---|---|---|
| nf4 (double quant, bf16 compute) | 2.90 GiB | 3.00 GiB | 4.14 GiB | 10.5 tok/s | PyTorch fallback for DeltaNet |
| nf4 + Hub kernels | 2.90 GiB | 3.00 GiB | 4.14 GiB | 13.1 tok/s | ADR-010 |
| int8 + Hub kernels | 4.51 GiB | 4.59 GiB | 5.89 GiB | 4.1 tok/s | bitsandbytes int8 decode is slow |
| bf16 + Hub kernels | 7.85 GiB | 7.92 GiB | 8.00 GiB (full) | 5.0 tok/s | **Overflowed into system RAM, no OOM** (ADR-011) |
| bf16, capped at 0.85 of VRAM | — | — | — | — | `torch.OutOfMemoryError` at load, as intended |
| nf4 + kernels, capped at 0.85 | 2.90 GiB | 3.00 GiB | 4.14 GiB | 13.4 tok/s | The cap doesn't affect runs that fit |

## What Phase 0 found (each finding changes something later)
1. **Ollama `/v1` is unsafe for eval.** It reloads the model at a 4096-token context and ignores `think:false`, so
   the 9B uses the native API (ADR-007).
2. **The 9B and the 4B can't share the GPU.** With the 9B loaded, 7.2 of 8 GB is in use. Unload with
   `keep_alive: 0` before loading a transformers model.
3. **VRAM overflow is silent on this machine.** Every GPU entry point caps the allocator at 0.85 (ADR-011).
4. **The fast DeltaNet kernels need** `kernels` 0.16.x, `einops` and `use_kernels=True` (ADR-010).
5. **LoRA targets must include the DeltaNet projections**; the plan's list would skip 24 of 32 token-mixing layers
   (ADR-008).
6. **Where 4-bit memory goes:** 1.18 of the 2.9 GiB is the bf16 embedding table (LEARNING_LOG).

## Phase 0 checklist
- [x] Hardware, disk and model ids verified; models confirmed with Bipul
- [x] uv, Python 3.11, CUDA PyTorch and the core dependencies installed (`uv sync --extra ml`)
- [x] Smoke tests pass, with VRAM recorded (above)
- [x] Model ids and licenses in `DECISIONS.md`
- [x] `git init`, `.gitignore`, first commit
- [x] GitHub Actions workflow (`ruff` + `pytest` on CPU, `uv sync --locked` without the `ml` extra)
- [x] CI green on https://github.com/BipulShaw/nl2sql-lab
  ([runs](https://github.com/BipulShaw/nl2sql-lab/actions/workflows/ci.yml))

## Open problems / notes
- A run fails at startup, before any example, if Ollama isn't reachable or the model isn't pulled
  (`OllamaLLM.describe`).
- Hub downloads are unauthenticated (rate-limited). Setting `HF_TOKEN` is optional.
- Commits use a repo-local identity (a GitHub noreply address), not the machine's global git config.
- The DeltaNet kernels' effect on training throughput is unmeasured until Phase 3's 30-step check.
