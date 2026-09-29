# Status

**Current phase:** Phase 1 (baseline) is done, tagged `v0.1-baseline`. Next: Phase 2 (pipeline).
**Last updated:** 2026-09-29

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

---

## Phase 1 — baseline (2026-09-28 to 2026-09-29)

### Results: zero-shot on Spider dev, all 1,034 examples

Both models ran in Ollama at Q4_K_M with the same prompts and settings: the whole schema in the prompt, no
schema linking, no repair, thinking off, greedy decoding. Configs: `configs/base_9b_local.yaml` and
`configs/base_4b_local.yaml`. The generated table with every run is `results/RESULTS.md`.

| Model | EX | EX, any column order | Failed to execute | p50 latency |
|---|---|---|---|---|
| `qwen3.5:9B` | **74.3%** (768) | 79.2% (819) | 10 | 1,256 ms |
| `qwen3.5:4b` | **73.4%** (759) | 77.5% (801) | 21 | 810 ms |

The 0.9-point gap is not significant. The two models disagree on 119 examples: the 9B alone is right on 64 and
the 4B alone on 55. McNemar's exact test gives p = 0.46 (`nl2sql results compare`). On the 200-example pilots
(stratified, seed 42) the gap looked like 3 points (75.0% vs 72.0%, p = 0.26), and the full split didn't bear
that out.

Checks on both full runs:
- Every schema fit whole (`schema_levels: full`).
- No reply contained thinking, and every reply put its SQL in a code fence.
- One reply hit the 256-token limit: the 4B on Spider dev 766, where greedy decoding repeated one `OR` condition
  until it ran out of tokens (LEARNING_LOG).
- Ollama's prompt token count equalled the client's (ADR-012). The client counts with the pinned
  `Qwen/Qwen3.5-4B` tokenizer for both models, so the 9B's counts matching on every call confirms the two
  share a tokenizer and chat template.

### Where the points go

`scripts/failure_breakdown.py` re-runs every wrong prediction and sorts it into the first category that fits.
One example is about 0.1 EX points.

| Category | 9B | 4B | What it means |
|---|---|---|---|
| different_values | 67 | 78 | Same shape as the gold, different values |
| more_rows | 56 | 41 | Same columns, more rows than the gold |
| column_order | 51 | 42 | The right rows once the columns are reordered |
| fewer_rows | 36 | 49 | Same columns, fewer rows |
| more_columns | 25 | 15 | Extra columns |
| set_equal | 11 | 12 | The same distinct rows, duplicated differently |
| pred_error | 10 | 21 | Fails to execute |
| fewer_columns | 9 | 15 | Missing columns |
| row_order | 1 | 2 | The right rows in the wrong order, where the gold's `ORDER BY` makes order count |
| **Wrong** | **266** | **275** | |

- **column_order** is what Spider's official evaluator would accept and ours doesn't (ADR-013).
- **set_equal** and **row_order** are what BIRD's set comparison would accept. 7 of the 9B's 11 set_equal
  misses leave out a `DISTINCT` that the gold query has (Spider dev 75, 535, 536, 958, 959, 998, 999).
- **more_columns:** 10 of the 9B's 25 are "which X has the most Y?" questions where the model returns the
  count next to X (for example Spider dev 26, 142, 462, 878).
- **pred_error:** 16 of the 4B's 21 are `no such column`: the query names a column that the table it points at
  doesn't have. One more is the looping reply to Spider dev 766, cut off mid-query.
- Some misses are wrong gold. In the pilot, at least four gold queries fail on letter case, such as Spider dev
  554's `'timmothy'` against the data's 'Timmothy' (LEARNING_LOG). The gold is not patched.

### Data (ADR-014)
- Spider (train 8,659, dev 1,034), BIRD dev (1,534) and BIRD mini-dev (500) are downloaded, each archive checked
  against a pinned SHA-256.
- `nl2sql data verify` ran all 11,727 gold queries. Spider dev is fully valid. Spider train has 3 broken
  queries, and BIRD dev has 2 that time out (also in mini-dev).
- Scoring the gold SQL as the prediction gives EX 100.0% on Spider dev, so the harness scores correctly.

### What Phase 1 found (each finding changes something later)
1. **Ollama's prompt token count can't reveal truncation.** It reports the tokens it *kept*, so
   `prompt_tokens < num_ctx` passes exactly when the prompt was cut. The client now counts every prompt with the
   pinned tokenizer and requires Ollama's count to equal it (ADR-012).
2. **The EX definition is worth about 5 points.** Allowing any column order adds 4.9 points for the 9B. RESULTS
   reports both, and the numbers are not directly comparable with the leaderboards (ADR-013).
3. **Repair has little to fix on Spider.** The repair loop (Phase 2) reacts to guard failures, errors and
   timeouts. Only 10 of the 9B's 266 misses failed to execute, and 21 of the 4B's 275. Almost every miss runs
   and returns a wrong answer.
4. **On Spider dev the 4B base is not measurably behind the 9B** (73.4% vs 74.3%, p = 0.46). With 119
   disagreements, a gap of 23 examples (2.2 points) would be needed for p < 0.05. So Spider dev alone can't
   show whether fine-tuning closes the gap to the 9B. Phase 3's comparison rests on BIRD, and every model
   comparison reports the paired test (`nl2sql results compare`, LEARNING_LOG).
5. **Spider schemas fit whole; BIRD's don't.** At a 3,072-token schema budget every Spider dev schema fits at
   full detail, but 5 of BIRD's 11 dev databases lose samples or descriptions. At the 1,536-token training
   budget, BIRD needs schema linking (ADR-015).
6. **Gold SQL is sometimes wrong,** so a single EX number hides real disagreements. The failure breakdown shows
   where the points go.

### Phase 1 checklist
- [x] Spider and BIRD loaders, schema serializer with token budgets, prompt builder (ADR-015)
- [x] `OllamaLLM` on the native API: `num_ctx` and thinking off on every call, exact prompt token check (ADR-012)
- [x] SQLite executor with timeout and row cap; EX metric, strict and any column order (ADR-013)
- [x] Harness, run manifests and response cache; `nl2sql eval` and `nl2sql results table`
- [x] Pinned data archives and gold verification (ADR-014); gold passthrough at 100% on Spider dev
- [x] 9B zero-shot on Spider dev: `--limit 200`, then the full split
- [x] 4B base on Spider dev through the same server: `--limit 200`, then the full split
- [x] `nl2sql results compare`: paired comparison of two runs with McNemar's exact test
- [x] First rows in `results/RESULTS.md`
- [x] LEARNING_LOG: EX vs exact match, chat templates and thinking (Phase 0), greedy decoding, context windows
  and truncation, KV cache and VRAM
- [x] Tag `v0.1-baseline`

### Moved to later phases
- **BIRD runs** belong to Phase 2, with the BIRD ablations. The loader and gold verification are done.
- **The nf4 vs int8 check for the 4B** (ADR-004) moves to the start of Phase 3, when the transformers backend
  exists. Phase 1 ran the 4B through Ollama at Q4_K_M. Phase 3 re-runs the base 4B in transformers, so the base
  and the fine-tuned model are compared through the same backend and quantization.
- **BIRD train (8.9 GB)** is not downloaded. It's only needed for Phase 3's training data, and at over 5 GB it
  needs Bipul's approval first.

## Next: Phase 2 (pipeline)
- Schema linking with a link-recall metric (target: recall ≥ 95%).
- SQL guard (`sqlglot`) with its table of tests, the repair loop, and typed events.
- Ablations on Spider dev and BIRD mini-dev for the 9B and the 4B base: full schema vs linked, repair 0 vs 2.
- Checkpoint: tag `v0.2-pipeline`.

## Open problems / notes
- A run fails at startup, before any example, if Ollama isn't reachable or the model isn't pulled
  (`OllamaLLM.describe`).
- Hub downloads are unauthenticated (rate-limited). Setting `HF_TOKEN` is optional.
- Commits use a repo-local identity (a GitHub noreply address), not the machine's global git config.
- The DeltaNet kernels' effect on training throughput is unmeasured until Phase 3's 30-step check.
