# Decisions

Short ADRs: context → decision → alternatives → consequence. Newest last. PLAN.md (cited as "PLAN §n") is the
author's private working plan and isn't in this repo; each ADR states the context it needs. Where an ADR changes
the plan, the ADR wins and says so.

---

## ADR-001 — Python side runs in WSL2 (Ubuntu 24.04), distro on D:, mirrored networking, 16 GB RAM
*2026-09-27, Phase 0*

**Context.**
- The host is Windows 11. Two kernels the plan depends on are written in Triton, which officially supports only
  Linux: Liger's fused cross-entropy (needed to fit training in VRAM) and flash-linear-attention (fast Gated
  DeltaNet layers in Qwen3.5).
- C: was 80% full. WSL's default memory cap (~50% of RAM) is too small to merge a 4B adapter on CPU.
- WSL's default NAT networking can't reach Ollama, which listens on Windows 127.0.0.1:11434.

**Decision.**
- The Ubuntu 24.04 distro was moved to D: (`wsl --manage Ubuntu --move`).
- `C:\Users\shawb\.wslconfig` sets `memory=16GB` and `networkingMode=mirrored`.
- `build-essential` is installed, because Triton compiles a small C launcher on first use.
- The repo lives at `~/nl2sql-lab` on the WSL filesystem, not under `/mnt/c` or OneDrive.
- Ollama stays on Windows.

**Alternatives.**
- Native Windows Python with uv. Rejected: no Triton, so no fused CE and slow DeltaNet layers.
- NAT networking plus `OLLAMA_HOST=0.0.0.0`. Rejected: this exposes Ollama on the LAN.
- Docker Desktop: more moving parts for no gain here.

**Consequence.**
- Verified: WSL reaches Ollama at `http://localhost:11434`, and WSL sees 15 GiB RAM.
- The WSL virtual disk grows on demand up to 1 TB. The real limit is free space on D: (~142 GB at Phase 0).
- Edit the repo through VS Code "Remote – WSL".
- The whole Windows side of the setup is one small `.wslconfig` file.

## ADR-002 — Zero-shot reference and demo model: `Qwen/Qwen3.5-9B`, served by Ollama as `qwen3.5:9B` (Q4_K_M)
*2026-09-27, Phase 0*

**Context.** PLAN §1 asks for the exact local tag and its Hugging Face id. What Ollama reports:
- Tag `qwen3.5:9B`; digest `6488c96fa5fa`.
- Architecture `qwen35`, 9.7B parameters, Q4_K_M, 6.14 GiB file.
- Capabilities: completion, vision, tools, thinking.
- Native context length 262,144.

**Decision.**
- The zero-shot and demo model is `Qwen/Qwen3.5-9B`, as served by that tag. License: Apache-2.0.
- PLAN.md calls it a "9B coder". It is a general instruct model (multimodal, with a thinking mode), not a coder
  variant. The README and resume bullet will say "9B instruct model".
- Every run manifest records the Ollama tag, digest and quantization, because a tag can be re-pointed.

**Alternatives.** The other local models are `qwen3.8:27b` and `qwen3.8-small`: the same 27.3B Q4_K_M blob,
16.5 GiB. Neither fits in VRAM; CPU offload would make eval far too slow.

**Consequence.** Thinking is on by default and must be disabled per request (ADR-007).

## ADR-003 — Fine-tune target: `Qwen/Qwen3.5-4B` at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
*2026-09-27, Phase 0*

**Context.** PLAN §1 prefers a 3–4B model from the same family as the 9B.

**Decision.**
- The target is `Qwen/Qwen3.5-4B`, pinned to commit `851bf6e` (repo last modified 2026-03-02; 9.34 GB download).
- It shares the tokenizer, chat template and architecture with the 9B. The headline comparison then varies model
  size and fine-tuning, and nothing else.
- License: Apache-2.0, so publishing a LoRA adapter is allowed.
- Support: transformers 5.17 has native `qwen3_5` modeling code, and Liger ships
  `apply_liger_kernel_to_qwen3_5`.

**Alternatives.**
- `Qwen3-4B-Instruct-2507`: a different family, which would confound the size-gap story.
- `Qwen2.5-Coder-3B-Instruct`: older, and a coder specialization with a different template.

**Consequence.**
- The checkpoint is multimodal. We load it text-only, and LoRA targets must avoid the vision tower (ADR-008).
- Pinning the revision means a silent upstream update can't change results.

## ADR-004 — Check nf4 vs int8 base quality before any QLoRA training; fallback is Qwen3.5-2B with bf16 LoRA
*2026-09-27, Phase 0*

**Context.** Unsloth's Qwen3.5 guide advises against QLoRA (4-bit) training for this family, citing larger
quantization differences than usual. bf16 LoRA on the 4B needs about 10 GB, which doesn't fit in ~7 GB usable.

**Decision.**
- In Phase 1, evaluate the base 4B zero-shot on BIRD mini-dev in both nf4 and int8, with identical prompts and
  greedy decoding.
- Proposed rule, to confirm when the numbers are in:
  - If nf4 is within ~2 EX points of int8, train QLoRA in nf4 as planned.
  - If the gap is larger, fall back to `Qwen/Qwen3.5-2B` with plain bf16 LoRA.

**Alternatives.** Train nf4 regardless (risks an adapter that learns to compensate for quantization error). An
8-bit LoRA on the 4B (likely too tight with 248k-vocab logits).

**Consequence.** Phase 1 costs one extra mini-dev run. The Phase 3 setup is decided by data, not by assumption.

*Update 2026-09-29 (Phase 1):* nf4 and int8 are bitsandbytes formats, so the check needs the transformers
backend (`HFLocalLLM`), which the plan builds in Phase 3. The check moves to the start of Phase 3, before any
training. Phase 1 evaluates the base 4B through Ollama (Q4_K_M) instead.

## ADR-005 — Drop the 9B fine-tune stretch goal
*2026-09-27, Phase 0; amends PLAN §1 role 3 and §7*

**Context.**
- Usable VRAM is ~7 GiB (the desktop takes the rest).
- The 9B's 4-bit weights alone are ~5.5–6 GiB.
- Logits at seq 1536 with a 248,320-token vocab are 1.42 GiB in fp32, before their gradient
  (LEARNING_LOG, "Why you can run a 9B…").
- The ADR-004 concern about 4-bit training applies here too.

**Decision.** No 9B fine-tune, and no `configs/lora_9b.yaml`. Unsloth is not installed.

**Consequence.** The headline experiment is unchanged: fine-tuned 4B vs zero-shot 9B. The interview answer to
"why not fine-tune the 9B?" is this ADR's arithmetic.

## ADR-006 — PyTorch 2.14 with CUDA 13.0 wheels (`cu130`)
*2026-09-27, Phase 0; amends PLAN §4 step 3 ("CUDA 12.x")*

**Context.**
- Windows driver 616.92 exposes CUDA 13.4 to WSL, so any wheel up to cu13x runs.
- For torch 2.14.0 the PyTorch index publishes `cu126`, `cu130` and `cu132`.

**Decision.**
- Use `cu130` through an explicit uv index scoped to the `ml` extra (`pyproject.toml`).
- It is a CUDA 13 build that the driver supports with margin, and more established than `cu132`.
- Verified in Phase 0: bitsandbytes nf4 matmul runs on it (see STATUS).

**Alternatives.** `cu126`, the plan's 12.x suggestion; also valid. `cu132`.

**Consequence.** Switching is a one-line index change plus `uv lock`. Do it if any dependency ships no CUDA 13
build.

## ADR-007 — Ollama request settings for eval
*2026-09-27, Phase 0*

**Context.**
- Ollama's default `num_ctx` is set from available VRAM and is far below what a full BIRD schema prompt can need.
  Longer prompts are **silently truncated**.
- The model's shipped defaults are `temperature 1, top_p 0.95, top_k 20, presence_penalty 1.5`, and thinking on.

**Decision.** Every eval request sends:
- `num_ctx 8192`
- `temperature 0`
- `presence_penalty 0`
- `seed 42`
- thinking off

The client asserts `prompt_tokens < num_ctx` on every response. *(Superseded by ADR-012: a truncated prompt
reports a count below `num_ctx`, so this check can never fail. The client now requires an exact match.)*

These go through Ollama's **native `/api/chat`**, not the OpenAI-compatible `/v1`. Measured by
`scripts/smoke_ollama.py` on Ollama 0.34.0:
- Native `/api/chat` with `think:false` and `options`: correct SQL, no thinking. `/api/ps` reports
  `context_length 8192`.
- `/v1/chat/completions` ignores `num_ctx`. The next `/v1` call **reloaded the model at context_length 4096**.
- `/v1` also ignores `think:false` in `extra_body`: all 50 tokens went to reasoning and the answer was empty.
  Only `reasoning_effort: "none"` turned thinking off.

**Alternatives.** A derived Modelfile tag (`PARAMETER num_ctx 8192`, `temperature 0`, `presence_penalty 0`),
used via `/v1` with `reasoning_effort: "none"`. Rejected: those settings would live in server state outside the
repo, and a run manifest couldn't show they were applied. With the native API every setting is in the request
and gets logged.

**Consequence.**
- The local 9B gets an `OllamaLLM` backend (native API).
- PLAN §6.4's `OpenAICompatLLM` stays for API baselines and llama.cpp. Both implement the same `LLM` interface.
- Never mix `/v1` and native calls in one run: each `num_ctx` change reloads the model (a cold load took 47 s).
- With the 9B loaded at 8192 context the GPU is at 7.2 of 8.0 GB, so the harness must unload it
  (`keep_alive: 0`) before loading any transformers model.

## ADR-008 — LoRA target modules for Qwen3.5's hybrid layers
*2026-09-27, Phase 0; amends PLAN §7 table ("q,k,v,o,gate,up,down proj")*

**Context.**
- Only 1 layer in 4 of Qwen3.5 is standard attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`). The other three are
  Gated DeltaNet, with their own projections.
- The plan's list would leave 75% of the token-mixing layers untouched.
- The checkpoint also contains vision-tower modules that a loose suffix match would catch.

**Evidence.** Read from the loaded model by `scripts/smoke_hf.py` and from the checkpoint headers:
- `AutoModelForCausalLM` loads the text-only `Qwen3_5ForCausalLM`. It has no vision modules; `model.visual.*` and
  `mtp.*` are ignored.
- Linear layers by name:

  | Module | Count | Shape |
  |---|---|---|
  | `q_proj` | 8 | 8192×2560 |
  | `k_proj`, `v_proj` | 8 each | 1024×2560 |
  | `o_proj` | 8 | 2560×4096 |
  | `in_proj_qkv` | 24 | 8192×2560 |
  | `in_proj_z` | 24 | 4096×2560 |
  | `out_proj` | 24 | 2560×4096 |
  | `in_proj_a`, `in_proj_b` | 24 each | **32×2560** |
  | `gate_proj`, `up_proj` | 32 each | 9216×2560 |
  | `down_proj` | 32 | 2560×9216 |
  | `lm_head` | 1 | tied to the embedding |

**Decision.** `target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
"gate_proj", "up_proj", "down_proj"]`, as exact names.

At r=16 that is **30,474,240 trainable parameters**, 0.72% of the 4.21B language-model parameters. The plan's list
would give 21,233,664 and skip every DeltaNet layer. PEFT's `print_trainable_parameters()` must confirm the count
in Phase 3.

Excluded:
- `in_proj_a` and `in_proj_b` output one number per value head (32): the decay-gate input and the write strength
  β. Rank 16 is half their full rank, so the update would no longer be low-rank. They also steer the recurrence's
  memory decay, which is the numerically sensitive part. This is a judgment call, not a measurement; an ablation is
  possible if time allows.
- `conv1d` is not a Linear layer.
- `lm_head` is tied to the 248,320 × 2560 embedding (636M parameters). Training it would also move the input
  embeddings.

**Consequence.** `configs/lora_4b.yaml` uses this list. If a future checkpoint renames modules, PEFT raises an error
for a target that matches nothing, so a silent mismatch can't happen.

## ADR-009 — Packaging: one uv project, GPU stack in an `ml` extra, Linux-only resolution
*2026-09-27, Phase 0*

**Context.** CI runs `ruff` and `pytest` on CPU GitHub runners and must not download ~4 GB of CUDA wheels. The
plan targets Linux only.

**Decision.**
- `pyproject.toml` layout:
  - Core deps under `[project.dependencies]`.
  - torch, transformers, peft, trl, bitsandbytes, liger-kernel, kernels, sentence-transformers and friends under
    the `ml` extra.
  - Dev tools under the `dev` dependency group.
  - `[tool.uv] environments = ["sys_platform == 'linux'"]`.
  - Build backend `uv_build`, console script `nl2sql`.
- CI runs `uv sync --locked` without the extra; the GPU machine runs `uv sync --extra ml`.
- Python pinned to 3.11 (`.python-version`).

**Consequence.**
- One lockfile covers both.
- Anything importing torch or transformers must be imported lazily, or live in modules that CPU tests don't
  import.

## ADR-010 — Fast Gated DeltaNet kernels come from the Hub (`use_kernels=True`), not from source builds
*2026-09-27, Phase 0*

**Context.**
- Without extra packages, transformers runs Qwen3.5's DeltaNet layers on reference PyTorch code and logs
  "falling back… much slower" warnings for four functions: `chunk_gated_delta_rule`,
  `fused_recurrent_gated_delta_rule`, `causal_conv1d_fn` and `causal_conv1d_update`.
- The `causal-conv1d` pip package compiles CUDA C++ at install time. That needs the CUDA toolkit (`nvcc`) in WSL,
  which we don't have.

**Decision.**
- Load models with `from_pretrained(..., use_kernels=True)`. transformers then pulls prebuilt kernels from the
  Hugging Face Hub via the `kernels` package. The HF cache shows three were fetched: `kernels-community/fla`
  (flash-linear-attention), `kernels-community/mamba-ssm` (major version 2) and `kernels-community/activation`,
  about 10 MB in total.
- This needs `kernels>=0.16,<0.17`, because transformers 5.17 rejects 0.17. It also needs `einops`, which the
  `fla` kernel imports.

**Evidence** (`scripts/smoke_hf.py`, nf4, the 19-token answer timed on a warm second call):
- The fallback warnings disappear.
- Decode goes from **10.5 to 13.1 tok/s**.
- Output is identical.
- The main payoff is expected on long sequences (prefill, training), which Phase 3's throughput check will measure.

**Alternatives.**
- `pip install flash-linear-attention` (pure Triton) plus building `causal-conv1d` from source: needs the CUDA
  toolkit, which is about 3 GB more.
- Accept the fallback.

**Consequence.**
- The first load of each kernel downloads it from the Hub; it's cached after that.
- The run manifest records `use_kernels` alongside the package versions.
- `RMSNormZeroCentered` has no Hub kernel and stays on its PyTorch implementation (harmless warning).

## ADR-011 — Cap PyTorch's VRAM so overflow fails loudly instead of spilling to system RAM
*2026-09-27, Phase 0*

**Context.**
- Loading the 4B in bf16 should not fit: 7.85 GiB of weights, on a card with ~7.3 GiB free next to the desktop.
  It loaded anyway (peak 7.92 GiB allocated; device reported 8.0/8.0 GiB used).
- Decode dropped to 5.0 tok/s, slower than nf4's 13.1.
- The Windows driver under WSL evidently oversubscribes VRAM by paging to system memory instead of raising OOM.
- For Phase 3 that is dangerous. A training config that doesn't fit would run 2–3× slower instead of crashing, and
  quietly inflate the overnight estimate.

**Decision.**
- Every GPU entry point (eval with `HFLocalLLM`, training, merge) calls
  `torch.cuda.set_per_process_memory_fraction(0.85)` before loading a model. That's a 6.8 GiB cap for PyTorch's
  allocator.
- The Phase 3 throughput check also reports `max_memory_allocated`.

**Evidence.**
- With the cap, bf16 raises `torch.OutOfMemoryError` at load time. transformers pre-allocates the model's full
  size, so it fails before touching any weights.
- nf4 runs unchanged: 2.9 GiB weights, 13.4 tok/s.

**Alternatives.** NVIDIA's "CUDA – Sysmem Fallback Policy" setting in the Windows control panel. Rejected: it's
invisible to the repo, and I haven't confirmed it applies to WSL processes.

**Consequence.**
- The plan's "if you hit OOM" fallbacks (PLAN §7) now actually trigger.
- The fraction lives in config, so a different card can change it.
- CUDA context and non-PyTorch allocations sit outside the cap, which is why it's 0.85 and not 0.9.

## ADR-012 — Detect prompt truncation with an exact token count, not `prompt_tokens < num_ctx`
*2026-09-28, Phase 1; amends ADR-007*

**Context.** `scripts/probe_ollama_context.py` measured how Ollama 0.34.0 counts and cuts prompts (`qwen3.5:9B`,
native `/api/chat`, thinking off, `num_ctx` 8192):

| Prompt | Ollama `prompt_eval_count` | Our count (HF tokenizer + chat template) |
|---|---|---|
| 100 numbered lines, first call | 1,520 | 1,520 |
| Same prompt again (KV cache reused) | 1,520 | 1,520 |
| Same prefix, new suffix | 1,523 | 1,523 |
| 1,000 numbered lines | **4,098** | **15,921** |

- The over-length prompt raised no error. The model was asked for the number of the first line and answered
  "747" (the truth is 1): it never saw the start of the prompt.
- The reported count, 4,098, is below `num_ctx`. So ADR-007's check `prompt_tokens < num_ctx` passes exactly when
  truncation has happened. It can never fire.
- Encouraging: the count is exact otherwise, including when Ollama reuses cached prefix tokens.

**Decision.**
- The client counts every prompt itself, with the served model's tokenizer and chat template (thinking off, no
  extra special tokens). The 9B's `tokenizer.json` and `chat_template.jinja` are byte-identical to the 4B's, so
  the pinned `Qwen/Qwen3.5-4B` revision counts for both (`src/nl2sql/llm/tokens.py`).
- **Before sending:** if prompt + `max_new_tokens` exceeds `num_ctx`, raise `PromptTooLongError` and send
  nothing. The harness first applies the config's tighter `prompt.max_total_tokens`; such an example is recorded
  as `prompt_too_long` and counts as wrong.
- **After the response:** `prompt_eval_count` must equal the client's count, or `PromptCountMismatchError`
  stops the run. This catches truncation, and also a re-pointed model tag whose template differs from our
  tokenizer's.

**Alternatives.**
- Keep the `<` check: it cannot detect truncation (above).
- Set a much larger `num_ctx`: costs VRAM and still detects nothing.

**Consequence.**
- A run that finishes has had every prompt read in full. There is no silent-truncation state.
- Each config names a tokenizer and revision that must match the served model. For a model outside the Qwen3.5
  family, the first call fails loudly until the config names the right tokenizer.

## ADR-013 — What EX means here, and how it differs from the official evaluators
*2026-09-28, Phase 1; amends PLAN §6.6 (row cap) and the wording of PLAN §6.9*

**Context.**
- The plan's metric: execute gold and predicted SQL, and compare rows as multisets. Compare them as ordered lists
  when the gold has a top-level `ORDER BY`. An error or timeout is wrong. The plan calls this "BIRD-style strict".
- The official scripts do something else:
  - **BIRD** (`evaluation.py`) compares `set(predicted) == set(gold)`. Duplicates and row order never count,
    even under `ORDER BY`.
  - **Spider's test-suite evaluator** (`exec_eval.py`) accepts the predicted columns in any order. It compares
    row order whenever the gold contains "order by" anywhere, including in a subquery. By default it removes
    `DISTINCT` from both queries before running them. Its timeout is 60 s.
- The plan's executor capped fetched rows at 10,000. Two truncated results can look equal when they aren't. At
  that cap, 27 distinct gold queries would have been cut:
  - 6 in Spider train, 2 in Spider dev (455, 456);
  - 19 in BIRD dev. Three of these (340, 346, 397) are also in mini-dev, which reuses BIRD dev's question ids.

**Decision.**
- **EX** (headline):
  - Row tuples are compared as a multiset (`collections.Counter`), or as an ordered list if the gold's outermost
    query has an `ORDER BY`. This is decided with sqlglot, with a scan outside parentheses and quotes for the few
    gold queries sqlglot cannot parse.
  - Columns must come in the gold's order.
  - `DISTINCT` is kept.
  - Values compare as Python values, so `1 == 1.0`.
- **Row cap:**
  - The gold result is capped at 1,000,000 rows. The largest gold result in any split is 278,230 rows (BIRD dev).
  - The prediction is fetched only up to the gold's row count + 1. More rows than the gold is wrong, and a
    runaway query never loads.
- **Unscoreable gold:** a gold query that fails, times out (30 s) or exceeds the cap makes its example wrong in
  EX. **EX (valid gold)** leaves those examples out. Both are reported.
- **EX (any col. order)** is reported as a secondary number. The prediction may reorder its columns, one
  reordering applied to every row, as Spider's evaluator allows. It shows what the strict column rule costs.
- The plan's "BIRD-style" label is dropped. BIRD's rule is set equality, which is more lenient. The README and
  `RESULTS.md` define EX explicitly.
- Running the official scripts on the final predictions, for leaderboard parity, is optional in Phase 4.

**Alternatives.**
- Set equality (BIRD's rule): it would score a query that returns duplicate rows where the question wants
  distinct ones as correct.
- Any column order as the headline: defensible, since column order is usually arbitrary. The plan specified the
  strict rule, though, and reporting both keeps the difference visible.

**Consequence.**
- On the 9B pilot (200 Spider dev examples), 10 of 50 misses were column order alone: 75.0% strict, 80.0% with
  any column order (`scripts/failure_breakdown.py`).
- Our numbers are not directly comparable with published leaderboard numbers, and the docs say so.

## ADR-014 — Benchmark data: pinned archives and verified gold SQL
*2026-09-28, Phase 1*

**Context.**
- Spider 1.0 comes from a Google Drive link on its website. BIRD comes from its official bucket.
- A changed upstream file would silently change the benchmark.
- Gold SQL is not guaranteed to run.

**Decision.**
- `scripts/download_data.sh` checks every archive against a pinned SHA-256. A mismatch stops the script.

  | Archive | SHA-256 |
  |---|---|
  | `spider_data.zip` | `00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b` |
  | BIRD `dev.zip` (unpacks to `dev_20240627/`) | `cdd6d19faeb45a23970b98d3ef6c40a87987c95459c2cf12076897a60cf5a630` |
  | BIRD `minidev.zip` (SQLite version used) | `cc48ba16838204e4e214512030cb572eeb5f7bcdd999bae4b9b6ff12ec13b92f` |

- BIRD train (8.9 GB) is opt-in (`--with-bird-train`) and not yet downloaded. It is only needed for Phase 3
  training data.
- `nl2sql data verify` runs every gold query with EX's own timeout and row cap. It writes the ones that cannot be
  scored to `data/gold_failures.json`.

**Evidence** (`nl2sql data verify`, 74 s):

| Split | Examples | Databases | Gold errors | of which timeouts | Over the 1M cap | Largest gold result |
|---|---|---|---|---|---|---|
| Spider train | 8,659 | 146 | 3 | 0 | 0 | 18,228 rows |
| Spider dev | 1,034 | 20 | 0 | 0 | 0 | 20,662 rows |
| BIRD dev | 1,534 | 11 | 2 | 2 | 0 | 278,230 rows |
| BIRD mini-dev | 500 | 11 | 2 | 2 | 0 | 29,936 rows |

- Spider train 3153 references a table that does not exist (`Ref_Company_Types`). Spider train 4513 and 4514 put
  `ORDER BY` before `INTERSECT`, which SQLite rejects.
- BIRD dev 518 took about 50 s when run alone, and 701 about 300 s. Both are over the 30 s timeout. The same two
  questions are in mini-dev.

**Consequence.**
- Best possible EX: 100% on Spider dev, 1532/1534 on BIRD dev, 498/500 on mini-dev.
- The three broken Spider train queries must be left out of Phase 3's training data.
- The harness sanity check (gold SQL scored as the prediction) gives EX 100.0% on Spider dev.

## ADR-015 — Schema serialization and token budgets
*2026-09-28, Phase 1*

**Context.**
- The plan's prompt shows the schema as DDL-style text within a token budget. The default is 1,536 schema tokens
  in 2,048 total, the training length. A full-schema, no-linking run may use up to 3,072.
- `scripts/schema_token_stats.py` measured the schemas with the pinned tokenizer.
  - Spider dev: median 517 tokens at full detail, and the largest is 1,904 (`dog_kennels`).
  - BIRD dev: median 2,691, and the largest is 7,612.

**Decision.**
- **Format:**
  - `CREATE TABLE` text with primary keys inline, or as a separate line for a composite key.
  - Foreign keys as `-- FK: a.x -> b.y` lines.
  - An aligned `--` comment per column, `description | e.g. <samples>`.
  - Sample values are written as SQL literals, so the model sees how to quote them and their exact case and
    date format.
  - Identifiers are backtick-quoted only when they need it (reserved words, spaces).
- **Samples:** 3 distinct non-null values from the first 1,000 rows, each at most 30 characters.
- **Descriptions** (BIRD's CSV files) are capped at 200 characters, and the boilerplate "commonsense evidence:" is
  removed. Some value descriptions run past 800 characters, while the 95th percentile is ~170.
- **Shedding order when over budget:** `full` → `no_samples` → `no_descriptions`. With a schema linker (Phase 2),
  two more levels follow: `keys_only_unselected` → `selected_only`. The first level that fits is used, and the
  manifest records it for every example.
- **9B zero-shot config:** schema budget 3,072 and 4,096 tokens in total (the plan's full-schema setting).

**Evidence.** Prompt tokens plus 256 new tokens, at schema budget 3,072:
- **Spider dev:** every database fits at full detail. Median total 972, 95th percentile 2,258, max 2,279.
- **BIRD dev:**
  - Databases: 6 at full detail, 3 without samples, 2 without descriptions.
  - Max total 3,132, so nothing exceeds 4,096.
- **BIRD dev at budget 6,144:** 409 examples would exceed 4,096 in total.

**Consequence.**
- The Spider baseline sees every schema complete.
- On BIRD, 5 of 11 databases lose samples or descriptions at this budget. At the 1,536 training budget, BIRD needs
  schema linking.

## ADR-016 — The SQL guard checks names against the whole database, and resolves each column once
*2026-09-29, Phase 2*

**Context.**
- The guard (PLAN §6.5) runs before execution. It allows exactly one read-only query and blocks one that names a
  table or column that doesn't exist. A block's message goes into the repair prompt.
- With linking, the prompt shows only some tables. The guard could check names against the prompt or against the
  database.
- sqlglot's scope analysis lists a subquery's unqualified columns under the query around it as well, since they
  might be correlated references. The first version resolved each column in every scope that listed it. It
  blocked valid queries: `SELECT name FROM stadium WHERE stadium_id IN (SELECT stadium_id FROM concert WHERE
  year = 2014)` was blocked because `year` isn't a column of `stadium`.

**Decision.**
- **Names are checked against the whole database.** A query that would run is never blocked because the linker
  left a table out. Using a table the prompt didn't show isn't an error: the model may know it from the question.
- **Each column is resolved once, from the innermost query that contains it,** and from there outwards through
  the enclosing queries, as SQLite does. Result aliases, CTEs, derived tables, `rowid`, and double-quoted strings
  that match no column follow SQLite's rules.
- Rules, in order:
  1. The SQL parses as one statement.
  2. That statement is a query (a SELECT, or a set operation such as UNION, with or without WITH).
  3. Nothing in it writes (INSERT, UPDATE, DELETE, CREATE, DROP, ALTER, PRAGMA, ATTACH, DETACH, `SELECT ... INTO`).
  4. Every table and column exists.
- A block carries a machine-readable reason (`unknown_column:T1.nme`) and a message in SQLite's own words
  (`no such column: T1.nme`).
- The guard doesn't try to catch everything that fails at execution. Ambiguous column names, aggregate misuse and
  wrong argument counts are left to SQLite, whose error message drives the repair just as well.
- `inject_limit` stays `false`: an injected LIMIT changes the answer, so eval never adds one.

**Evidence.**
- **Gold SQL blocked (must be 0):**
  - Spider dev 0 of 1,034; BIRD dev 0 of 1,534; BIRD mini-dev 0 of 500.
  - Spider train 1 of 8,659: example 3153, whose gold names a table that doesn't exist (ADR-014).
- **Phase 1 predictions on Spider dev** (1,034 each), guard verdict vs what SQLite did:

  | | allowed, ran | allowed, failed | blocked, failed | blocked, would have run |
  |---|---|---|---|---|
  | 9B | 1,024 | 0 | 10 | 0 |
  | 4B | 1,013 | 3 | 18 | 0 |

  The 4B's three that got past the guard failed with an ambiguous column, a misused aggregate and a wrong
  argument count.
- `tests/test_guard.py`: 45 cases, and every case the guard allows also runs in SQLite.

**Consequence.**
- The guard makes no false blocks on any gold query or Phase 1 prediction. Its value is an early, exact error
  message, not extra accuracy: without repair it can't change EX. So it is on only in the repair configs.

## ADR-017 — Schema linking: bge-small, top 4 plus foreign-key neighbors, and what to cut when the budget binds
*2026-09-30, Phase 2; amends PLAN §6.1 and ADR-015 (the shedding order with a linker)*

**Context.**
- PLAN §6.2 specifies:
  - One doc per table and one per column.
  - The query is the question plus BIRD's evidence.
  - A table's score is max(table-doc similarity, best column similarity), plus a small bonus when the
    question names it.
  - Keep every table of a database with at most 6; otherwise keep the top k=4 and their one-hop foreign-key
    neighbors.
  - Target at least 95% link recall on both dev sets, and tune k only if below.
- The linked configs use the training budget: a 1,536-token schema in a 2,048-token prompt, including the 256-token
  reply. `scripts/schema_token_stats.py` showed BIRD's full schemas don't fit (ADR-015).
- PLAN §6.1's order for linked prompts: drop samples, then descriptions (from every table), then reduce unselected
  tables to their keys, then drop them.

**Decision.**
- **Embedder:** `BAAI/bge-small-en-v1.5` at a pinned revision, via sentence-transformers on the CPU.
  - Vectors are normalized, so similarity is a dot product.
  - The question gets bge's retrieval instruction ("Represent this sentence for searching relevant passages: ");
    the docs don't.
  - Docs spell names as words: `setCode` becomes `set code`, `singer_in_concert` becomes `singer in concert`.
  - A table doc is its name and column names. A column doc is the table and column name, BIRD's description and
    the sample values.
  - Doc vectors are cached per database, keyed by a hash of the docs.
- **Selection:** `top_k: 4`, `fk_hops: 1`, `all_tables_up_to: 6`, `lexical_bonus: 0.1`. A table is "named" when
  every word of its name, with a plural `s` removed, appears in the question or evidence.
- **Tuning happened on Spider train, never on a dev set.** The bonus (0, 0.05, 0.1) and the instruction (on, off)
  were chosen by recall on Spider train's 3,916 questions on databases with more than 6 tables. k and fk_hops
  stayed at the plan's values because the dev recall target was met, as the plan prescribes.
- **Shedding order for linked prompts (changed from PLAN §6.1):**
  1. `full`
  2. `keys_only_unselected`
  3. `selected_only`
  4. `selected_no_samples`
  5. `selected_no_descriptions`
  6. `pruned_columns`

  Tables the linker didn't select shrink and go before the selected ones lose samples or descriptions.
  `pruned_columns` is the plan's "top-N columns (keep PK/FK) if over budget": the selected tables keep their
  key columns plus as many of their other columns as fit, best column similarity first. The largest number that
  fits is found by binary search. Without a linker the order is unchanged: `full`, `no_samples`,
  `no_descriptions`.

**Evidence.**
- **Link recall** (share of questions whose kept tables include every table the gold SQL reads). "Big" means
  databases with more than 6 tables; smaller ones keep every table.

  | k=4, fk_hops=1 | Spider train (big) | Spider dev | BIRD dev | BIRD mini-dev |
  |---|---|---|---|---|
  | instruction, bonus 0 | 93.8% | 100% | 98.8% big, 99.3% all | 99.0% big, 99.4% all |
  | instruction, bonus 0.05 | 94.2% | 100% | 99.1% big | 99.3% big |
  | **instruction, bonus 0.1** | **94.3%** | **100%** | **99.5% big, 99.7% all** | **99.7% big, 99.8% all** |
  | no instruction, bonus 0.1 | 94.4% | 100% | 99.4% big | 99.7% big |

  - Without foreign-key hops (k=4), recall on BIRD dev's big databases is 76.6–87.1%. The hops matter.
  - The bonus helps most there: 75.7% → 77.5% on Spider train and 79.6% → 87.1% on BIRD dev (big).
  - At k=4, the instruction changes Spider train recall by at most 0.5 points either way. It stays on
    because it's how bge is meant to be queried.
- **The cost of the hops** (`nl2sql link-recall`): on databases with more than 6 tables, the linker keeps 8.26
  of 9.13 tables on average in BIRD dev, 8.4 of 9.29 in BIRD mini-dev, and 6.78 of 9.46 in Spider dev. So
  linking doesn't make BIRD prompts small on its own. What it adds is knowing which tables matter when something
  has to be cut.
- **Shedding order**, measured at budget 1,536 with k=4 and fk_hops=1 (bonus 0.05, before the bonus was chosen):

  | | PLAN §6.1 order | unselected first |
  |---|---|---|
  | Spider dev: gold columns' samples in the prompt | 83.4% | 91.8% |
  | BIRD mini-dev: gold columns' descriptions in the prompt | 61.7% | 64.8% |
  | BIRD mini-dev: every gold column visible | 100% | 99.6% (2 questions) |
  | BIRD mini-dev: over budget even at the last level | 10.2% | 10.2% |

  With the plan's order, linking barely changes a prompt. On Spider it produces exactly the full-schema prompt
  at the same budget.
- **Final linked prompts** (`pruned_columns` included; prompt + 256 reply tokens):

  | | over 2,048 | median | 95th pct | every gold column visible | levels |
  |---|---|---|---|---|---|
  | Spider dev | 0 | 904 | 1,701 | 100% | 874 full, 52 keys-only unselected, 35 selected only, 73 no samples |
  | BIRD mini-dev | 0 | 1,503 | 1,932 | 99.2% | 122 full, 193 no samples, 134 no descriptions, 51 pruned |
  | BIRD dev | 0 | 1,504 | 1,912 | 99.3% | 338 full, 1 keys-only unselected, 654 no samples, 412 no descriptions, 129 pruned |

**Alternatives.**
- Keep PLAN §6.1's order. It keeps unselected tables visible longest, which protects against linker misses. But
  it strips samples and descriptions from the tables that matter first, and 10% of BIRD mini-dev still wouldn't
  fit.
- Tune k and the bonus on a dev set. The grid is cheap, but tuning on dev would make dev recall optimistic.
- A cross-encoder reranker over the top tables. That's future work in the plan; recall is already above target.

**Consequence.**
- Every linked prompt fits the 2,048-token training length, the same length Phase 3 trains at.
- Row B of the ablation (linking at 1,536/2,048 against full schema at 3,072/4,096) changes the budget as well as
  the linking. That's deliberate: the linked configuration is the one that fits the training length, and the
  results say so.

## ADR-018 — Pipeline events, the repair loop, and what the harness records
*2026-09-30, Phase 2*

**Context.**
- PLAN §6.7: on a guard failure, an execution error or a timeout, send a repair turn with the original messages,
  the model's reply, and `The query failed: {reason}. Fix it and return only the corrected SQL.`
  - At most 2 repairs.
  - An empty result is not a trigger by default.
- PLAN §6.8: the pipeline emits typed events, and the eval harness consumes the same events. The benchmark and
  the demo then run one code path.
- ADR-012 requires the client's token count to equal the server's on every call. Repair turns are multi-turn
  chats, which Phase 1 never sent.

**Decision.**
- **`Pipeline.run` yields pydantic events:**
  - `run_started`, `schema_linked`, `prompt_built`
  - `sql_generated`, `guard_result`, `executed`, `repair_started`
  - `run_finished` or `run_failed`

  The harness builds each prediction record from them (`record_from_events`).
- **Triggers:** a guard block, an execution error, a timeout, and a reply with no SQL. A reply with no SQL has
  nothing for the guard to check, so it counts as a guard failure.
- **The reason** is the guard's SQLite-worded message, or SQLite's own error text without the Python exception
  class, or "it did not finish within 30 seconds".
- **An empty result is not a trigger.** `on_empty_result` accepts only `false`, so a config can't claim it
  (BIRD gold can be empty).
- **The pipeline executes the model's query only to learn whether it fails,** fetching at most 10,000 rows
  (PLAN §6.6). It never sees gold. The harness scores the final query separately, with ADR-013's rules.
- **The scored prediction is the last SQL the model wrote,** even when the pipeline gave up on it. `failure`
  records why: `guard_blocked`, `exec_error`, `timeout`, `no_sql` or `prompt_too_long`.
- **Budgets:** `max_total_tokens` applies to the first turn, the training length. Repair turns may go past it.
  The client still refuses any prompt the context window can't hold, and still requires the server's count to
  match (ADR-012).
- **Latency** is per question, end to end: linking through the last execution, repairs included. A reply served
  from the response cache counts at its original generation time, so a rerun reports the same latency.
  Manifests say `latency_scope: pipeline`. Phase 1 timed the model call alone, and `RESULTS.md` marks those
  numbers.
- **Metrics:**
  - `guard_block_rate`: first answers blocked, over all questions.
  - `repair_rate`: questions with at least one repair.
  - `repair_success_rate`: repaired questions that ended on a query that ran.
  - `link_recall`
  - `repaired_correct`: repaired questions scored correct. Each is a point the first answer lost, since that
    answer couldn't run.

  Each metric is `null` when its stage is off.

**Evidence.**
- **Multi-turn token counts** (`system, user, assistant, user`) on 24 Spider dev conversations per model:
  - The client's count equals Ollama's `prompt_eval_count` exactly, on both the 9B and the 4B, 24 of 24 each.
  - The assistant turn was tried four ways: the raw reply, the reply with a trailing newline, the bare SQL, and
    the reply wrapped in `<think>` tags.
- `tests/test_pipeline.py` drives the pipeline with a scripted model:
  - A guard block is repaired, and the second request is the first plus two turns.
  - An execution error is repaired without the guard.
  - A run gives up after `max_repairs` and keeps the last SQL.
  - An over-budget prompt is never sent.
  - Cached replies keep their latency.
  - Events survive a JSON round trip.

**Consequence.**
- A repair can only turn a failed query into one that runs; it can't fix a query that runs and is wrong. Its
  ceiling is the first answers' failure rate. In Phase 1 on Spider dev, 10 (9B) and 21 (4B) of 1,034 first
  answers failed to run.
- Results for the linked and repair configs record per-attempt detail (`attempts[]`) in `predictions.jsonl`,
  so any failure can be traced to its turn.

## ADR-019 — Retry replies Ollama didn't finish
*2026-09-30, Phase 2; extends ADR-012*

**Context.**
- Two calls failed while the base 4B ran on BIRD mini-dev (long prompts, ~2,000–2,900 tokens). In both, Ollama's
  log shows llama-server stopping at the same step: it had processed all but the last 4 prompt tokens and never
  logged a prompt-eval time.
  - **First:** the request hung for 14 min 51 s and then returned HTTP 500. The client's retry got a normal reply
    in 0.9 s. The run then stalled until the session's time limit stopped it; the cause of that stall isn't known.
  - **Second, on rerunning the split:** Ollama returned **HTTP 200** after 2.1 s and then cancelled the task. The
    reply had no `prompt_eval_count`, so ADR-012's check stopped the run with "server read None".
- ADR-012's check exists to catch truncation, which shows up as a count that differs from the client's. A missing
  count isn't truncation. It's a reply the server never finished.

**Decision.**
- A 200 reply without `done: true` or without `prompt_eval_count` counts as a failed call. It's retried with
  backoff (1, 2, 4 s) like a 5xx or a dropped connection. The run stops only after 4 failed attempts.
- Every retry is logged as a warning, so it shows in the run's log.
- A reply that reports a count different from the client's still stops the run at once (ADR-012).
- The read timeout drops from 600 s to 300 s. A 256-token reply takes seconds, and a cold 9B load under a minute.
- Retrying can't change results: decoding is greedy with a fixed seed, and only a finished reply is cached or
  scored.

**Consequence.**
- Ablation runs 1–7 ran at commit `42dbc46` and runs 8–12 at the commit with this change. The two differ only
  in how failed calls are handled; each manifest records its commit.
