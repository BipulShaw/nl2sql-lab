# Decisions

Short ADRs: context → decision → alternatives → consequence. Newest last. Where an ADR changes PLAN.md, the ADR
wins and says so.

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

The client asserts `prompt_tokens < num_ctx` on every response.

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
