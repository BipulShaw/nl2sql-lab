# Learning log

One entry for each new LLM concept: what it is, why it matters here, and where it lives in the repo.
Written as interview prep, so each number is either measured on this machine or worked out in the entry.

---

## Phase 0

### Quantization, nf4 and QLoRA
- **Quantization** stores each weight in fewer bits. A bf16 weight takes 2 bytes; a 4-bit weight takes 0.5 bytes,
  plus a little for scale factors. It's lossy compression, like giving each block of 64 weights its own
  16-color palette (a GIF, not a PNG).
- **nf4** ("normal float 4") places those 16 levels at the quantiles of a normal distribution. Trained weights are
  roughly normally distributed, so this wastes fewer levels than evenly spaced ones would. **Double quant** also
  quantizes the per-block scale factors, which saves about another 0.4 bits per weight.
- **LoRA** freezes the base weights W and trains a small delta: W' = W + B·A, where A is r×d and B is d×r. With
  r=16 on a 4096×4096 matrix that's 131k trainable parameters instead of 16.8M (0.78%).
- **QLoRA** is LoRA on a base model stored in nf4. For each matmul the weights are dequantized to bf16 on the fly;
  gradients flow only into A and B.
- **Caveat for this model family:** Unsloth's Qwen3.5 guide advises against 4-bit training for Qwen3.5 because
  of larger quantization differences. So Phase 1 compares base-model EX in nf4 vs int8 before we commit
  (DECISIONS ADR-004).
- **Where:** `scripts/smoke_hf.py` (`BitsAndBytesConfig`); later `src/nl2sql/train/sft.py`.

### Why you can run a 9B but can't fine-tune it on this card
- **Inference** needs the weights plus a small KV cache. The 9B in Ollama is a 6.14 GiB Q4_K_M file, and about
  7 GiB of the 8 GB card is usable (the desktop takes the rest). It fits.
- **Training** adds four things on top:
  - LoRA parameters and their optimizer state (small, ~0.3 GB).
  - Activations saved for the backward pass (~0.5 GB even with gradient checkpointing).
  - CUDA context (~0.4 GB).
  - The **logits**: seq × vocab × 4 bytes. With Qwen3.5's 248,320-token vocab at seq 1536 that is
    1,525,678,080 bytes = **1.42 GiB**, and the backward pass needs a gradient tensor the same size.
- That leaves under 1 GiB for all of it after ~5.5+ GB of 4-bit weights, so the 9B fine-tune is dropped
  (ADR-005).

### Chat templates and thinking mode
- A chat model never sees JSON messages. It was trained on one exact string format with special tokens
  (`<|im_start|>user\n...<|im_end|>`). The **chat template** is the serializer from a `messages` list to that
  format; a Jinja template ships with the tokenizer.
- Use the model's own template. A hand-rolled format still "works" but is quietly out-of-distribution and
  scores worse. That's why PLAN §6.3 uses one prompt builder for both training and inference.
- **Thinking mode:** Qwen3.5 writes a `<think>…</think>` reasoning block before answering, and does so by default.
  For eval and training data we turn it off:
  - `enable_thinking=False` in `apply_chat_template` (transformers);
  - `"think": false` in Ollama's native API.
- Thinking off keeps outputs short, deterministic and comparable across models. Qwen3.5 has no `/no_think` prompt
  switch.
- **How "off" is encoded:** the template pre-fills an *empty* reasoning block, so the rendered prompt ends with
  `<|im_start|>assistant\n<think>\n\n</think>\n\n`. The model sees it has "already thought" and goes straight to
  the answer. That's why the same prompt is 114 tokens with thinking off and 112 with it on (measured).
- **Where:** `scripts/smoke_hf.py` prints the end of the rendered prompt, so you can see how "thinking off" is
  encoded; `scripts/smoke_ollama.py`.

### Hybrid attention: Gated DeltaNet + Gated Attention (Qwen3.5)
- **Standard attention** compares each new token with every earlier token. Its KV cache grows with context
  length.
- **Gated DeltaNet** is a *linear-attention* layer that works like an RNN. It keeps a fixed-size state matrix, a
  small key→value associative memory:
  - The **delta rule** overwrites the value stored for a key instead of piling more on top.
  - A **gate** decides how fast old memory decays.
- Qwen3.5 stacks 8 blocks of (3 DeltaNet layers + 1 full attention layer) = 32 layers. Only 1 layer in 4 keeps a
  KV cache, so long contexts are cheap.
- **Costs:**
  - Speed needs custom GPU kernels (flash-linear-attention, causal-conv1d). Without them transformers falls back
    to slow pure-PyTorch loops.
  - The layers have different projection names, which affects LoRA target modules (ADR-008).
- **Where:** transformers `models/qwen3_5/modeling_qwen3_5.py`; kernel status is recorded in `docs/STATUS.md`.

### What's inside 2.9 GiB of "4-bit" weights: tied embeddings
- The 4B loaded in nf4 takes 2.90 GiB, not the ~2.1 GiB that "4.21B params × 0.5 bytes" suggests.
- The difference is the **embedding table**: 248,320 tokens × 2,560 dims = 636M parameters. bitsandbytes doesn't
  quantize embeddings, so they stay bf16: 1.18 GiB, 28% of the memory for 15% of the parameters.
- **Tied embeddings** (`tie_word_embeddings: true`) mean the output layer `lm_head` reuses that same matrix instead
  of storing a second copy. That saves another 1.18 GiB, and it's one reason we don't LoRA-train `lm_head`
  (ADR-008).
- The other 3.57B params in nf4 come to ≈1.66 GiB, plus scale factors: that adds up to the measured 2.9 GiB.
- Big-vocab models pay for their vocabulary twice: once here and once in the logits (next entry).

### Vocabulary size, logits memory and fused cross-entropy
- The last layer turns each position's hidden vector into a score for every vocabulary token (the logits). For
  training loss that's a [seq × 248,320] matrix: **1.89 GiB** in fp32 at seq 2048, plus the same again for its
  gradient.
- **Fused linear cross-entropy** (Liger kernel) computes the output projection and the loss in chunks of
  positions, so the full matrix never exists at once. It's like paginating a query instead of loading the whole
  table into memory.
- The loss is identical; peak memory drops by gigabytes.
- **Where:** TRL `SFTConfig(use_liger_kernel=True)` in `src/nl2sql/train/sft.py` (Phase 3).

### Triton, and why the Python side runs in WSL2
- **Triton** is a Python-embedded language that JIT-compiles GPU kernels at runtime; it's how people write custom
  kernels without CUDA C++. Liger (fused CE) and flash-linear-attention (DeltaNet) are both written in it.
- Triton officially supports Linux only. It also needs a C compiler on first use to build a small launcher
  (hence `build-essential`).
- On native Windows those kernels don't load: no fused CE (OOM risk in training) and slow DeltaNet layers.
- WSL2 runs a real Linux kernel and passes CUDA through to the Windows NVIDIA driver. The Linux wheels therefore
  run at essentially native speed (ADR-001).

### VRAM overflow that doesn't crash (Windows/WSL oversubscription)
- On Linux servers, asking for more GPU memory than exists raises `CUDA out of memory` immediately.
- On Windows (and so in WSL2) the driver can instead **page GPU memory out to system RAM**, the way an OS swaps to
  disk. Nothing fails; everything just gets slower.
- Measured here: the 4B in bf16 (7.85 GiB) "fit" on a card with ~7.3 GiB free, and decoded at 5.0 tok/s, slower
  than the 4-bit model's 13.1.
- The fix is to cap PyTorch's allocator with `torch.cuda.set_per_process_memory_fraction(0.85)`, which makes
  PyTorch raise OOM above 6.8 GiB.
- The JS analogy: it's like turning an unbounded memory leak into a hard `--max-old-space-size` crash you can
  react to.
- **Where:** `scripts/smoke_hf.py --mem-fraction`; later every GPU entry point (ADR-011).

### Serving defaults: context window and sampling (Ollama)
- **`num_ctx`** is the context window the server allocates. Ollama picks a default based on available VRAM,
  far below the model's 262,144-token maximum.
- If a prompt is longer than `num_ctx`, Ollama **silently truncates** it. The schema gets cut off and the SQL is
  wrong, with no error anywhere.
- Our defenses:
  - Send `num_ctx: 8192` on every request.
  - Assert `prompt_tokens < num_ctx` on every response.
- The model's shipped sampling defaults are `temperature 1` and `presence_penalty 1.5`:
  - Presence penalty discourages reusing tokens already in the text. That's bad for SQL, which legitimately
    repeats column and table names.
  - For eval we send `temperature 0` (greedy, deterministic) and `presence_penalty 0`.
- **Where:** `scripts/smoke_ollama.py`; later `src/nl2sql/llm/openai_compat.py`.
