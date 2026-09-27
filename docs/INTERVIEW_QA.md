# Interview Q&A

Questions an interviewer is likely to ask, with answers grounded in this repo. Grown phase by phase.

---

## Phase 0 — environment and model choice

**Q: You run a 9B model on an 8 GB laptop GPU. Why not fine-tune it?**
Inference only needs the weights and a small KV cache: the 9B in 4-bit is a 6.14 GiB file, which fits in the
~7 GiB the card leaves free. Training adds optimizer state for the LoRA weights, saved activations, CUDA context
and, above all, the logits. With Qwen3.5's 248,320-token vocabulary, logits at seq 1536 are 1.42 GiB in fp32,
and the backward pass needs a gradient of the same size. That doesn't fit next to ~5.5 GiB of weights, so I
fine-tune the 4B from the same family instead (DECISIONS ADR-005, LEARNING_LOG).

**Q: Why a 4B from the same family rather than a small coder model?**
The headline question is whether fine-tuning closes a model-size gap. Using `Qwen3.5-4B` against `Qwen3.5-9B`
keeps the tokenizer, chat template and architecture identical, so the comparison varies only size and training.
It's Apache-2.0, so I can publish the adapter. I pinned the exact Hugging Face commit so the result can be
reproduced (ADR-003).

**Q: What's the most dangerous silent failure when benchmarking through Ollama?**
Context truncation. Ollama sizes its context window (`num_ctx`) from available VRAM, far below the model's
maximum, and silently drops prompt tokens beyond it. A cut-off schema produces wrong SQL with no error. Every
request sets `num_ctx` to 8192, and the client asserts `prompt_tokens < num_ctx` on every response. The shipped
sampling defaults (temperature 1, presence penalty 1.5) are also overridden to 0, because presence penalty
punishes the repeated column names SQL needs (ADR-007).

**Q: Why WSL2 instead of running Python directly on Windows?**
Two kernels this project needs are written in Triton, which officially supports only Linux. One is Liger's fused
cross-entropy, which keeps the 248k-vocab logits from blowing VRAM during training. The other is
flash-linear-attention, which makes Qwen3.5's Gated DeltaNet layers fast. WSL2 passes CUDA through to the Windows
driver, so the Linux wheels run at native speed. Ollama stays on Windows, reached via mirrored networking
(ADR-001).

**Q: Did anything in environment setup surprise you?**
Yes: running out of GPU memory didn't crash. I loaded the 4B in bf16 as a check I expected to fail: 7.85 GiB of
weights on a card with ~7.3 GiB free. It loaded and generated, just slower than the 4-bit model (5.0 vs
13.1 tok/s). Under WSL the Windows driver pages GPU memory to system RAM instead of raising OOM. In training, that
would turn "this config doesn't fit" into "this run is mysteriously 3× slower" and wreck the overnight estimate.
So every GPU entry point caps PyTorch at 85% of VRAM with `set_per_process_memory_fraction`. I verified that bf16
then fails loudly and nf4 is unaffected (ADR-011).
