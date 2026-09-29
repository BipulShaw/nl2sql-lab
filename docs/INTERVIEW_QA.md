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
request sets `num_ctx` to 8192, and the client checks every response for truncation. (In Phase 0 that check was
`prompt_tokens < num_ctx`; Phase 1 showed it can never fire and replaced it, see below.) The shipped
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

## Phase 1 — data, metric and baseline

**Q: How do you know the model actually saw the whole prompt?**
I measured it, because the obvious check turned out to be useless. I sent Ollama a 15,921-token prompt with an
8,192-token window. There was no error, and it reported reading 4,098 tokens. Asked for the number of the first
line, the model answered "747": it never saw the start. The server reports the count of what it *kept*, so a
check like `prompt_tokens < num_ctx` passes exactly when truncation has happened. Now the client counts every
prompt with the model's own tokenizer and chat template, refuses to send one that can't fit, and requires the
server's count to equal its own afterwards. The probe showed the two counts agree to the token when nothing is
cut, even with KV-cache reuse, so any difference means something is wrong and the run stops (ADR-012).

**Q: What exactly is your EX metric? Would your numbers match the leaderboards?**
Not exactly, and the README says so. EX executes gold and prediction on the same database. It compares rows as
multisets, or as ordered lists when the gold has a top-level `ORDER BY`, and requires the gold's column order.
BIRD's official script compares *sets*, so duplicates and row order never matter there. Spider's test-suite
script accepts any column order and removes `DISTINCT` before comparing. Each choice moves the score. In my 9B
pilot, column order alone was 5 points, 75% vs 80%. So I report strict EX as the headline and EX with any column
order next to it (ADR-013). I also don't cap result rows at 10,000 as first planned: two truncated results can
compare equal when the full ones differ. The gold result is capped at a million rows, and the prediction is
fetched only one row past the gold's count.

**Q: Did you check the benchmark data itself?**
Yes, before any model run. Every archive is pinned by SHA-256, so a changed upstream file fails the download. A
`data verify` command runs all 11,727 gold queries. Three Spider train queries are broken: one uses a table that
doesn't exist, and two put `ORDER BY` before `INTERSECT`, which SQLite rejects. Two BIRD dev queries take about
50 s and 300 s, over the 30 s timeout. So BIRD dev's best possible EX is 1532/1534, and I report EX both with and
without those examples. The harness is checked by scoring the gold SQL as the prediction: 100% on Spider dev
(ADR-014).

**Q: When the model is marked wrong, is it always wrong?**
No. Reading through the misses found gold queries that are wrong. For example, Spider dev 554 looks for
`first_name = 'timmothy'`, while the data has 'Timmothy'. The gold returns nothing, the model's correctly cased
query finds the student, and EX marks the model wrong. The 200-example pilot had at least four such case
mismatches. I don't
patch the gold, because then my numbers would stop being comparable with anyone else's. Instead, a failure
breakdown script sorts the misses: wrong row counts, wrong values, column order, extra columns. That way I can
say where points go rather than quoting one number.
