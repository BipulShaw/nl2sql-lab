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
  - ~~Assert `prompt_tokens < num_ctx` on every response.~~ Phase 1 showed this check can never fire; see
    "Context windows and truncation" below (ADR-012).
- The model's shipped sampling defaults are `temperature 1` and `presence_penalty 1.5`:
  - Presence penalty discourages reusing tokens already in the text. That's bad for SQL, which legitimately
    repeats column and table names.
  - For eval we send `temperature 0` (greedy, deterministic) and `presence_penalty 0`.
- **Where:** `scripts/smoke_ollama.py`; `src/nl2sql/llm/ollama.py` (`OllamaLLM.options`).

---

## Phase 1

### Execution accuracy (EX) vs exact match
- There are two ways to grade generated SQL:
  - **Exact match** compares the SQL text, clause by clause. A correct query written differently fails: a JOIN
    instead of a subquery, or `COUNT(id)` instead of `COUNT(*)`.
  - **Execution accuracy (EX)** runs both the gold and the predicted query on the database and compares the
    results. Any query that returns the right answer passes. It's the headline metric of both Spider and BIRD.
- EX has **false positives**. A wrong query can return the right rows by coincidence on this particular data. The
  unit tests have one: on rows (1,'x'), (2,'y'), (3,'y'), `WHERE a >= 2` and `WHERE b = 'y'` return the same
  rows. Spider's "test-suite" accuracy runs each query on many generated databases to catch these. We use the one
  official database, as BIRD does.
- EX has **false negatives** when the gold is wrong. Some pilot examples:
  - Spider dev 554: the gold looks for `first_name = 'timmothy'`, but the data says 'Timmothy'. The gold returns
    0 rows, the model's correctly cased query returns the student, and the model is marked wrong.
  - Spider dev 387, 583 and 786 fail the same way.
- **"Compare the results" hides decisions**, and each one moves the score (ADR-013):
  - Multiset or set? Duplicates count for us but not for BIRD.
  - Does row order count? Only under a top-level `ORDER BY`.
  - Must columns come in the gold's order? For us yes; Spider's evaluator allows any order.
  - In the 9B pilot, column order alone cost 5 points (75.0% vs 80.0%).
- **Where:** `src/nl2sql/eval/ex_metric.py`; `scripts/failure_breakdown.py` sorts the misses by kind.

### Greedy decoding, and why eval answers are cached
- At each step the model produces a score for every token in its vocabulary (248,320 here), and decoding picks
  one:
  - **Sampling** draws at random. Temperature reshapes the odds: logits are divided by T, so T < 1 sharpens them
    and T > 1 flattens them. `top_p` and `top_k` cut off the unlikely tail.
  - **Greedy** (temperature 0) always takes the single most likely token. The same prompt gives the same output,
    in principle.
- For eval we use greedy. One run is enough: no averaging over samples. Models compare on equal terms, and every
  failure can be looked at again.
- "In principle": a GPU may add floating-point numbers in a different order from one run to the next (batch
  size, kernel choice). That can flip a near-tie between two tokens, and from then on the outputs differ.
  Phase 2 measured it: identical within one loaded session, but 14–15 of 60 replies differ after a model reload
  ("Greedy isn't deterministic across model loads", Phase 2).
- That's one reason for the **response cache** (`data/cache/llm_responses.sqlite`):
  - The key is a hash of the backend, the model digest, the quantization, the adapter and the full request.
  - A rerun reuses the recorded answer instead of hoping for the same one. So a fixed or added metric re-scores
    old answers exactly, at no GPU cost. The any-column-order metric was added that way.
- Greedy's failure mode is the **repetition loop**, where the likeliest next token keeps restarting the same
  phrase. `max_new_tokens` (256) bounds it, and the manifest counts replies that hit the limit (`finish_length`):
  0 of 200 in the pilot. Replies averaged 39 tokens.
  - The full 4B run hit one (Spider dev 766, "people living in nations that do not use English"). Its `WHERE`
    clause cycles through `OR cl.Language IS NULL AND cl.IsOfficial IS NULL`, `OR cl.Language != 'English'
    AND ...` and the like, repeating the same few lines until the 256 tokens run out. The SQL is cut off
    mid-clause, so SQLite rejects it ("incomplete input") and it counts as wrong.
- `seed 42` is still sent. At temperature 0 it changes nothing, but it costs nothing and is recorded.
- **Where:** `OllamaLLM.options()` in `src/nl2sql/llm/ollama.py`; `src/nl2sql/llm/cache.py`.

### Context windows and truncation: the "747" experiment
- The **context window** is how many tokens the model attends to at once: the prompt plus everything it
  generates. Qwen3.5 supports 262,144. The server allocates only `num_ctx` (8,192 here), because the memory for
  the window is reserved at load time (next entry).
- **What happens when a prompt doesn't fit**, measured by `scripts/probe_ollama_context.py`:
  - The probe sent 1,000 numbered lines (15,921 tokens) with `num_ctx` 8,192 and asked for the number of the first
    line.
  - No error came back. Ollama reported reading 4,098 tokens.
  - The model answered "747". It had only seen part of the prompt, not its start.
- In our prompts the schema comes first. A truncated prompt means SQL written against tables the model never saw,
  and still no error.
- **Why the obvious check fails:** the server reports the count of what it *kept*. That's 4,098, under 8,192, so
  `prompt_tokens < num_ctx` passes exactly when truncation happened.
- **What works:** count the prompt yourself with the same tokenizer and chat template, and require the server's
  count to equal yours (ADR-012). This relies on the two counts agreeing whenever nothing is cut. The probe showed
  they do: 1,520 = 1,520, even when Ollama reused cached prefix tokens.
- Analogy: MySQL in non-strict mode cuts a string that is too long for a `VARCHAR(255)` column and raises only a
  warning. The insert succeeds and the data is wrong. Only comparing lengths yourself catches it.
- **Where:** `scripts/probe_ollama_context.py`; `src/nl2sql/llm/ollama.py`; `src/nl2sql/llm/tokens.py`.

### KV cache: what `num_ctx` costs in VRAM
- In attention, each new token looks at the **keys and values** of every earlier token. Recomputing them at every
  step would redo all the past work, so the server stores them: the **KV cache**. It grows linearly with the
  context window and is reserved when the model loads.
- Size per token = 2 (K and V) × attention layers × KV heads × head dim × bytes per number.
- The Qwen3.5-9B numbers, from Ollama's model metadata (`/api/show`):
  - 32 layers, but only every 4th is full attention, so 8 layers keep a KV cache.
  - 4 KV heads. This is grouped-query attention: 16 query heads share 4 sets of keys and values.
  - Head dim 256. Numbers are f16, 2 bytes each (Ollama's default cache type).
- 2 × 8 × 4 × 256 × 2 bytes = **32 KiB per token**:

  | Context | KV cache |
  |---|---|
  | 8,192 (`num_ctx`) | 256 MiB |
  | 262,144 (the model's maximum) | 8 GiB, more than the whole card |
  | 8,192, if all 32 layers were standard attention | 1 GiB |

- The other 24 layers are Gated DeltaNet (Phase 0 entry). Each keeps a fixed-size state instead: 32 value heads ×
  a 128 × 128 matrix, 524,288 numbers per layer, about 2 MiB in fp32. That doesn't grow with the context.
- Measured while the eval runs: Ollama puts the loaded 9B at 5.24 GiB of VRAM at `num_ctx` 8,192, covering
  weights, KV cache and compute buffers. The card shows 6.85 of 8.0 GiB used, desktop included.
- This is why Ollama picks a small default `num_ctx`. For most models, where every layer keeps a KV cache, a long
  window costs gigabytes.
- **Where:** `context_window` in `configs/*.yaml`; Ollama `/api/show` (architecture) and `/api/ps` (VRAM).

### Is a one-point gap real? Paired comparison and McNemar's test
- On the same 1,034 Spider dev questions the 9B scored 74.3% and the 4B 73.4%. Is the 9B better, or is
  0.9 points noise?
- **The data is paired.** Both models answered the same questions, and most questions are easy for both or hard
  for both: 704 both right, 211 both wrong. Those 915 say nothing about which model is better. Only the 119
  disagreements do: the 9B alone right on 64, the 4B alone on 55.
- **McNemar's test** asks: if the models were equally good, each disagreement would be a coin flip, so how likely
  is a split at least as uneven as 64–55? The binomial distribution answers exactly: p = 0.46. A split like
  that happens by chance about half the time, so it's no evidence either way.
- **How big a gap would count?** With 119 disagreements, 71–48 (a gap of 23 examples, 2.2 points) gives
  p = 0.043, while 70–49 gives p = 0.066.
- **A pilot is a smoke test, not a measurement.** The 200-example pilots showed 75.0% vs 72.0% (13–7
  disagreements, p = 0.26). On the full split the gap shrank to 0.9 points.
- **Why not compare two confidence intervals?** Treating the scores as independent samples ignores the pairing.
  Here that gives a standard error of about 1.9 points for the difference, against about 1.1 for the paired
  comparison, because the question-to-question variation that the pairing cancels is counted in.
- What it changes: Spider dev can't tell these two models apart, so it can't show a fine-tune closing the gap
  either. Phase 3's comparison rests on BIRD, and each model comparison reports the paired p-value.
- **Where:** `src/nl2sql/eval/paired.py`; `uv run nl2sql results compare <run A> <run B>`.

## Phase 2

### Schema linking is retrieval: bi-encoders and recall
- **The problem:** BIRD's schemas run to 7,600 tokens with descriptions and samples, and the training prompt is
  2,048. Something has to decide what the model sees, before the model sees anything.
- **A bi-encoder** turns each text into one vector, independently. The docs (a table's name and columns, a
  column's name, description and sample values) are embedded once per database and cached. Each question costs
  one more encoding and a matrix product. Scoring is a dot product, because the vectors are normalized. A
  **cross-encoder** would read question and doc together and score more accurately, but it would run once per
  table per question. That's the plan's future work.
- **Asymmetric retrieval:** bge was trained with an instruction prefix on queries ("Represent this sentence for
  searching relevant passages: ") and none on passages. Here it moved Spider train recall by at most half a
  point, but it's how the model is meant to be used.
- **Recall, not precision, is the metric.** A table the linker keeps but the query doesn't need costs tokens. A
  table it drops that the query needs makes the question unanswerable. So link recall is the share of questions
  whose kept tables include every table the gold query reads.
  - The gold tables come from parsing the gold SQL with the guard's own name resolver.
  - With k=4 plus foreign-key neighbors, recall is 100% on Spider dev and 99.7% on BIRD dev.
- **The foreign-key hop is most of the recall.** Without it (the top 4 alone), BIRD dev recall on big databases
  is 87%; with it, 99.5%. Join
  tables (`singer_in_concert`) rarely look like the question, but they sit next to the tables that do. The price:
  on BIRD the linker keeps 8.3 of 9.1 tables, so linking alone doesn't shrink prompts.
- **Where:** `src/nl2sql/linking/`; `uv run nl2sql link-recall --dataset bird --split dev`.

### Tuning without touching the test set
- Every choice made by looking at a score overfits to the data that produced the score. Tune on dev and dev
  numbers stop being an honest estimate.
- The linker's knobs (lexical bonus, query instruction) were chosen on **Spider train**: 3,916 questions on
  databases with more than 6 tables. Dev recall was only read afterwards, and k stayed at the plan's value
  because dev met the target, as the plan prescribes.
- The grid still printed dev recall next to train. That's fine as long as the choice is written down before
  looking at dev, and the ADR says which set chose what (ADR-017).

### What to cut when the prompt doesn't fit
- The serializer has an ordered list of ever-smaller renderings, and the first one under budget wins.
- The plan's order stripped samples and descriptions from every table before touching the ones the linker didn't
  pick. On Spider that made linking a no-op: the prompt came out identical to the full schema. It also removed
  detail from exactly the tables that matter.
- The new order drops unselected tables first, then strips the selected ones (ADR-017). That kept more of the
  gold columns' samples (Spider, 83% → 92%) and descriptions (BIRD, 62% → 65%). The cost was 2 of 500 BIRD
  questions that lost a gold column because the linker missed its table.
- Even the selected tables alone didn't fit for 10% of BIRD mini-dev. **Column pruning** keeps each selected
  table's keys, then as many of its other columns as fit, best linker score first. Tokens only grow as columns
  are added, so a binary search finds the largest set in about 7 tokenizer calls.
- Result: every linked prompt in the three dev sets fits 2,048 tokens, and 99.2–100% keep every gold column.
- **Where:** `src/nl2sql/schema/serialize.py`.

### Scopes: why "which table is this column from?" is hard
- `SELECT name FROM stadium WHERE stadium_id IN (SELECT stadium_id FROM concert WHERE year = 2014)`. The inner
  query can see its own tables and, if it's correlated, the outer query's. So an unqualified `year` could
  belong to `concert` or to `stadium`.
- SQLite resolves a name from the innermost query outwards. sqlglot's scope analysis doesn't pick for you: it
  lists `year` under the inner query and, as a possible correlated reference, under the outer one too.
- The first guard checked every listing and blocked this valid query, because `stadium` has no `year`. The fix:
  resolve each column once, from the deepest scope that contains it, walking outwards (ADR-016).
- The test that proves the guard right is **agreement with SQLite**. On 11,727 gold queries and 2,068 model
  queries, nothing the guard blocked would have run.
- **Where:** `src/nl2sql/guard/sql_guard.py`; `tests/test_guard.py`.

### Self-repair: execution feedback as a second turn
- When a query fails, the pipeline sends a repair turn with the model's own reply and the error:
  `The query failed: no such column: T1.nme. Fix it and return only the corrected SQL.` This is at most 2 times.
- **It can only fix queries that fail.** A query that runs and returns the wrong rows looks like success to the
  pipeline, which never sees gold. So repair's ceiling is the first answers' failure rate (1–2% of Spider dev in
  Phase 1), not the whole error rate.
- **Multi-turn prompts have to be counted too.** The client asserts that its token count equals the server's on
  every call (ADR-012). A conversation with an assistant turn runs through parts of the chat template that
  single-turn prompts never reach. So the counts were checked again on 24 conversations per model, with the
  assistant turn in four forms (one containing a `<think>` block), and they matched exactly.
- **Where:** `src/nl2sql/pipeline/runner.py`; the metrics `repair_rate`, `repair_success_rate` and
  `repaired_correct` in every manifest.

### Greedy isn't deterministic across model loads
- Temperature 0 means "take the likeliest token". So the same prompt should give the same reply. Within one
  loaded model it does: 60 of 60 prompts sent twice back to back came back identical.
- Across model loads it doesn't. The same 60 BIRD prompts, compared with replies recorded a few hours earlier
  (the models had been swapped in and out in between), differed on 14–15.
  - Mostly the differences are cosmetic: a column alias, an added `COALESCE`.
  - Some change the query. One question got a degenerate reply (a run of zeros that Ollama cut off) five times in
    one session and a normal answer after a reload.
- **The likely mechanism** (not verified here): when the server loads a model it picks runtime settings from
  free VRAM, such as how layers are split and which kernels run. A different order of floating-point additions
  can flip a near-tie between the top two tokens, and every token after that follows the new path.
- **What it does to a score:** regenerating the 9B's full 500-question BIRD run from scratch gave 480 identical
  queries and EX 43.6% vs 43.8%, one flipped answer. That's small next to the effects measured, but it's not zero.
- **How the project lives with it** (ADR-020):
  - The response cache is the record: every reported number re-scores recorded replies, and re-scoring is exact.
  - Comparisons are paired on the same questions, where random flips can't manufacture a one-sided gap.
  - Configs that share a first turn share its recorded reply, so a repair comparison only measures repair.
- **Where:** `scripts/determinism_probe.py`, `scripts/rerun_noise.py`.
