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
mismatches. I don't patch the gold, because then my numbers would stop being comparable with anyone else's.
Instead, a failure breakdown script sorts the misses: wrong row counts, wrong values, column order, extra
columns. That way I can say where points go rather than quoting one number.

**Q: What did the baseline show?**
Both models ran zero-shot on all 1,034 Spider dev questions, through Ollama at Q4_K_M with identical prompts.
The 9B scored 74.3% EX and the 4B 73.4%, and that gap isn't significant. The models disagree on 119 questions,
split 64–55, and McNemar's exact test gives p = 0.46. The 200-question pilots had suggested 3 points, which is
why I treat a pilot as a smoke test, not a measurement. Where the points go says more than the headline. About
a quarter of the misses return different values and a third return the wrong number of rows. Column order alone
costs the 9B 4.9 points, which Spider's official evaluator wouldn't count. Only 10 of the 9B's misses fail to
execute, so an execution-repair loop has little to work with on Spider. For fine-tuning, this means two things.
The before-and-after comparison has to be on BIRD, whose schemas are about five times larger. And it has to run
the base and fine-tuned 4B through the same backend and quantization. Otherwise I'd be measuring Ollama's
Q4_K_M against nf4 in transformers, not the effect of training.

## Phase 2 — pipeline: linking, guard, repair

**Q: How does your schema linker work, and how do you know it doesn't drop tables the query needs?**
It's a retrieval problem. Every table gets a doc of its name and columns, and every column a doc of its name,
BIRD's description and sample values. bge-small embeds those once per database, and the question plus BIRD's
evidence is embedded per question. A table scores its best similarity over its own doc and its columns' docs,
plus a small bonus if the question names it. Databases with at most 6 tables keep everything. Otherwise the linker
keeps the top 4 and their foreign-key neighbors. I measure it by link recall: the share of questions whose kept
tables include every table the gold SQL reads, with the gold tables found by the guard's own name resolver. It's
100% on Spider dev, 99.7% on BIRD dev and 99.8% on BIRD mini-dev. The foreign-key hop does most of the work.
Without it, BIRD recall on big databases falls to 87%, because join tables rarely look like the question. I tuned
the two knobs, the bonus and bge's query instruction, on Spider train, never on dev.

**Q: Your linker keeps 8 of 9 tables on BIRD. Then what is it for?**
Deciding what to cut. BIRD prompts don't fit 2,048 tokens even with the right tables. The plan's order was to
strip samples and descriptions everywhere first, and only then shrink the tables the linker didn't pick. I
measured that order against the reverse before choosing. The plan's order threw away detail from exactly the
tables that matter. On Spider it made linking a no-op, because the prompt came out identical to the full schema.
Shrinking unselected tables first kept more of the gold columns' samples and descriptions, and cost 2 of 500 BIRD
questions a gold column. Even then, 10% of BIRD mini-dev didn't fit. So the last resort prunes the selected tables'
lowest-scored columns and keeps their keys, with a binary search on how many columns fit. Every prompt in all three
dev sets now fits 2,048 tokens, and 99.2–100% still show every gold column (ADR-017).

**Q: How does the SQL guard work, and does it ever block a valid query?**
It parses the reply with sqlglot. It allows exactly one statement, requires it to be a query, rejects anything
that writes (including `SELECT ... INTO`), and checks every table and column against the database. A block
carries SQLite's own wording, which goes straight into the repair prompt. I validated it against SQLite itself
on all 11,727 gold queries and on 2,068 model queries. It blocked one gold query, which really does name a table
that doesn't exist, and nothing it blocked would have run. The first version did block valid queries: sqlglot
lists a subquery's unqualified columns under the outer query too, as possible correlated references. The fix
resolves each column once, from the innermost query outwards, as SQLite does. The guard checks names against
the whole database, not just the tables in the prompt. A query that would run shouldn't fail because the linker
left a table out.

**Q: What did the pipeline ablation show?**
On Spider, almost nothing moves. Linking changes EX by a tenth of a point, and repair adds 0.6–0.7 (p = 0.03 and
0.02), because under 3% of Spider answers fail to run. BIRD is where it matters. Squeezing BIRD's schemas into
the 2,048-token training length costs 3.2 points on the 9B and 2.4 on the 4B, and both are significant in a paired
test. It isn't the linker: recall is 99.8%. It's what the prompt has to leave out, mostly sample values and
descriptions. Repair then wins back 3.2 and 5.4 points and loses none, since it only touches queries that failed.
Net: the full pipeline at 2,048 tokens ties the 9B's full-schema prompt at 4,096, and beats the 4B's by 3 points
(p = 0.04). And the 9B's significant 5.8-point lead over the 4B on BIRD shrinks to 2.8, which isn't significant.

**Q: How much does self-repair actually help, and where does it stop?**
It turns about a quarter of the failures it retries into correct answers: 16 of 73 for the 9B on BIRD, 27 of 104
for the 4B. Most first failures are guard blocks for a column that doesn't exist, and the repair turn usually
fixes the name. But the query that then runs is right less than a third of the time, because fixing a name
doesn't fix the reasoning. Repair can't help a query that runs and returns wrong rows, which is most misses. Two
costs are worth knowing. Repair turns can outgrow the training length (19 of 133 for the 4B). And the 4B loops more
when pushed back on: 12 of its 18 replies that ran out of tokens came in repair turns.

**Q: If you rerun an experiment, do you get the same number?**
From the recorded replies, exactly. Regenerating them, almost. Greedy decoding repeats itself within one loaded
model, 60 of 60 in my probe. But after the server reloads the model, 14–15 of 60 replies differ, mostly in trivial
ways. Regenerating a whole 500-question BIRD run gave 480 identical queries and one flipped answer: 43.6% against
43.8%. So I treat the response cache as the record of a run. Every number in the tables re-scores recorded
replies, every comparison is paired on the same questions, and configs that share a first turn share its recorded
reply. The repair comparison measures repair and nothing else (ADR-020).
