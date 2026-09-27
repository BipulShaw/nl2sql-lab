"""Phase 0 smoke test: load Qwen3.5-4B in transformers, generate 50 tokens greedily, record VRAM.

Run each quantization in its own process so the peak-memory numbers don't mix:
    uv run python scripts/smoke_hf.py --quant nf4
    uv run python scripts/smoke_hf.py --quant int8
"""

import argparse
import json
import time
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
# Same prompt as scripts/smoke_ollama.py (PLAN §6.3 format).
SYSTEM = (
    "You are an expert SQL assistant. Given a database schema and a question, write a single SQLite SELECT "
    "query that answers the question. Use only tables and columns from the schema. Output only the SQL in a "
    "```sql code block."
)
USER = (
    "### Database: concert_singer\n### Schema\n"
    "CREATE TABLE singer (\n  singer_id INTEGER PRIMARY KEY,\n  name TEXT,\n  country TEXT\n);\n"
    "### Evidence\n(none)\n### Question\nHow many singers are from France?"
)
MESSAGES = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}]


def quantization_config(quant: str) -> BitsAndBytesConfig | None:
    if quant == "nf4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    if quant == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None  # bf16


def gib(n_bytes: int) -> float:
    return round(n_bytes / 2**30, 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant", choices=["nf4", "int8", "bf16"], required=True)
    parser.add_argument("--use-kernels", action="store_true", help="swap in Hub kernels (fla, causal-conv1d)")
    parser.add_argument(
        "--mem-fraction",
        type=float,
        help="cap PyTorch's allocator at this fraction of VRAM, so overflow raises OOM instead of spilling "
        "to system RAM (the Windows driver oversubscribes VRAM under WSL)",
    )
    args = parser.parse_args()
    if args.mem_fraction:
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    start = time.perf_counter()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=REVISION,
            dtype=torch.bfloat16,
            quantization_config=quantization_config(args.quant),
            device_map={"": 0},
            use_kernels=args.use_kernels,
        )
    except torch.OutOfMemoryError:
        # bf16 weights of the 4B are 7.85 GiB, so with --mem-fraction this is the expected outcome.
        print(
            json.dumps(
                {
                    "quant": args.quant,
                    "error": "CUDA OOM while loading weights",
                    "peak_allocated_gib": gib(torch.cuda.max_memory_allocated()),
                },
                indent=2,
            )
        )
        return
    load_s = time.perf_counter() - start
    weights_bytes = torch.cuda.memory_allocated()
    # Leaf Linear layers by name suffix: these are the candidate LoRA target modules.
    linear_suffixes = Counter(
        name.rsplit(".", 1)[-1]
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    )

    prompt = tokenizer.apply_chat_template(
        MESSAGES, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(0)
    # The first call pays one-off costs (CUDA init, Triton JIT compiling the kernels), so time the second one.
    start = time.perf_counter()
    model.generate(**inputs, max_new_tokens=50, do_sample=False)
    first_call_s = time.perf_counter() - start
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    torch.cuda.synchronize()
    gen_s = time.perf_counter() - start
    new_tokens = output[0, inputs.input_ids.shape[1] :]
    free, total = torch.cuda.mem_get_info()

    print(
        json.dumps(
            {
                "quant": args.quant,
                "use_kernels": args.use_kernels,
                "model_class": type(model).__name__,
                # num_parameters() unpacks 4-bit storage (two weights per byte); a raw numel() sum would not.
                "params_b": round(model.num_parameters() / 1e9, 2),
                "load_s": round(load_s, 1),
                "weights_gib": gib(weights_bytes),
                "peak_allocated_gib": gib(torch.cuda.max_memory_allocated()),
                "device_used_gib_incl_desktop": gib(total - free),
                "prompt_tokens": inputs.input_ids.shape[1],
                "new_tokens": len(new_tokens),
                "first_call_s": round(first_call_s, 1),
                "gen_s": round(gen_s, 2),
                "gen_tok_per_s": round(len(new_tokens) / gen_s, 1),
                "output": tokenizer.decode(new_tokens, skip_special_tokens=True),
                "prompt_tail": prompt[-60:],  # shows how enable_thinking=False is encoded in the template
                "has_vision_modules": any("visual" in name for name, _ in model.named_modules()),
                "linear_module_suffixes": dict(sorted(linear_suffixes.items())),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
