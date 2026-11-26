"""Validate greedy cached decoding against Transformers and the legacy CLI."""
import argparse
import os
import subprocess
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("runtime", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint.resolve().parent, dtype=torch.float32,
        attn_implementation="eager", local_files_only=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint.resolve().parent, local_files_only=True)

    def prompt(text):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=True,
            add_generation_prompt=True, return_dict=True,
        )["input_ids"]

    def reference(tokens, count):
        with torch.inference_mode():
            output = model.generate(
                torch.tensor([tokens]), attention_mask=torch.ones((1, len(tokens)), dtype=torch.long),
                do_sample=False, max_new_tokens=count, eos_token_id=7, pad_token_id=0,
                repetition_penalty=1.0, use_cache=True,
            )
        return output[0, len(tokens):].tolist()

    def run(tokens, count, threads=6):
        options = [] if count is None else ["--generate", str(count)]
        return subprocess.run(
            [str(args.runtime.resolve()), str(args.checkpoint), *options, *map(str, tokens)],
            env=dict(os.environ, LFM_THREADS=str(threads)),
            capture_output=True, text=True, timeout=180,
        )

    france = prompt("What is the capital of France?")
    counting = prompt("Count from 1 to 30, putting each number on its own line.")
    boundary = (france * 32)[:-1]
    cases = (("eos", france, 32), ("limit", france, 3),
             ("long", counting, 64), ("cache_boundary", boundary, 2))
    for label, tokens, count in cases:
        expected = reference(tokens, count)
        actual = run(tokens, count)
        if actual.returncode or list(map(int, actual.stdout.split())) != expected:
            raise AssertionError((label, expected, actual.stdout, actual.stderr))
        if label == "eos" and (expected[-1] != 7 or len(expected) >= count):
            raise AssertionError("EOS case did not exercise early termination")
        if label in ("limit", "long", "cache_boundary") and len(expected) != count:
            raise AssertionError(f"{label} did not exercise requested length")
        if label == "long":
            single = run(tokens, count, threads=1)
            if single.returncode or single.stdout != actual.stdout:
                raise AssertionError("Generated sequence depends on thread count")
        print(f"{label}: {len(tokens)} prompt tokens, {len(expected)} generated tokens match", flush=True)
        if label == "eos":
            print(tokenizer.decode(expected, skip_special_tokens=True), flush=True)

    legacy = run(france, None)
    explicit = run(france, 1)
    if legacy.returncode or explicit.returncode or legacy.stdout != explicit.stdout:
        raise AssertionError("Legacy one-token CLI changed")
    invalid = (
        (france, "0"), (france, "-1"), (france, ""), (france, "2x"),
        (france, "9999999999999999999"), (france * 32, 2),
        ([], 1), ([1, "bad"], 3),
    )
    for tokens, count in invalid:
        actual = run(tokens, count)
        if actual.returncode != 1 or actual.stdout:
            raise AssertionError(f"Invalid request accepted: {count}, {tokens[:3]}")
    print("Legacy CLI and argument validation passed", flush=True)


if __name__ == "__main__":
    main()
