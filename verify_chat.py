"""Check retained chat state, reset, rejected turns and configurable KV capacity."""
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
    runtime = str(args.runtime.resolve())
    torch.set_num_threads(4)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint.resolve().parent, dtype=torch.float32,
        attn_implementation="eager", local_files_only=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint.resolve().parent, local_files_only=True)

    def generate(ids, count):
        with torch.inference_mode():
            output = model.generate(
                torch.tensor([ids]), attention_mask=torch.ones((1, len(ids)), dtype=torch.long),
                do_sample=False, max_new_tokens=count, eos_token_id=7, pad_token_id=0,
                repetition_penalty=1.0, use_cache=True,
            )
        return output[0, len(ids):].tolist()

    def expected_chat(turns, count):
        history, responses = [], []
        for text in turns:
            if text is None:
                history = []
                continue
            segment = ("" if history else "<|startoftext|>") + (
                "<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n")
            history += tokenizer.encode(segment, add_special_tokens=False)
            generated = generate(history, count)
            responses.append(tokenizer.decode(generated, skip_special_tokens=True).encode("utf8") + b"\n")
            history += generated
            if history[-1] != 7:
                history.append(7)
            history.append(708)
        return b"".join(responses)

    def run(arguments, data=b"", context="128"):
        return subprocess.run(
            [runtime, str(args.checkpoint), *map(str, arguments)], input=data,
            env=dict(os.environ, LFM_CONTEXT=context, LFM_THREADS="6"),
            capture_output=True, timeout=180,
        )

    france = "What is the capital of France?"
    turns = [france, "And of Italy?", "And of Germany?", None, france]
    expected = expected_chat(turns, 16)
    lines = [
        b"", france.encode(), b"/unknown", b"\xff", b"\0bad",
        b"a" * 9000, b"x " * 200, b"word " * 90,
        b"And of Italy?\r", b"And of Germany?", b"/clear\r", france.encode(),
    ]
    actual = run(["--chat", 16], b"\n".join(lines))  # final line ends at EOF
    if actual.returncode or actual.stdout != expected:
        raise AssertionError(("chat state", expected, actual.stdout, actual.stderr))
    for notice in (b"Input rejected", b"Context full", b"Unknown command", b"Conversation cleared"):
        if notice not in actual.stderr:
            raise AssertionError(("Missing recovery notice", notice, actual.stderr))
    print("Four turns, reset, rejected inputs, CRLF and partial-line EOF match Transformers", flush=True)

    expected = expected_chat([france, "And of Italy?"], 2)
    actual = run(["--chat", 2], (france + "\nAnd of Italy?\n/quit\nignored\n").encode())
    if actual.returncode or actual.stdout != expected or actual.stderr.count(b"token limit") != 2:
        raise AssertionError(("truncated turn", expected, actual.stdout, actual.stderr))
    print("Truncated turns, synthetic EOS and /quit passed", flush=True)

    exact = run(["--chat", 1],
                (france + "\n" + france + "\n/clear\n" + france + "\n/quit\n").encode(),
                context="19")
    expected = expected_chat([france, None, france], 1)
    if exact.returncode or exact.stdout != expected or b"Context full" not in exact.stderr:
        raise AssertionError(("Exact cache capacity", exact.stdout, exact.stderr))
    print("Exact cache capacity and reset after exhaustion passed", flush=True)

    # Cross the previous 512-position limit using the ordinary numeric entry point.
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": france}], tokenize=True,
        add_generation_prompt=True, return_dict=True,
    )["input_ids"] * 32 + [1]
    expected_ids = generate(prompt, 2)
    actual = run(["--generate", 2, *prompt], context="1024")
    if actual.returncode or list(map(int, actual.stdout.split())) != expected_ids:
        raise AssertionError(("extended context", expected_ids, actual.stdout, actual.stderr))
    rejected = run(["--generate", 2, *prompt], context="512")
    if rejected.returncode != 1 or rejected.stdout:
        raise AssertionError("512-position limit not enforced")
    print("513-token prompt with 1024-position cache matches Transformers", flush=True)

    text = "a" * 4096
    encoded = run(["--encode", text], context="1024")
    expected_ids = tokenizer.encode(text, add_special_tokens=False)
    if encoded.returncode or list(map(int, encoded.stdout.split())) != expected_ids:
        raise AssertionError("Tokenizer did not use configured context limit")

    for context in ("", "0", "-1", "32769", "1x"):
        result = run([1], context=context)
        if result.returncode != 1 or result.stdout:
            raise AssertionError(("Invalid context accepted", context))
    for context in ("1", "32768"):
        result = run([1], context=context)
        if result.returncode or result.stdout != b"1\n":
            raise AssertionError(("Context boundary", context, result))
    empty = run(["--chat"], b"/quit\n")
    if empty.returncode or empty.stdout:
        raise AssertionError("Empty chat failed")
    print("Context boundaries and invalid configuration passed", flush=True)


if __name__ == "__main__":
    main()
