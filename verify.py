"""Compare the command-line prediction with an independent FP32 forward pass."""
import argparse
import subprocess
from pathlib import Path


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("runtime", type=Path)
    parser.add_argument("logits_runtime", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint.resolve().parent, dtype=torch.float32,
        attn_implementation="eager", local_files_only=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint.resolve().parent, local_files_only=True,
    )
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is the capital of France?"}],
        add_generation_prompt=True, tokenize=True, return_dict=True,
    )["input_ids"]
    for prompt in ([1], [1, 314], [1, 314, 271, 314, 42], chat,
                   chat * 4, chat * 32):
        with torch.inference_mode():
            logits = model(torch.tensor([prompt]), use_cache=False).logits[0, -1]
        expected = int(logits.argmax())
        command = [str(args.runtime.resolve()), str(args.checkpoint), *map(str, prompt)]
        actual = int(subprocess.check_output(command, text=True))
        print(f"{len(prompt)} tokens -> assembly {actual}, reference {expected}", flush=True)
        if actual != expected:
            raise SystemExit("Token mismatch")
        command[0] = str(args.logits_runtime.resolve())
        output = subprocess.check_output(command)
        if len(output) != 65536 * 4:
            raise SystemExit("Invalid logits output length")
        actual_logits = torch.frombuffer(bytearray(output), dtype=torch.float32)
        error = (actual_logits - logits).abs().max().item()
        print(f"Maximum absolute logits error {error:.8f}")
        torch.testing.assert_close(actual_logits, logits, rtol=1e-4, atol=1e-3)
        print(f"Predicted token {tokenizer.decode([actual])!r}")

    for tokens in ([], ["-1"], ["65536"], ["1x"], [""], ["1"] * 513):
        result = subprocess.run(
            [str(args.runtime.resolve()), str(args.checkpoint), *tokens],
            capture_output=True,
        )
        if result.returncode != 1 or result.stdout:
            raise SystemExit(f"Invalid argument handling for {tokens[:3]}")
    print("Invalid token arguments rejected")


if __name__ == "__main__":
    main()
