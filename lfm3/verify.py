"""Check the LFM3 assembly runtime against the FP32 reference of reference.py.

Matrix inputs are quantized to Q8_K as in ggml. Q8_K rounding is discontinuous: a value at a
rounding boundary moves by a whole step for a tiny upstream difference, and such flips grow
through the 24 layers. End-to-end logits therefore differ from any reference by up to about one
logit, so the runtime is checked in two ways:

- block replay: each reference block, with the same Q8_K inputs, is run on the runtime's own
  hidden states for every token of a prompt and must reproduce the runtime's block output. A
  rounding flip still shows up locally: at one token, or at the tokens attending to the key and
  value it changed. A defect would show up at most tokens, so every block's median error must
  be at FP32 level. The share of (token, block) pairs above that level is compared with the
  share between the reference and itself when its inputs move by one FP32 ulp;
- FP32 reference: the final argmax must agree, and greedy continuations must be the reference
  argmax except at near ties.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference import GGUFLinear, load, last_logits, teacher  # noqa: E402

FRANCE = [124894, 124899, 5922, 207, 2992, 355, 278, 5205, 302, 3980, 39, 124900, 207, 124899,
          63514, 207, 124901, 207, 124902, 207]
CLEAN = 1e-5                # block error relative to the block's largest output, per token
FLIP_MARGIN = 0.01          # pairs above CLEAN allowed beyond twice the reference's own share
TIE_TOLERANCE = 0.25        # a greedy choice may differ from the reference argmax only within this gap
DUMPS = 49                  # embedding, then the normalized output of every block, per token


def run(binary, model, args, threads=1, context=1024):
    env = dict(os.environ, LFM_THREADS=str(threads), LFM_CONTEXT=str(context))
    return subprocess.run([binary, model, *args], capture_output=True, env=env)


def prompts(tokenizer_dir):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    text = (Path(__file__).resolve().parent / "verify_text.txt").read_text(encoding="utf8")
    body = tokenizer(text, add_special_tokens=False)["input_ids"]
    chat = tokenizer("<|startoftext|><|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n<think></think>",
                     add_special_tokens=False)["input_ids"]
    return {"BOS": [124894], "France": FRANCE, "long chat": chat, "plain 2x": [124894] + (body * 2)[:720]}


def relative(got, expected):
    return np.abs(got - expected).max(axis=1) / np.abs(expected).max(axis=1)


@torch.no_grad()
def replay(model, config, binary, gguf, ids):
    """Per (token, block) relative errors of reference blocks run on the runtime's hidden states,
    and the same errors between the reference and itself with inputs moved by one FP32 ulp:
    the rounding flips any correct FP32 implementation shows."""
    captured = {}
    hooks = [layer.register_forward_pre_hook(lambda m, a, k, n=n: captured.__setitem__(n, k), with_kwargs=True)
             for n, layer in enumerate(model.model.layers)]
    model(torch.tensor([ids]), use_cache=False)
    for hook in hooks:
        hook.remove()
    result = run(binary, gguf, [str(i) for i in ids])
    dumps = np.frombuffer(result.stdout, dtype=np.float32)[:len(ids) * DUMPS * 2048]
    dumps = dumps.reshape(len(ids), DUMPS, 2048)
    hidden = dumps[:, 0].copy()
    errors, floor = {}, {}
    generator = np.random.default_rng(0)

    def block(layer, kwargs, part, h):
        h = torch.from_numpy(h.copy())[None]
        if part == "mlp":
            y = layer.post_ffn_norm(layer.feed_forward(layer.ffn_norm(h)))
        elif layer.is_attention_layer:
            y, _ = layer.self_attn(hidden_states=layer.operator_norm(h),
                                   position_embeddings=kwargs["position_embeddings"],
                                   attention_mask=kwargs["attention_mask"],
                                   position_ids=kwargs.get("position_ids"))
            y = layer.post_operator_norm(y)
        else:
            y = layer.post_operator_norm(layer.conv(hidden_states=layer.operator_norm(h),
                                                    attention_mask=kwargs["attention_mask"]))
        return y[0].numpy()

    for n, layer in enumerate(model.model.layers):
        for part in ("operator", "mlp"):
            expected = block(layer, captured[n], part, hidden)
            nudge = generator.choice([-1.0, 1.0], size=hidden.shape).astype(np.float32)
            moved = np.nextafter(hidden, hidden + nudge * np.float32(np.inf)).astype(np.float32)
            name = f"layer {n} {config.layer_types[n]} {part}"
            got = dumps[:, 1 + 2 * n + (part == "mlp")]
            errors[name] = relative(got, expected)
            floor[name] = relative(block(layer, captured[n], part, moved), expected)
            hidden = hidden + got
    return errors, floor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("refdir")
    parser.add_argument("next")
    parser.add_argument("logits")
    parser.add_argument("layers")
    args = parser.parse_args()
    model, config = load(args.model, args.refdir)
    failures = 0
    cases = prompts(args.refdir)
    GGUFLinear.quantize_inputs = True
    for name in ("France", "long chat"):
        errors, floor = replay(model, config, args.layers, args.model, cases[name])
        values = np.concatenate(list(errors.values()))
        share = float((values > CLEAN).mean())
        intrinsic = float((np.concatenate(list(floor.values())) > CLEAN).mean())
        median, median_block = max((float(np.median(e)), block) for block, e in errors.items())
        worst, worst_block = max((float(e.max()), block) for block, e in errors.items())
        ok = median <= CLEAN and share <= 2 * intrinsic + FLIP_MARGIN
        failures += not ok
        print(f"block replay, {name}: {len(cases[name])} tokens x {len(errors)} blocks; above {CLEAN:.0e}: "
              f"runtime {100 * share:.2f}%, reference against itself one ulp apart {100 * intrinsic:.2f}%; "
              f"largest block median {median:.1e} ({median_block}); largest {worst:.1e} ({worst_block})"
              f"{'' if ok else '  FAIL'}")
    GGUFLinear.quantize_inputs = False
    for name, ids in cases.items():
        result = run(args.logits, args.model, [str(i) for i in ids])
        got = np.frombuffer(result.stdout, dtype=np.float32)
        if result.returncode or got.size != 128000:
            print(f"{name}: runtime failed ({result.returncode})")
            failures += 1
            continue
        expected = last_logits(model, ids).numpy()
        error = float(np.abs(got - expected).max())
        top_got, top_ref = np.argsort(-got)[:5], np.argsort(-expected)[:5]
        ok = top_got[0] == top_ref[0]
        failures += not ok
        print(f"FP32 logits, {name}: {len(ids)} tokens, argmax {top_got[0]} vs {top_ref[0]}, top-5 overlap "
              f"{len(set(top_got) & set(top_ref))}/5, max |logit difference| {error:.3f}{'' if ok else '  FAIL'}")
    ids = cases["long chat"]
    result = run(args.next, args.model, ["--generate", "96", *map(str, ids)])
    generated = [int(i) for i in result.stdout.split()]
    rows = teacher(model, ids + generated, len(ids))
    misses = [row for row in rows if row[1] != row[2]]
    worst = max((gap for _, _, _, gap in misses), default=0.0)
    ok = result.returncode == 0 and worst <= TIE_TOLERANCE and len(generated) > 0
    failures += not ok
    print(f"greedy after {len(ids)} prompt tokens: {len(rows) - len(misses)}/{len(rows)} tokens are the FP32 "
          f"reference argmax, largest gap at a miss {worst:.5f}{'' if ok else '  FAIL'}")
    accepted = [invalid for invalid in (["--generate", "0", "1"], ["128000"], ["--generate", "1", "-1"], ["1", "x"])
                if run(args.next, args.model, invalid).returncode == 0]
    print(f"accepted invalid arguments {accepted}" if accepted else "Invalid arguments rejected")
    failures += len(accepted)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
