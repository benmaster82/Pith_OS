"""FP32 reference for an LFM3 GGUF checkpoint through the official modeling_lfm3.py.

The official architecture is built without weights; every Linear and the embedding dequantize
their GGUF tensor when called, so only one matrix is resident at a time. Norms and convolution
taps are F32 in the GGUF and are loaded directly. The checks below fail on any tensor that is
unused, misshapen, or disagrees with config.json.

usage: reference.py GGUF REFDIR check
       reference.py GGUF REFDIR logits OUT ID [ID ...]      last-position FP32 logits
       reference.py GGUF REFDIR generate N ID [ID ...]      greedy IDs, one per line
       reference.py GGUF REFDIR chat N "prompt"             template IDs, then greedy IDs
       reference.py GGUF REFDIR teacher P ID [ID ...]       check a greedy continuation

Each forward dequantizes the whole checkpoint, so greedy generation costs one pass per token.
teacher scores another engine's output in a single pass: the first P IDs are the prompt, the
rest its greedy continuation, and every continuation token should be the reference argmax.
"""
import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import init_empty_weights
from gguf import GGUFReader
from gguf.quants import dequantize
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


class Checkpoint:
    def __init__(self, path):
        self.reader = GGUFReader(path)
        self.tensors = {t.name: t for t in self.reader.tensors}
        self.used = set()

    def field(self, key):
        part = self.reader.fields[key]
        values = [part.parts[i] for i in part.data]
        if part.types[0].name == "STRING":
            return bytes(values[0]).decode()
        flat = [v.tolist()[0] if len(v) == 1 else v.tolist() for v in values]
        return flat if part.types[0].name == "ARRAY" else flat[0]

    def shape(self, name):
        self.used.add(name)
        return tuple(int(n) for n in reversed(self.tensors[name].shape))

    def rows(self, name, index=None):
        """Dequantized rows [out][in] as FP32, optionally only the given row indices."""
        self.used.add(name)
        tensor = self.tensors[name]
        data = tensor.data if index is None else tensor.data[index]
        return torch.from_numpy(np.array(dequantize(data, tensor.tensor_type), dtype=np.float32))


def q8k(x):
    """Round-trip through ggml's Q8_K: per 256 values, scale 127 / max |x|, round half to even."""
    blocks = x.reshape(*x.shape[:-1], -1, 256)
    peak = blocks.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(peak > 0, torch.tensor(127.0) / peak, torch.zeros_like(peak))
    step = torch.where(peak > 0, 1 / scale, torch.zeros_like(peak))
    return (torch.round(blocks * scale) * step).reshape(x.shape)


class GGUFLinear(torch.nn.Module):
    quantize_inputs = False     # reproduce the runtime's Q8_K activations

    def __init__(self, checkpoint, name, chunk=None):
        super().__init__()
        self.checkpoint, self.name, self.chunk = checkpoint, name, chunk
        self.out_features, self.in_features = checkpoint.shape(name)

    def forward(self, x):
        if self.quantize_inputs:
            x = q8k(x)
        if self.chunk is None:
            return F.linear(x, self.checkpoint.rows(self.name))
        parts = [F.linear(x, self.checkpoint.rows(self.name, slice(i, i + self.chunk)))
                 for i in range(0, self.out_features, self.chunk)]
        return torch.cat(parts, dim=-1)


class GGUFEmbedding(torch.nn.Module):
    def __init__(self, checkpoint, name):
        super().__init__()
        self.checkpoint, self.name = checkpoint, name
        checkpoint.used.add(name)

    def forward(self, ids):
        flat = ids.reshape(-1).tolist()
        unique = sorted(set(flat))
        table = self.checkpoint.rows(self.name, unique)
        lookup = {token: i for i, token in enumerate(unique)}
        return table[[lookup[t] for t in flat]].reshape(*ids.shape, -1)


LINEAR = {
    "ffn_gate": "feed_forward.w1", "ffn_up": "feed_forward.w3", "ffn_down": "feed_forward.w2",
    "shortconv.in_proj": "conv.in_proj", "shortconv.out_proj": "conv.out_proj",
    "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj", "attn_gate": "self_attn.gate_proj",
}
VECTOR = {
    "attn_norm": "operator_norm.weight", "ffn_norm": "ffn_norm.weight",
    "post_attention_norm": "post_operator_norm.weight", "post_ffw_norm": "post_ffn_norm.weight",
    "attn_q_norm": "self_attn.q_norm.weight", "attn_k_norm": "self_attn.k_norm.weight",
    "shortconv.conv": "conv.conv.weight",
}


def check_config(checkpoint, config):
    arch = checkpoint.field("general.architecture")
    get = lambda key: checkpoint.field(f"{arch}.{key}")
    assert arch == "lfm3"
    assert get("block_count") == config.num_hidden_layers
    assert get("embedding_length") == config.hidden_size
    assert get("feed_forward_length") == config.intermediate_size
    assert get("attention.head_count") == config.num_attention_heads
    assert get("attention.key_length") == get("attention.value_length") == config.head_dim
    assert get("shortconv.l_cache") == config.conv_L_cache
    assert get("attention.sliding_window") == config.sliding_window
    assert abs(get("attention.layer_norm_rms_epsilon") - config.norm_eps) < 1e-9
    kv = get("attention.head_count_kv")
    swa = get("attention.sliding_window_pattern")
    for i, kind in enumerate(config.layer_types):
        assert kv[i] == (0 if kind == "conv" else config.num_key_value_heads), (i, kind)
        assert bool(swa[i]) == (kind == "sliding_attention"), (i, kind)
    rope = config.rope_parameters
    assert get("rope.freq_base") == rope["full_attention"]["rope_theta"]
    assert get("rope.freq_base_swa") == rope["sliding_attention"]["rope_theta"]
    # rope_freqs divides the full-dimension progression; a huge divisor is the zero tail.
    rotated = int(rope["full_attention"]["partial_rotary_factor"] * config.head_dim) // 2
    divisors = checkpoint.rows("rope_freqs.weight").reshape(-1)
    assert torch.all(divisors[:rotated] == 1) and torch.all(divisors[rotated:] > 1e20), divisors
    assert len(checkpoint.field("tokenizer.ggml.tokens")) == config.vocab_size
    assert checkpoint.field("tokenizer.ggml.bos_token_id") == config.bos_token_id
    assert checkpoint.field("tokenizer.ggml.eos_token_id") == config.eos_token_id


def load(gguf_path, refdir):
    checkpoint = Checkpoint(gguf_path)
    config = AutoConfig.from_pretrained(refdir, trust_remote_code=True)
    config._attn_implementation = "eager"
    check_config(checkpoint, config)
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, dtype=torch.float32)
    base = model.model
    base.embed_tokens = GGUFEmbedding(checkpoint, "token_embd.weight")
    model.lm_head = GGUFLinear(checkpoint, "token_embd.weight", chunk=8192)
    base.embedding_norm.weight = torch.nn.Parameter(checkpoint.rows("output_norm.weight").reshape(-1))
    for i, layer in enumerate(base.layers):
        prefix = f"blk.{i}."
        for gguf_name, path in LINEAR.items():
            name = prefix + gguf_name + ".weight"
            if name in checkpoint.tensors:
                parent, _, attribute = path.rpartition(".")
                module = layer.get_submodule(parent)
                old = getattr(module, attribute)
                new = GGUFLinear(checkpoint, name)
                assert (new.out_features, new.in_features) == (old.out_features, old.in_features), name
                setattr(module, attribute, new)
        for gguf_name, path in VECTOR.items():
            name = prefix + gguf_name + ".weight"
            if name in checkpoint.tensors:
                parent, _, attribute = path.rpartition(".")
                module = layer.get_submodule(parent)
                old = getattr(module, attribute)
                value = checkpoint.rows(name).reshape(old.shape)
                setattr(module, attribute, torch.nn.Parameter(value))
    leftover = [n for n, p in model.named_parameters() if p.is_meta]
    assert not leftover, leftover
    unused = set(checkpoint.tensors) - checkpoint.used
    assert not unused, unused
    return model.eval(), config


@torch.no_grad()
def last_logits(model, ids):
    return model(torch.tensor([ids]), use_cache=False).logits[0, -1].float()


@torch.no_grad()
def greedy(model, ids, count, eos):
    output = model(torch.tensor([ids]), use_cache=True)
    cache, generated = output.past_key_values, []
    while True:
        token = int(output.logits[0, -1].argmax())
        generated.append(token)
        if token == eos or len(generated) == count:
            return generated
        output = model(torch.tensor([[token]]), past_key_values=cache, use_cache=True)
        cache = output.past_key_values


@torch.no_grad()
def teacher(model, ids, prompt):
    """Per continuation token: the reference argmax and the logit gap to the chosen token."""
    logits = model(torch.tensor([ids]), use_cache=False).logits[0].float()
    rows = []
    for position in range(prompt, len(ids)):
        row = logits[position - 1]
        best = int(row.argmax())
        rows.append((position - prompt, ids[position], best, float(row[best] - row[ids[position]])))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf")
    parser.add_argument("refdir")
    parser.add_argument("action", choices=("check", "logits", "generate", "chat", "teacher"))
    parser.add_argument("rest", nargs="*")
    parser.add_argument("--q8k", action="store_true", help="quantize matrix inputs to Q8_K like the runtime")
    args = parser.parse_args()
    GGUFLinear.quantize_inputs = args.q8k
    model, config = load(args.gguf, args.refdir)
    if args.action == "teacher":
        rows = teacher(model, [int(i) for i in args.rest[1:]], int(args.rest[0]))
        misses = [row for row in rows if row[1] != row[2]]
        for step, chosen, best, gap in misses:
            print(f"step {step}: engine {chosen}, reference {best}, logit gap {gap:.5f}")
        print(f"{len(rows) - len(misses)}/{len(rows)} continuation tokens are the reference argmax")
        return 1 if misses else 0
    if args.action == "check":
        print("GGUF matches config.json; all tensors mapped onto the official architecture")
    elif args.action == "logits":
        last_logits(model, [int(i) for i in args.rest[1:]]).numpy().tofile(args.rest[0])
    elif args.action == "generate":
        for token in greedy(model, [int(i) for i in args.rest[1:]], int(args.rest[0]), config.eos_token_id):
            print(token)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.refdir)
        text = tokenizer.apply_chat_template([{"role": "user", "content": args.rest[1]}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        print("prompt", " ".join(map(str, ids)))
        tokens = greedy(model, ids, int(args.rest[0]), config.eos_token_id)
        print("ids", " ".join(map(str, tokens)))
        print("text", repr(tokenizer.decode(tokens)))


if __name__ == "__main__":
    sys.exit(main())
