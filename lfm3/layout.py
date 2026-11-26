"""Generate NASM offsets and the layer list for an LFM3 GGUF checkpoint."""
import argparse
import math
import struct
from pathlib import Path

D, FF, HEADS, KV_HEADS, HEAD_DIM, VOCAB, CONV_TAPS, WINDOW = 2048, 7680, 16, 4, 128, 128000, 7, 256
F32, Q4_K, Q6_K = 0, 12, 14
BLOCK_BYTES = {Q4_K: 144, Q6_K: 210}
SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


class Reader:
    def __init__(self, data):
        self.data, self.at = data, 0

    def take(self, fmt):
        values = struct.unpack_from("<" + fmt, self.data, self.at)
        self.at += struct.calcsize("<" + fmt)
        return values[0]

    def string(self):
        size = self.take("Q")
        self.at += size
        return self.data[self.at - size:self.at].decode("utf8")

    def value(self, kind):
        if kind in SCALAR:
            return self.take(SCALAR[kind])
        if kind == 8:
            return self.string()
        if kind == 9:
            item, count = self.take("I"), self.take("Q")
            return [self.value(item) for _ in range(count)]
        raise ValueError(f"Unknown GGUF value type {kind}")


def read(path):
    size = path.stat().st_size
    with path.open("rb") as stream:
        head = stream.read(min(size, 64 << 20))
    reader = Reader(head)
    if reader.take("I") != 0x46554747 or reader.take("I") != 3:
        raise ValueError("Not a GGUF version 3 file")
    tensors, fields = reader.take("Q"), reader.take("Q")
    meta = {}
    for _ in range(fields):
        key = reader.string()
        meta[key] = reader.value(reader.take("I"))
    infos = {}
    for _ in range(tensors):
        name = reader.string()
        dims = [reader.take("Q") for _ in range(reader.take("I"))]
        kind, offset = reader.take("I"), reader.take("Q")
        infos[name] = (dims, kind, offset)
    alignment = meta.get("general.alignment", 32)
    data = -(-reader.at // alignment) * alignment
    return size, data, head[:data], meta, infos


def expected(meta):
    kinds = []
    for i in range(24):
        if meta["lfm3.attention.head_count_kv"][i] == 0:
            kinds.append("conv")
        else:
            kinds.append("swa" if meta["lfm3.attention.sliding_window_pattern"][i] else "full")
    yield "EMBED", "token_embd.weight", [D, VOCAB], (Q6_K,)
    yield "FINAL", "output_norm.weight", [D], (F32,)
    yield "ROPE_FREQS", "rope_freqs.weight", [HEAD_DIM // 2], (F32,)
    for i, kind in enumerate(kinds):
        fields = [("OP_NORM", "attn_norm", [D], (F32,)), ("FF_NORM", "ffn_norm", [D], (F32,)),
                  ("POST_OP", "post_attention_norm", [D], (F32,)), ("POST_FF", "post_ffw_norm", [D], (F32,)),
                  ("W1", "ffn_gate", [D, FF], (Q4_K, Q6_K)), ("W3", "ffn_up", [D, FF], (Q4_K, Q6_K)),
                  ("W2", "ffn_down", [FF, D], (Q4_K, Q6_K))]
        if kind == "conv":
            fields += [("IN", "shortconv.in_proj", [D, 3 * D], (Q4_K, Q6_K)),
                       ("CONV", "shortconv.conv", [CONV_TAPS, D], (F32,)),
                       ("OUT", "shortconv.out_proj", [D, D], (Q4_K, Q6_K))]
        else:
            fields += [("Q", "attn_q", [D, HEADS * HEAD_DIM], (Q4_K, Q6_K)),
                       ("K", "attn_k", [D, KV_HEADS * HEAD_DIM], (Q4_K, Q6_K)),
                       ("V", "attn_v", [D, KV_HEADS * HEAD_DIM], (Q4_K, Q6_K)),
                       ("GATE", "attn_gate", [D, HEADS], (Q4_K, Q6_K)),
                       ("OUT", "attn_output", [HEADS * HEAD_DIM, D], (Q4_K, Q6_K)),
                       ("QN", "attn_q_norm", [HEAD_DIM], (F32,)), ("KN", "attn_k_norm", [HEAD_DIM], (F32,))]
        for tag, name, shape, types in fields:
            yield f"L{i}_{tag}", f"blk.{i}.{name}.weight", shape, types
    yield None, kinds, None, None


def generate(path):
    size, data, header, meta, infos = read(path)
    checks = {"general.architecture": "lfm3", "lfm3.block_count": 24, "lfm3.embedding_length": D,
              "lfm3.feed_forward_length": FF, "lfm3.attention.head_count": HEADS,
              "lfm3.attention.key_length": HEAD_DIM, "lfm3.attention.value_length": HEAD_DIM,
              "lfm3.rope.dimension_count": HEAD_DIM, "lfm3.attention.sliding_window": WINDOW,
              "lfm3.shortconv.l_cache": CONV_TAPS, "lfm3.rope.freq_base": 1e6, "lfm3.rope.freq_base_swa": 1e4}
    for key, value in checks.items():
        if meta.get(key) != value:
            raise ValueError(f"Unsupported checkpoint: {key} = {meta.get(key)!r}")
    if any(n not in (0, KV_HEADS) for n in meta["lfm3.attention.head_count_kv"]):
        raise ValueError("Unsupported key/value head count")
    if len(meta["tokenizer.ggml.tokens"]) != VOCAB:
        raise ValueError("Unsupported vocabulary size")
    if abs(meta["lfm3.attention.layer_norm_rms_epsilon"] - 1e-5) > 1e-9:
        raise ValueError("Unsupported normalization epsilon")
    lines = [f"%define FILE_SIZE {size}", f"%define HEADER_SIZE {data}",
             f"%define HEADER_HASH 0x{fnv1a(header):08x}",
             f"%define BOS_ID {meta['tokenizer.ggml.bos_token_id']}",
             f"%define EOS_ID {meta['tokenizer.ggml.eos_token_id']}",
             f"%define NEWLINE_ID {meta['tokenizer.ggml.tokens'].index(chr(0x10a))}"]
    used = set()
    for tag, name, shape, types in expected(meta):
        if tag is None:
            kinds = name
            break
        if name not in infos:
            raise ValueError(f"Missing tensor {name}")
        dims, kind, offset = infos[name]
        if dims != shape or kind not in types:
            raise ValueError(f"Incompatible tensor {name}: {dims} type {kind}")
        count = math.prod(dims)
        length = 4 * count if kind == F32 else count // 256 * BLOCK_BYTES[kind]
        if offset % 32 or data + offset + length > size:
            raise ValueError(f"Tensor {name} outside the file")
        lines.append(f"%define {tag} {data + offset}")
        if kind != F32:
            lines.append(f"%define {tag}_TYPE {kind}")
        used.add(name)
    if used != set(infos):
        raise ValueError(f"Unexpected tensors: {sorted(set(infos) - used)}")
    layers = []
    for i, kind in enumerate(kinds):
        layers.append(f"    CONV_LAYER {i}" if kind == "conv" else f"    ATT_LAYER {i},{int(kind == 'swa')}")
        layers.append(f"    MLP {i}")
    attention = [i for i, kind in enumerate(kinds) if kind != "conv"]
    lines += [f"%define ATT_LAYERS {len(attention)}", f"%define CONV_LAYERS {kinds.count('conv')}"]
    return "\n".join(lines) + "\n", "\n".join(layers) + "\n"


def fnv1a(data):
    value = 2166136261
    for byte in data:
        value = ((value ^ byte) * 16777619) & 0xffffffff
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("layout", type=Path)
    parser.add_argument("layers", type=Path)
    args = parser.parse_args()
    layout, layers = generate(args.model)
    args.layout.write_text(layout, encoding="ascii")
    args.layers.write_text(layers, encoding="ascii")
