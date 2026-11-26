"""Generate NASM offsets for an official BF16 safetensors checkpoint."""
import argparse
import json
import math
import struct
from pathlib import Path

D = 1024
FF = 4608
ATTENTION = {2, 5, 8, 10, 12, 14}


def layout():
    yield "EMBED", "model.embed_tokens.weight", [65536, D]
    yield "FINAL", "model.embedding_norm.weight", [D]
    for i in range(16):
        fields = [
            ("OP_NORM", "operator_norm", [D]),
            ("FF_NORM", "ffn_norm", [D]),
            ("W1", "feed_forward.w1", [FF, D]),
            ("W2", "feed_forward.w2", [D, FF]),
            ("W3", "feed_forward.w3", [FF, D]),
        ]
        if i in ATTENTION:
            fields += [
                ("Q", "self_attn.q_proj", [D, D]),
                ("K", "self_attn.k_proj", [512, D]),
                ("V", "self_attn.v_proj", [512, D]),
                ("OUT", "self_attn.out_proj", [D, D]),
                ("QN", "self_attn.q_layernorm", [64]),
                ("KN", "self_attn.k_layernorm", [64]),
            ]
        else:
            fields += [
                ("IN", "conv.in_proj", [3 * D, D]),
                ("CONV", "conv.conv", [D, 1, 3]),
                ("OUT", "conv.out_proj", [D, D]),
            ]
        for tag, name, shape in fields:
            yield f"L{i}_{tag}", f"model.layers.{i}.{name}.weight", shape


def read_checkpoint(path):
    size = path.stat().st_size
    with path.open("rb") as stream:
        length, = struct.unpack("<Q", stream.read(8))
        if not 0 < length <= min(size - 8, 16_000_000):
            raise ValueError("Invalid safetensors header length")
        raw_header = stream.read(length)
        header = json.loads(raw_header)
    data = length + 8
    checksum = 2166136261
    for byte in raw_header:
        checksum = ((checksum ^ byte) * 16777619) & 0xffffffff
    offsets = []
    for tag, name, shape in layout():
        tensor = header[name]
        start, end = tensor["data_offsets"]
        if (tensor["shape"] != shape or tensor["dtype"] != "BF16"
                or not 0 <= start <= end <= size - data
                or end - start != 2 * math.prod(shape)):
            raise ValueError(f"Incompatible tensor {name}")
        offsets.append(data + start)
    return size, length, checksum, offsets


def generate(path):
    size, length, checksum, offsets = read_checkpoint(path)
    lines = [f"%define FILE_SIZE {size}", f"%define HEADER_SIZE {length}",
             f"%define HEADER_HASH 0x{checksum:08x}"]
    for (tag, _, _), offset in zip(layout(), offsets):
        lines.append(f"%define {tag} {offset}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(generate(args.model), encoding="ascii")
