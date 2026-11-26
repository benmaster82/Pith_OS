"""Compile the official tokenizer into read-only tables for the NASM runtime."""
import argparse
import json
import struct
from pathlib import Path

from tokenizers import Regex, normalizers

def fnv1a(data):
    value = 2166136261
    for byte in data:
        value = ((value ^ byte) * 16777619) & 0xffffffff
    return value


def build(source, destination, include, vocab_size=65536, hash_bits=17):
    HASH_SIZE = 1 << hash_bits
    tokenizer = json.loads(source.read_text(encoding="utf8"))
    model = tokenizer["model"]
    if (tokenizer["normalizer"] is not None or model["type"] != "BPE"
            or model["dropout"] is not None or model["byte_fallback"]):
        raise ValueError("Unsupported tokenizer configuration")
    if len(model["merges"]) * 3 > HASH_SIZE * 2:
        raise ValueError("Merge table too small for the BPE merges")
    expected_regex = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    stages = tokenizer["pre_tokenizer"]["pretokenizers"]
    if (stages[0]["pattern"].get("Regex") != expected_regex
            or stages[0]["behavior"] != "Isolated" or stages[0]["invert"]
            or stages[1]["type"] != "ByteLevel" or stages[1]["add_prefix_space"]
            or stages[1]["use_regex"]):
        raise ValueError("Unsupported pre-tokenizer")

    vocabulary = model["vocab"]
    direct = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    byte_chars = {byte: chr(byte) for byte in direct}
    extra = 256
    for byte in range(256):
        if byte not in byte_chars:
            byte_chars[byte] = chr(extra)
            extra += 1
    reverse = {value: key for key, value in byte_chars.items()}
    if max(token["id"] for token in tokenizer["added_tokens"]) >= vocab_size:
        raise ValueError("Token IDs exceed the configured vocabulary")
    token_bytes = [b""] * vocab_size
    special = bytearray(vocab_size)
    for value, index in vocabulary.items():
        token_bytes[index] = bytes(reverse[char] for char in value)
    # Trie nodes contain first-child, next-sibling, byte, and terminal ID.
    nodes = [[0, 0, 0, 0xffffffff]]
    for token in tokenizer["added_tokens"]:
        if token["single_word"] or token["lstrip"] or token["rstrip"]:
            raise ValueError("Unsupported added-token boundary rule")
        index = token["id"]
        token_bytes[index] = token["content"].encode("utf8")
        special[index] = int(token["special"])
        current = 0
        for byte in token_bytes[index]:
            child = nodes[current][0]
            while child and nodes[child][2] != byte:
                child = nodes[child][1]
            if not child:
                child = len(nodes)
                nodes.append([0, nodes[current][0], byte, 0xffffffff])
                nodes[current][0] = child
            current = child
        nodes[current][3] = index

    merges = bytearray(HASH_SIZE * 16)
    for rank, (left, right) in enumerate(model["merges"]):
        left_id, right_id = vocabulary[left], vocabulary[right]
        slot = ((left_id * 0x9e3779b1) ^ (right_id * 0x85ebca6b)) & (HASH_SIZE - 1)
        while struct.unpack_from("<I", merges, slot * 16 + 8)[0]:
            if struct.unpack_from("<II", merges, slot * 16) == (left_id, right_id):
                raise ValueError("Duplicate BPE pair")
            slot = (slot + 1) & (HASH_SIZE - 1)
        struct.pack_into("<IIII", merges, slot * 16, left_id, right_id,
                         rank + 1, vocabulary[left + right])

    # Use the same Oniguruma engine as the tokenizer for Unicode properties.
    characters = "".join(chr(cp) for cp in range(0x110000) if not 0xd800 <= cp <= 0xdfff)
    flags = bytearray(0x110000)
    for pattern, flag in ((r"\p{L}", 1), (r"\p{N}", 2), (r"\s", 4)):
        transformed = normalizers.Replace(Regex(pattern), "\uffff").normalize_str(characters)
        if len(transformed) != len(characters):
            raise ValueError("Unicode property replacement changed character count")
        for char, replacement in zip(characters, transformed):
            if replacement == "\uffff" and char != "\uffff":
                flags[ord(char)] |= flag

    blob = bytearray()
    records = bytearray()
    for value in token_bytes:
        records.extend(struct.pack("<II", len(blob), len(value)))
        blob.extend(value)
    sections = {
        "TOK_BYTE_IDS": struct.pack("<256I", *(vocabulary[byte_chars[i]] for i in range(256))),
        "TOK_MERGES": merges,
        "TOK_TRIE": b"".join(struct.pack("<4I", *node) for node in nodes),
        "TOK_FLAGS": flags,
        "TOK_RECORDS": records,
        "TOK_SPECIAL": special,
        "TOK_BYTES": blob,
    }
    definitions = [f"%define TOK_HASH_MASK {HASH_SIZE - 1}"]
    if model["ignore_merges"]:
        # A pre-token present whole in the vocabulary is emitted without merges.
        # Open addressing on FNV-1a of the token bytes; entries hold ID + 1.
        word_size = 1 << (2 * len(vocabulary) - 1).bit_length()
        words = bytearray(word_size * 4)
        for index in vocabulary.values():
            slot = fnv1a(token_bytes[index]) & (word_size - 1)
            while struct.unpack_from("<I", words, slot * 4)[0]:
                slot = (slot + 1) & (word_size - 1)
            struct.pack_into("<I", words, slot * 4, index + 1)
        sections["TOK_WORDS"] = words
        definitions += ["%define TOK_IGNORE_MERGES 1", f"%define TOK_WORD_MASK {word_size - 1}"]
    output = bytearray(b"LFMTOK01")
    for name, contents in sections.items():
        output.extend(b"\0" * (-len(output) % 16))
        definitions.append(f"%define {name} {len(output)}")
        output.extend(contents)
    checksum = 2166136261
    for byte in output:
        checksum = ((checksum ^ byte) * 16777619) & 0xffffffff
    definitions.extend([f"%define TOK_SIZE {len(output)}", f"%define TOK_CHECKSUM 0x{checksum:08x}"])
    destination.write_bytes(output)
    include.write_text("\n".join(definitions) + "\n", encoding="ascii")
    print(f"Tokenizer tables: {len(output)} bytes, {len(nodes)} trie nodes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("include", type=Path)
    parser.add_argument("--vocab", type=int, default=65536, help="token ID table size")
    parser.add_argument("--hash-bits", type=int, default=17, help="log2 of the merge table size")
    args = parser.parse_args()
    build(args.source, args.destination, args.include, args.vocab, args.hash_bits)
