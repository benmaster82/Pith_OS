"""Compare the native tokenizer and streaming decoder with the official backend."""
import argparse
import json
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

from tokenizers import Tokenizer
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("runtime", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    reference = Tokenizer.from_file(str(args.tokenizer))
    runtime = str(args.runtime.resolve())

    def call(*arguments):
        return subprocess.run([runtime, "_", *map(str, arguments)],
                              capture_output=True, timeout=20)

    corpus = [
        "", "hello", "What is the capital of France?", "Ciao, com'è andata? È già tardi.",
        "你好世界。今日は晴れです。", "مرحبا بالعالم", "שלום עולם", "Привет мир",
        "👩🏽‍💻🙂🚀 hello", "e\u0301 café naïve", "１２３４５６ 123456789 Ⅻ²½",
        "I'M we're they've he'll she'd don't 'S 'T 'RE 'VE 'M 'LL 'D 'ſ",
        " \t\n\r\n  word\t \n", "   ", " word ", "  word", "\n\tword",
        "!?\r\nabc", "\u0085\u00a0\u1680\u2003\u2028\u2029\u202f\u3000text",
        "\x1c\x1d\x1e\x1f", "pythonic MathiasMathias", "<think>hello</think>",
        "<|im_start|>user\nhello<|im_end|>\n", " " * 4096, "a" * 4096,
    ]
    metadata = json.loads(args.tokenizer.read_text(encoding="utf8"))
    corpus += [item["content"] for item in metadata["added_tokens"]]
    rng = random.Random(781)
    alphabet = list(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'!?.,\n\r\t")
    alphabet += list("éÈßſ中日あア한Жمرحبا🙂") + ["\u0301", "\u00a0", "\u2028", "\x1c"]
    corpus += ["".join(rng.choices(alphabet, k=rng.randrange(1, 100))) for _ in range(160)]
    # Sample the full Unicode range, excluding surrogates and NUL in argv.
    for _ in range(80):
        values = [rng.randrange(1, 0x110000) for _ in range(12)]
        corpus.append("".join(chr(cp) for cp in values if not 0xd800 <= cp <= 0xdfff))
    for index, text in enumerate(corpus):
        expected = reference.encode(text, add_special_tokens=False).ids
        actual = call("--encode", text)
        if len(expected) > 512:
            if actual.returncode != 1 or actual.stdout:
                raise AssertionError("Token capacity not enforced")
            continue
        if actual.returncode or list(map(int, actual.stdout.split())) != expected:
            raise AssertionError(("encode", index, repr(text[:100]), expected[:100], actual.stdout[:500], actual.stderr))
    print(f"{len(corpus)} encoding cases passed, including capacity limits", flush=True)

    # Byte-level vocabulary IDs exercise malformed/truncated and split UTF-8.
    direct = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    byte_chars = {byte: chr(byte) for byte in direct}
    extra = 256
    for byte in range(256):
        if byte not in byte_chars:
            byte_chars[byte] = chr(extra)
            extra += 1
    byte_ids = [metadata["model"]["vocab"][byte_chars[i]] for i in range(256)]
    cases = [[token] for token in byte_ids]
    cases += [[byte_ids[b] for b in raw] for raw in (
        b"\xf0\x9f\x99\x82", b"\xe2\x82", b"\xe2A\x82",
        b"\xe0\x80\x80", b"\xed\xa0\x80", b"\xf4\x90\x80\x80",
    )]
    cases += [[item["id"]] for item in metadata["added_tokens"]]
    cases += [rng.choices(byte_ids, k=80) for _ in range(100)]
    cases += [[65535], [64402], [byte_ids[0xe2], 7, byte_ids[0x82], byte_ids[0xac]]]
    cases += [list(range(start, start + 256)) for start in range(0, 65536, 256)]
    for index, ids in enumerate(cases):
        expected = reference.decode(ids, skip_special_tokens=True).encode("utf8")
        actual = call("--decode", *ids)
        if actual.returncode or actual.stdout != expected:
            raise AssertionError(("decode", index, ids, expected, actual.stdout, actual.stderr))
    print(f"{len(cases)} decoding cases match exactly", flush=True)

    for arguments in (("--encode", "a" * 8192), ("--encode", "🙂" * 600),
                      ("--text", "0", "hello"), ("--decode", "-1"),
                      ("--decode", "1098", "bad")):
        actual = call(*arguments)
        if actual.returncode != 1 or actual.stdout:
            raise AssertionError(("invalid", arguments[0], actual))
    malformed = subprocess.run(
        [runtime.encode(), b"_", b"--encode", b"\xf0\x80\x80\x80"],
        capture_output=True, timeout=20,
    )
    if malformed.returncode != 1 or malformed.stdout:
        raise AssertionError("Invalid UTF-8 accepted")
    print("Input bounds and invalid UTF-8 rejected", flush=True)

    with tempfile.TemporaryDirectory(prefix="lfm tokenizer ") as directory:
        executable = Path(directory) / "lfm-next"
        tables = Path(directory) / "tokenizer.bin"
        shutil.copy2(args.runtime, executable)
        command = [str(executable), "_", "--encode", "hello"]
        missing = subprocess.run(command, cwd="/", capture_output=True, timeout=20)
        if missing.returncode != 1 or missing.stdout:
            raise AssertionError("Missing tokenizer tables accepted")
        shutil.copy2(args.runtime.resolve().parent / "tokenizer.bin", tables)
        valid = subprocess.run(command, cwd="/", capture_output=True, timeout=20)
        if valid.returncode or list(map(int, valid.stdout.split())) != reference.encode("hello", add_special_tokens=False).ids:
            raise AssertionError("Tokenizer location depends on working directory")
        with tables.open("r+b") as stream:
            stream.write(b"X")
        corrupt = subprocess.run(command, cwd="/", capture_output=True, timeout=20)
        if corrupt.returncode != 1 or corrupt.stdout:
            raise AssertionError("Corrupt tokenizer tables accepted")
    print("Table location and corruption checks passed", flush=True)

    if args.checkpoint:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer.resolve().parent, local_files_only=True)
        for prompt in ("What is the capital of France?", "Scrivi un saluto in italiano.", "你好！"):
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=True,
                add_generation_prompt=True, return_dict=True,
            )["input_ids"]
            numeric = subprocess.check_output(
                [runtime, str(args.checkpoint), "--generate", "16", *map(str, ids)], timeout=60)
            expected = reference.decode(list(map(int, numeric.split())), skip_special_tokens=True).encode("utf8")
            text = subprocess.check_output(
                [runtime, str(args.checkpoint), "--text", "16", prompt], timeout=60)
            if text != expected:
                raise AssertionError(("text inference", prompt, expected, text))
            print(f"Text inference matches: {prompt!r} -> {text.decode('utf8')!r}", flush=True)


if __name__ == "__main__":
    main()
