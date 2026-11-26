"""Compare the LFM3 runtime tokenizer with the official one, encode and decode."""
import argparse
import os
import random
import subprocess

from tokenizers import Tokenizer

TEXTS = [
    "What is the capital of France?", "Ciao, è già l'ora di pranzo? Sì, perché no!",
    "Ünïcödé façade naïve coöperate — “quotes” ‘single’ … ellipsis", "你好，世界！今天天气很好。",
    "こんにちは、世界。カタカナとひらがな。", "안녕하세요 세계", "Привет, мир! Как дела?", "مرحبا بالعالم",
    "नमस्ते दुनिया", "สวัสดีชาวโลก", "Emoji 🙂🚀👩‍💻🇮🇹 and ZWJ", "def f(x):\n    return x ** 2  # square\n",
    "  leading spaces\tand\ttabs\n\n\ntrailing   ", "\r\nwindows\r\nlines\r\n", "1234567890 3.14159 1e-10 0xFF",
    "I'm you're they've we'd she'll it's DON'T WON'T", "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think></think>",
    "<think>\nreasoning\n</think>\nanswer", "a" * 300, "supercalifragilisticexpialidocious antidisestablishmentarianism",
    "https://example.com/path?q=1&b=2#frag user@example.org", "Mixed: résumé, café, Zürich, São Paulo, Kraków, Øresund",
]


def run(binary, args, env):
    return subprocess.run([binary, "_", *args], capture_output=True, env=env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tokenizer")
    parser.add_argument("binary")
    args = parser.parse_args()
    official = Tokenizer.from_file(args.tokenizer)
    env = dict(os.environ, LFM_CONTEXT="32768")
    random.seed(3)
    texts = TEXTS + ["".join(random.choice(TEXTS) for _ in range(4)) for _ in range(40)]
    failures = 0
    for text in texts:
        expected = official.encode(text, add_special_tokens=False).ids
        result = run(args.binary, ["--encode", text], env)
        got = [int(line) for line in result.stdout.split()]
        if result.returncode or got != expected:
            failures += 1
            print(f"ENCODE MISMATCH {text[:40]!r}: {got[:12]} vs {expected[:12]}")
    print(f"{len(texts) - failures}/{len(texts)} encodings match")
    decode_failures = 0
    for start in range(0, 128000, 2000):
        ids = list(range(start, start + 2000))
        expected = official.decode(ids, skip_special_tokens=True).encode("utf8", "replace")
        result = run(args.binary, ["--decode", *map(str, ids)], env)
        if result.returncode or result.stdout != expected:
            decode_failures += 1
            print(f"DECODE MISMATCH ids {start}..{start + 1999}")
    print(f"{64 - decode_failures}/64 decode ranges of 2000 IDs match")
    return 1 if failures or decode_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
