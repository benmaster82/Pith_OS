"""Compare the Pithos CORE guest (LFM3) with the Linux runtime; requires QEMU and KVM."""
import argparse
import hashlib
import os
from pathlib import Path
import selectors
import struct
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "os"))
from run import command  # noqa: E402

BEGIN = b"<lfm-output>\n"
END = b"\n</lfm-output>\n"
FRANCE = "What is the capital of France?"
KERNEL = "build/lfm3/lfm3-os.elf"
LOGITS_KERNEL = "build/lfm3/lfm3-os-logits.elf"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    env = dict(os.environ, LFM_THREADS="1", LFM_CONTEXT="2048")

    with args.bundle.open("rb") as stream:
        magic, model_offset, model_size, tok_offset, tok_size, total = struct.unpack("<8sQQQQQ16x", stream.read(64))
        assert magic == b"LFMBT001" and total == args.bundle.stat().st_size
        for offset, size, source in ((model_offset, model_size, args.checkpoint),
                                     (tok_offset, tok_size, Path("build/lfm3/tokenizer.bin"))):
            assert size == source.stat().st_size
            stream.seek(offset)
            digest, remaining = hashlib.sha256(), size
            while remaining:
                block = stream.read(min(1 << 20, remaining))
                assert block
                digest.update(block)
                remaining -= len(block)
            with source.open("rb") as original:
                assert digest.digest() == hashlib.file_digest(original, "sha256").digest()
    print("Bundle preserves the GGUF and tokenizer bytes (SHA-256 comparison)", flush=True)

    def linux(arguments, executable="lfm3-next", data=None):
        result = subprocess.run([f"build/lfm3/{executable}", str(args.checkpoint), *map(str, arguments)],
                                input=data, capture_output=True, env=env, timeout=300)
        assert result.returncode == 0, result.stderr
        return result.stdout

    def ids(prompt):
        template = ("<|startoftext|><|im_start|>user\n" + prompt
                    + "<|im_end|>\n<|im_start|>assistant\n<think></think>")
        return list(map(int, linux(["--encode", template]).split()))

    def boot(options, kernel=KERNEL, bundle=None, memory="2G", cpu=None, cpus=1, status=33):
        argv = command(kernel, bundle or args.bundle, options, memory=memory, cpus=cpus)
        if cpu:
            argv[argv.index("-cpu") + 1] = cpu
        if bundle is False:
            index = argv.index("-initrd")
            del argv[index:index + 2]
        result = subprocess.run(argv, capture_output=True, timeout=300)
        assert result.returncode == status, (options, result.returncode, result.stdout[-1000:], result.stderr)
        return result.stdout

    def payload(output):
        assert BEGIN in output and output.endswith(END), output[-1000:]
        return output.split(BEGIN, 1)[1][:-len(END)]

    cases = [(FRANCE, 16), (FRANCE, 3), ("Qual è la capitale dell'Italia?", 16),
             ("中国的首都是哪里？", 16), ("Write a detailed explanation of how a computer works.", 64)]
    for prompt, count in cases:
        actual = payload(boot(f"mode=ids tokens={count} threads=1 prompt={prompt}"))
        assert actual == linux(["--generate", count, *ids(prompt)]), (prompt, actual)
    print("Five greedy sequences match Linux: EOS, truncation, Italian, Chinese and 64-token limit", flush=True)

    prompt, count = cases[-1]
    expected = linux(["--generate", count, *ids(prompt)])
    for cores in (2, 3, 4, 6):
        output = boot(f"mode=ids tokens={count} threads={cores} prompt={prompt}", cpus=cores)
        assert f"Active CPU cores: {cores}".encode() in output and payload(output) == expected, cores
    print("The 64-token sequence matches Linux on 2, 3, 4 and 6 active cores", flush=True)

    for prompt in (FRANCE, "Scrivi un saluto in italiano.", "中国的首都是哪里？"):
        actual = payload(boot(f"mode=text tokens=16 threads=1 prompt={prompt}"))
        assert actual == linux(["--text", 16, prompt]), (prompt, actual)
    print("UTF-8 text output matches Linux in English, Italian and Chinese", flush=True)

    for prompt in (FRANCE, "中国的首都是哪里？"):
        expected = linux(ids(prompt), "lfm3-logits")
        for cores in (1, 4):
            actual = payload(boot(f"mode=ids tokens=1 threads={cores} prompt={prompt}", kernel=LOGITS_KERNEL,
                                  cpus=cores))
            assert len(actual) == 128000 * 4 and actual == expected, (prompt, cores, len(actual))
    print("All 128,000 FP32 logits are bitwise identical to Linux for two prompts on 1 and 4 cores", flush=True)

    for cpu in ("host,-avx2", "host,-f16c", "host,-avx,-avx2,-fma,-xsave"):
        output = boot(f"mode=ids tokens=1 prompt={FRANCE}", cpu=cpu, status=35)
        assert b"LFM3 needs a CPU with AVX2, FMA and F16C" in output and BEGIN not in output, (cpu, output[-500:])
    print("CPUs without AVX2, F16C or XSAVE stop with a clear message", flush=True)

    def chat(lines, tokens=16, context=2048):
        process = subprocess.Popen(command(KERNEL, args.bundle, f"mode=chat tokens={tokens} context={context}"),
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        pending = bytearray()

        def until_prompt():
            deadline = time.monotonic() + 300
            while b"user> " not in pending:
                remaining = deadline - time.monotonic()
                assert remaining > 0 and selector.select(max(0, remaining)), bytes(pending[-1000:])
                chunk = os.read(process.stdout.fileno(), 65536)
                assert chunk, (process.poll(), bytes(pending[-1000:]))
                pending.extend(chunk)
            end = pending.index(b"user> ") + len(b"user> ")
            output = bytes(pending[:end])
            del pending[:end]
            return output

        try:
            until_prompt()
            responses = []
            for line in lines:
                process.stdin.write(line)
                process.stdin.flush()
                responses.append(until_prompt())
            process.stdin.write(b"\x04")
            process.stdin.flush()
            assert process.wait(timeout=10) == 33
            return responses
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            selector.close()
            for stream in (process.stdin, process.stdout, process.stderr):
                stream.close()

    lines = ["What is the capital of Francé\x7fe?\r\n".encode(), b"And of Italy?\n", b"\xff\n",
             b"/unknown\n", b"/clear\n", FRANCE.encode() + b"\n"]
    results = chat(lines)
    # COM1 carries both streams; Linux prints the limit notice on stderr.
    notice = b"Response reached the token limit. The assistant turn was closed.\n"
    actual = b"".join(part.split(b"assistant> ", 1)[1][:-len(b"user> ")].replace(notice, b"")
                      for part in results if b"assistant> " in part)
    assert all(notice in part for part in results if b"assistant> " in part)
    expected = linux(["--chat", 16], data=(FRANCE + "\nAnd of Italy?\n/clear\n" + FRANCE + "\n/quit\n").encode())
    assert actual == expected, (actual, expected)
    assert b"Input rejected" in results[2] and b"Unknown command" in results[3]
    assert b"Conversation cleared" in results[4] and b"\x08 \x08" in results[0]
    print("Serial chat retains history, handles CRLF/backspace, rejects invalid UTF-8, resets and exits", flush=True)

    # One turn fills the cache exactly: prompt, one answer token, EOS and newline.
    capacity = len(ids(FRANCE)) + 3
    lines = [FRANCE.encode() + b"\n", FRANCE.encode() + b"\n", b"/clear\n", FRANCE.encode() + b"\n"]
    results = chat(lines, tokens=1, context=capacity)
    assert b"assistant> " in results[0] and b"Context full" in results[1]
    assert results[0] == results[3] and b"token limit" in results[3]
    print(f"Exact cache capacity of {capacity} positions rejects another turn and recovers after /clear", flush=True)

    for options in ("context=0", "tokens=0", "mode=wrong", "unknown=1"):
        assert b"FATAL:" in boot(options, status=35)
    too_long = boot("mode=ids tokens=1 context=1", status=35)
    assert BEGIN not in too_long and b"Invalid UTF-8 or prompt exceeds byte/context limits" in too_long, too_long
    assert b"FATAL:" in boot("mode=ids tokens=1", bundle=False, status=35)
    large_cache = boot(f"mode=ids tokens=1 context=32768 prompt={FRANCE}", memory="3G")
    assert payload(large_cache) == linux(["--generate", 1, *ids(FRANCE)])
    assert b"Failed to allocate the KV cache" in boot("mode=ids tokens=1 context=32768", memory="2G", status=35)
    with tempfile.TemporaryDirectory(prefix="lfm3-boot-test-") as directory:
        bad_bundle = Path(directory) / "bad.bundle"
        with bad_bundle.open("wb") as stream:
            stream.write(b"INVALID!")
            stream.truncate(args.bundle.stat().st_size)
        assert b"FATAL:" in boot("mode=ids tokens=1", bundle=bad_bundle, status=35)
    print("Invalid options, missing/corrupt module and insufficient cache memory fail cleanly", flush=True)


if __name__ == "__main__":
    main()
