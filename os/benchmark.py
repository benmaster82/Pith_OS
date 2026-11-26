"""Measure guest greedy decode scaling with identical BF16 weights and tokens."""
import argparse
import json
import os
from pathlib import Path
import selectors
import statistics
import subprocess
import time

from run import command

PROMPT = "Count from 1 to 30, putting each number on its own line."


def measure(bundle, threads):
    argv = command("build/lfm-os.elf", bundle,
                   f"mode=ids tokens=64 context=512 threads={threads} prompt={PROMPT}", cpus=6)
    start = time.perf_counter()
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    arrivals, tokens, pending = [], [], b""
    begin = None
    active = False
    cores = None
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while True:
                remaining = 120 - (time.perf_counter() - start)
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError("Guest inference exceeded 120 seconds")
                chunk = os.read(process.stdout.fileno(), 65536)
                if not chunk:
                    break
                now = time.perf_counter()
                pending += chunk
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    if line.startswith(b"Active CPU cores: "):
                        cores = int(line.split(b": ")[1])
                    elif line == b"<lfm-output>":
                        begin, active = now, True
                    elif line == b"</lfm-output>":
                        active = False
                    elif active and line:
                        tokens.append(int(line))
                        arrivals.append(now)
        assert process.wait(timeout=10) == 33, process.stderr.read()
        assert len(tokens) == 64 and cores == threads and begin is not None, (cores, tokens)
        return tokens, {
            "startup_seconds": begin - start,
            "prefill_first_token_seconds": arrivals[0] - begin,
            "decode_tokens_per_second": 63 / (arrivals[-1] - arrivals[0]),
            "total_seconds": time.perf_counter() - start,
        }
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        process.stdout.close()
        process.stderr.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 6])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("build/os-benchmark.json"))
    args = parser.parse_args()
    if args.repeats < 1 or any(not 1 <= n <= 6 for n in args.threads):
        parser.error("Positive repeats and worker counts 1..6 are required for this six-vCPU benchmark")
    expected = None
    samples = {str(n): [] for n in args.threads}
    for n in args.threads:
        tokens, _ = measure(args.bundle, n)
        assert expected is None or tokens == expected
        expected = tokens
    for iteration in range(args.repeats):
        for n in args.threads if iteration % 2 == 0 else args.threads[::-1]:
            tokens, metrics = measure(args.bundle, n)
            assert tokens == expected
            samples[str(n)].append(metrics)
            print(f"{n} cores, run {iteration+1}: {metrics}", flush=True)
    medians = {n: {key: statistics.median(row[key] for row in rows) for key in rows[0]}
               for n, rows in samples.items()}
    report = {"prompt": PROMPT, "generated_ids": expected, "vcpus": 6, "memory": "2G",
              "accel": "kvm", "samples": samples, "medians": medians}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(medians, indent=2), flush=True)


if __name__ == "__main__":
    main()
