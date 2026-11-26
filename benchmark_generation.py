"""Measure streaming time to first token and cached greedy decode throughput."""
import argparse
import json
import os
import selectors
import statistics
import subprocess
import time
from pathlib import Path

# Official chat template for "Count from 1 to 30, putting each number on its own line."
PROMPT = [1, 6, 6423, 708, 9203, 988, 730, 526, 811, 730, 1664, 521,
          13935, 1895, 2080, 884, 1352, 2063, 2717, 523, 7, 708, 6, 64015, 708]


def measure(runtime, checkpoint, count, threads):
    start = time.perf_counter()
    deadline = start + 120
    process = subprocess.Popen(
        [str(runtime.resolve()), str(checkpoint), "--generate", str(count), *map(str, PROMPT)],
        env=dict(os.environ, LFM_THREADS=str(threads)),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
    )
    tokens, arrivals = [], []
    pending = b""
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while True:
                remaining = deadline - time.perf_counter()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError("Generation exceeded 120 seconds")
                chunk = os.read(process.stdout.fileno(), 4096)
                if not chunk:
                    break
                now = time.perf_counter()
                pending += chunk
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    tokens.append(int(line))
                    arrivals.append(now)
        returncode = process.wait(timeout=10)
        if returncode or pending or len(tokens) < 2:
            raise RuntimeError(f"Invalid generation output: {process.stderr.read()!r}")
        duration = arrivals[-1] - arrivals[0]
        if duration <= 0:
            raise RuntimeError("Insufficient timer resolution for decode")
        return tokens, {
            "first_token_seconds": arrivals[0] - start,
            "decode_tokens_per_second": (len(tokens) - 1) / duration,
            "total_seconds": time.perf_counter() - start,
            "generated_tokens": len(tokens),
        }
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()
        process.stderr.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("runtime", type=Path)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 6])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 2 <= args.tokens <= 512 - len(PROMPT) + 1:
        parser.error("--tokens must be between 2 and 488")
    if args.repeats < 1 or any(not 1 <= count <= 16 for count in args.threads):
        parser.error("Positive repeats and thread counts between 1 and 16 are required")
    expected = None
    samples = {str(count): [] for count in args.threads}
    for count in args.threads:
        tokens, _ = measure(args.runtime, args.checkpoint, args.tokens, count)
        if expected is not None and tokens != expected:
            raise RuntimeError("Warmup tokens differ")
        expected = tokens
    for iteration in range(args.repeats):
        order = args.threads if iteration % 2 == 0 else list(reversed(args.threads))
        for count in order:
            tokens, metrics = measure(args.runtime, args.checkpoint, args.tokens, count)
            if tokens != expected:
                raise RuntimeError("Generated sequence changed")
            samples[str(count)].append(metrics)
            print(f"{count} threads, run {iteration + 1}: {metrics}", flush=True)
    medians = {
        count: {metric: statistics.median(row[metric] for row in rows)
                for metric in rows[0]}
        for count, rows in samples.items()
    }
    report = {"prompt_ids": PROMPT, "max_new_tokens": args.tokens,
              "samples": samples, "medians": medians}
    print(json.dumps(medians, indent=2), flush=True)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
