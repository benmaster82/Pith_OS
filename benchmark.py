"""Measure end-to-end next-token latency with warm filesystem cache."""
import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, nargs="+", default=[1])
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if any(count < 1 or count > 16 for count in args.threads):
        parser.error("--threads must be between 1 and 16")
    prompts = {
        "one_token": [1],
        "chat_16_tokens": [1, 6, 6423, 708, 3493, 856, 779, 5706,
                           803, 4481, 540, 7, 708, 6, 64015, 708],
    }
    report = {}
    for label, tokens in prompts.items():
        configurations = {"baseline": (args.baseline, 1)}
        configurations.update({f"candidate_{count}T": (args.candidate, count)
                               for count in args.threads})
        commands = {}
        environments = {}
        for name, (path, count) in configurations.items():
            commands[name] = [str(path.resolve()), str(args.checkpoint), *map(str, tokens)]
            environments[name] = dict(os.environ, LFM_THREADS=str(count))
        expected = subprocess.check_output(commands["baseline"], env=environments["baseline"], timeout=120)
        for name in commands:
            if subprocess.check_output(commands[name], env=environments[name], timeout=120) != expected:
                raise RuntimeError(f"Warmup prediction changed in {label}, {name}")
        samples = {name: [] for name in commands}
        # Alternate order to reduce temperature and scheduling bias.
        for iteration in range(args.repeats):
            order = list(commands)
            if iteration % 2:
                order.reverse()
            for name in order:
                start = time.perf_counter()
                actual = subprocess.check_output(commands[name], env=environments[name], timeout=120)
                samples[name].append(time.perf_counter() - start)
                if actual != expected:
                    raise RuntimeError(f"Prediction changed in {label}, {name}")
        medians = {name: statistics.median(values) for name, values in samples.items()}
        report[label] = {
            "samples_seconds": samples, "median_seconds": medians,
            "speedup": {name: medians["baseline"] / value
                        for name, value in medians.items() if name != "baseline"},
        }
        print(label, json.dumps(report[label]), flush=True)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
