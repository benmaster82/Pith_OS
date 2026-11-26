"""Check deterministic row partitioning, wakeups, and independent processes."""
import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CHAT = [1, 6, 6423, 708, 3493, 856, 779, 5706,
        803, 4481, 540, 7, 708, 6, 64015, 708]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("runtime", type=Path)
    args = parser.parse_args()

    def run(tokens, threads):
        env = dict(os.environ)
        env.pop("LFM_THREADS", None)
        if threads is not None:
            env["LFM_THREADS"] = str(threads)
        return subprocess.run(
            [str(args.runtime.resolve()), str(args.checkpoint), *map(str, tokens)],
            env=env, capture_output=True, timeout=120,
        )

    for tokens in ([1], CHAT, CHAT * 4, CHAT * 32):
        reference = run(tokens, 1)
        if reference.returncode or len(reference.stdout) != 65536 * 4:
            raise RuntimeError("Invalid single-thread logits")
        counts = (2, 3, 4, 6, 12, 16, None) if len(tokens) <= 16 else (3, 6, 16)
        for count in counts:
            result = run(tokens, count)
            if result.returncode or result.stdout != reference.stdout:
                raise RuntimeError(f"Logits differ for {len(tokens)} tokens, {count} threads")
        print(f"{len(tokens)} tokens, bit-identical logits for {counts}", flush=True)

    reference = run(CHAT, 1).stdout
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: run(CHAT, 6), range(8)))
    if any(result.returncode or result.stdout != reference for result in results):
        raise RuntimeError("Concurrent process results differ")
    print("Eight overlapping requests passed", flush=True)
    for value in ("", "0", "17", "-1", "2x"):
        result = run([1], value)
        if result.returncode != 1 or result.stdout:
            raise RuntimeError(f"Invalid thread count accepted: {value!r}")
    print("Invalid thread counts rejected", flush=True)


if __name__ == "__main__":
    main()
