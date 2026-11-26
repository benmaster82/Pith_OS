"""Reproducible bilingual perplexity diagnostic and native response checks."""
import argparse
import hashlib
import json
import math
import os
import subprocess
import unicodedata
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def normalize_answer(text):
    # Ignore case, whitespace and terminal full stops, not extra explanations.
    return "".join(unicodedata.normalize("NFKC", text).casefold().split()).rstrip(".")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--corpus", type=Path, default=Path("quality_corpus.json"))
    parser.add_argument("--runtime", type=Path, default=Path("build/lfm-next"))
    parser.add_argument("--logits", type=Path, default=Path("build/lfm-logits"))
    parser.add_argument("--output", type=Path, default=Path("build/quality-report.json"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    corpus_bytes = args.corpus.read_bytes()
    corpus = json.loads(corpus_bytes)
    model = AutoModelForCausalLM.from_pretrained(args.source.resolve().parent,
        dtype=torch.float32, attn_implementation="eager", local_files_only=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.source.resolve().parent, local_files_only=True)
    documents = [(doc, [1] + tokenizer.encode(doc["text"], add_special_tokens=False))
                 for doc in corpus["documents"]]
    report = dict(description=corpus["description"], corpus_sha256=hashlib.sha256(corpus_bytes).hexdigest(),
                  method="Teacher-forced FP32 Transformers with official BF16 weights; independent BOS per document; no truncation, padding or EOS; native assembly for response tasks",
                  format="BF16")
    rows = []
    checked_languages = set()
    for doc, ids in documents:
        if len(ids) > 512:
            raise ValueError(f"Document too long for the tested context: {doc['id']}")
        with torch.inference_mode():
            logits = model(torch.tensor([ids[:-1]]), use_cache=False).logits[0]
            targets = torch.tensor(ids[1:])
            nll = torch.nn.functional.cross_entropy(logits, targets, reduction="sum").item()
        native_error = None
        if doc["language"] not in checked_languages:
            output = subprocess.check_output([str(args.logits.resolve()), str(args.source), *map(str, ids[:-1])],
                env=dict(os.environ, LFM_THREADS="6"), timeout=120)
            native = torch.frombuffer(bytearray(output), dtype=torch.float32)
            torch.testing.assert_close(native, logits[-1], rtol=1e-4, atol=1e-3)
            native_error = float((native - logits[-1]).abs().max())
            checked_languages.add(doc["language"])
        rows.append(dict(id=doc["id"], language=doc["language"], tokens=len(targets),
                         nll_sum=nll, native_last_logits_max_error=native_error))
    summary = {}
    for language in ("it", "en", "all"):
        selected = [row for row in rows if language == "all" or row["language"] == language]
        count = sum(row["tokens"] for row in selected)
        nll = sum(row["nll_sum"] for row in selected) / count
        summary[language] = dict(tokens=count, nll=nll, perplexity=math.exp(nll))
    tasks = []
    for task in corpus["tasks"]:
        answer = subprocess.check_output([str(args.runtime.resolve()), str(args.source),
            "--text", "32", task["prompt"]], env=dict(os.environ, LFM_THREADS="6"),
            timeout=120).decode("utf8")
        tasks.append(dict(id=task["id"], language=task["language"], expected=task["expected"], answer=answer,
                          match=normalize_answer(answer) == normalize_answer(task["expected"])))
    report.update(summary=summary, documents=rows, tasks=tasks,
                  response_matches=sum(task["match"] for task in tasks))
    print("BF16", json.dumps(summary), f"responses {sum(task['match'] for task in tasks)}/{len(tasks)}", flush=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
