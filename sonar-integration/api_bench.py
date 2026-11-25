"""Measure Sonar time to first token with distinct, uncached prompt IDs."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import urllib.request

URL = "http://127.0.0.1:2242/v1/completions"
MODEL = "qwen3.8-27b"


def sample(size, repeat, namespace, count):
    rng = random.Random(4190 + size + 101 * repeat)
    data = {
        "model": MODEL, "prompt": [rng.randrange(1000, 30000) for _ in range(size)],
        "max_tokens": count, "temperature": 0, "seed": 3090, "ignore_eos": True,
        "cache_salt": f"medium-collectives-{namespace}-{size}-{repeat}",
        "stream": True, "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(URL, data=json.dumps(data).encode(),
                                 headers={"Content-Type": "application/json"})
    begin = time.perf_counter()
    first = last = usage = None
    text = []
    with urllib.request.urlopen(req, timeout=120) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                break
            event = json.loads(payload)
            if event.get("error"):
                raise RuntimeError(event["error"])
            fragments = [choice["text"] for choice in event.get("choices", []) if choice.get("text")]
            if fragments:
                last = time.perf_counter()
                if first is None:
                    first = last
                text.extend(fragments)
            if event.get("usage"):
                usage = event["usage"]
    if first is None or usage is None or (usage["prompt_tokens"], usage["completion_tokens"]) != (size, count):
        raise RuntimeError(f"Unexpected response shape for {size}: {usage}")
    return {"tokens": size, "repeat": repeat, "ttft_s": first - begin,
            "elapsed_s": last - begin, "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "text_sha256": hashlib.sha256("".join(text).encode()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label")
    parser.add_argument("--sizes", type=int, nargs="+", default=[512, 1024, 1536, 2048, 3072, 5000])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()
    if args.rounds <= 0:
        parser.error("At least one measured round is required.")
    output = Path(f"api-{args.label}.jsonl")
    with output.open("x") as log:
        for index in range(args.rounds + 1):
            repeat = args.offset + index
            sizes = list(args.sizes)
            random.Random(3090 + repeat).shuffle(sizes)
            for size in sizes:
                row = sample(size, repeat, args.label, 24)
                row.update(label=args.label, warmup=index == 0)
                line = json.dumps(row)
                print(line, file=log, flush=True)
                print(line, flush=True)


if __name__ == "__main__":
    main()
