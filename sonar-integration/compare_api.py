"""Compare identical requests and check whether the measured gain justifies deployment."""

import argparse
import json
from pathlib import Path
from statistics import median


def load(label):
    rows = [json.loads(line) for line in Path(f"api-{label}.jsonl").read_text().splitlines()]
    result = {(row["tokens"], row["repeat"]): row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError(f"Duplicate cases in {label}.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate")
    parser.add_argument("--qualify", action="store_true")
    args = parser.parse_args()
    baseline = load("incumbent")
    candidate = load(args.candidate)
    assert set(candidate) <= set(baseline)
    if args.qualify:
        assert set(candidate) == set(baseline)
    fields = ("text_sha256", "prompt_tokens", "completion_tokens")
    mismatches = [key for key, row in candidate.items()
                  if any(row[field] != baseline[key][field] for field in fields)]
    assert not mismatches, mismatches
    result = {}
    for tokens in sorted({key[0] for key in candidate}):
        keys = [key for key, row in candidate.items() if key[0] == tokens and not row["warmup"]]
        old = median(baseline[key]["ttft_s"] for key in keys)
        new = median(candidate[key]["ttft_s"] for key in keys)
        result[tokens] = {"incumbent_s": old, "candidate_s": new,
                          "less_ttft_percent": 100 * (1 - new / old)}
    print(json.dumps({"responses_matched": len(candidate), "ttft": result}, indent=2), flush=True)
    if args.qualify:
        controls = [result[t]["less_ttft_percent"] for t in (256, 5000, 9000)]
        medium = [result[t]["less_ttft_percent"] for t in (512, 1024, 2048, 3072)]
        assert min(controls) >= -5, f"Unexpected control regression: {controls}"
        assert min(medium) >= -5 and sum(gain >= 3 for gain in medium) >= 2, medium


if __name__ == "__main__":
    main()
