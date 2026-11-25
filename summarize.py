"""Compare raw pipeline timings with either NCCL reference."""

import argparse
import csv
import json
from collections import defaultdict
from statistics import median


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pipeline_csv", type=argparse.FileType())
    parser.add_argument("nccl_jsonl", type=argparse.FileType())
    args = parser.parse_args()

    measured = defaultdict(list)
    for rows, mode, _, elapsed in csv.reader(args.pipeline_csv):
        if int(mode) == 3:
            measured[int(rows)].append(float(elapsed))

    references = {}
    for line in args.nccl_jsonl:
        if not line.startswith("{"):
            continue
        row = json.loads(line)
        if "samples_us" in row:
            references[row["tokens"]] = median(row["samples_us"])

    if not measured or set(measured) != set(references):
        parser.error("Pipeline and NCCL must have timings for the same payload sizes.")

    print("MiB per GPU | NCCL ms | Pipeline ms | Less time")
    for rows, baseline in sorted(references.items()):
        candidate = median(measured[rows])
        payload_mib = rows * 5120 * 2 / (1 << 20)
        reduction = 100 * (1 - candidate / baseline)
        print(f"{payload_mib:11g} | {baseline/1000:7.3f} | {candidate/1000:11.3f} | {reduction:8.2f}%")


if __name__ == "__main__":
    main()
