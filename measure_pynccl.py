"""Measure two-rank Sonar PyNCCL BF16 all-reduce outside the serving process."""

import argparse
from datetime import timedelta
import json
import os
import random
import statistics
import time

import torch
import torch.distributed as dist

from sonar.distributed.device_communicators.pynccl import PyNcclCommunicator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="default")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--prefill", action="store_true", help="Time one graph launch at 40 and 80 MiB per rank.")
    args = parser.parse_args()

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", timeout=timedelta(seconds=30))
    comm = PyNcclCommunicator(dist.group.WORLD, rank)
    if not comm.available:
        raise RuntimeError("Sonar PyNCCL is unavailable")

    shapes = [1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 128, 512, 1024]
    if args.prefill:
        shapes = [4096, 8192]
    rng = random.Random(3090)
    rng.shuffle(shapes)
    for tokens in shapes:
        count = tokens * 5120
        source = (torch.arange(count, device="cuda", dtype=torch.int32) % 21 - 10
                  + 2 * rank).to(torch.bfloat16)
        expected = (torch.arange(count, device="cuda", dtype=torch.int32) % 21 * 2
                    - 18).to(torch.bfloat16)
        result = torch.empty_like(source)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        batch = min(128, max(8, 2**23 // (count * 2)))
        if args.prefill:
            batch = 1
        with torch.cuda.stream(stream):
            for _ in range(8):
                comm.all_reduce(source, result)
        stream.synchronize()
        if not torch.equal(expected, result):
            raise RuntimeError(f"Rank {rank} incorrect warmup at {tokens} tokens")
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(batch):
                comm.all_reduce(source, result)
        graph.replay()
        torch.cuda.synchronize()
        if not torch.equal(expected, result):
            raise RuntimeError(f"Rank {rank} incorrect graph output at {tokens} tokens")

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(args.rounds):
            dist.barrier()
            if args.prefill:
                began = time.perf_counter()
                graph.replay()
                torch.cuda.synchronize()
                microseconds = (time.perf_counter() - began) * 1e6
            else:
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                microseconds = 1_000 * start.elapsed_time(end) / batch
            elapsed = torch.tensor([microseconds], dtype=torch.float64)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            times.append(elapsed.item())
        if rank == 0:
            print(json.dumps({"mode": args.mode, "tokens": tokens, "bytes": count * 2,
                              "median_us": statistics.median(times),
                              "samples_us": times}), flush=True)
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
