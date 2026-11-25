"""Reproduce the two-rank NCCL reference using a CUDA-enabled PyTorch install."""

import json
import os
import statistics
import time
from datetime import timedelta

import torch
import torch.distributed as dist


def benchmark(rank, tokens):
    # Integers in this range and their sums are exactly representable in BF16.
    base = torch.arange(tokens * 5120, device=rank, dtype=torch.int32)
    base.remainder_(21).sub_(10)
    source = base.add(2 * rank).to(torch.bfloat16)
    expected = base.mul_(2).add_(2).to(torch.bfloat16)
    del base

    output = source.clone()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(8):
            output.copy_(source)
            dist.all_reduce(output)
    side.synchronize()
    dist.barrier()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        dist.all_reduce(output)
    samples = []
    for _ in range(7):
        output.copy_(source)
        torch.cuda.synchronize()
        dist.barrier()
        start = time.perf_counter()
        graph.replay()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1e6
        if not torch.equal(output, expected):
            raise RuntimeError(f"Incorrect BF16 output at rank {rank}, shape {tokens}")
        samples.append(elapsed)
    rank_samples = [None, None]
    dist.all_gather_object(rank_samples, samples)
    if rank == 0:
        slower = list(map(max, zip(*rank_samples)))
        print(json.dumps({
            "mode": "torch_nccl",
            "tokens": tokens,
            "bytes": 2 * source.numel(),
            "median_us": statistics.median(slower),
            "samples_us": slower,
        }), flush=True)


def main():
    rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 2:
        raise RuntimeError("Run with exactly two local CUDA ranks.")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=45), device_id=torch.device("cuda", rank))
    try:
        if rank == 0:
            print(json.dumps({"torch": torch.__version__, "nccl": torch.cuda.nccl.version()}), flush=True)
        for tokens in (4096, 8192):
            benchmark(rank, tokens)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
