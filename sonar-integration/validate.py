import json
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist
from sonar.distributed.device_communicators.pynccl import PyNcclCommunicator
import adapter

rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
dist.init_process_group('gloo', timeout=timedelta(seconds=120))
comm = PyNcclCommunicator(dist.group.WORLD, rank)
assert comm._host_staged_handle
checks = 0


def compare(actual, expected):
    global checks
    torch.cuda.synchronize()
    if not torch.equal(actual.view(torch.int16), expected.view(torch.int16)):
        raise RuntimeError('Bitwise mismatch at rank ' + str(rank))
    checks += 1


sizes = [5 * 2**20 - 2, 5 * 2**20, 5 * 2**20 + 514, 10 * 2**20, 20 * 2**20,
         30 * 2**20, 40 * 2**20, 80 * 2**20, 5120 * 2, 80 * 2**20 + 514]
for size in sizes:
    for iteration in range(4):
        torch.manual_seed(3090 + rank + 100 * iteration)
        tensor = torch.randn(size // 2, device=rank, dtype=torch.bfloat16)
        reference = adapter._original_reduce(comm, tensor)
        if rank == 1 and iteration % 2:
            time.sleep(0.03)
        compare(comm.all_reduce(tensor), reference)
        compare(comm.all_reduce(tensor, tensor), reference)

# Multiple differently sized collectives in the same captured graph, including
# partial chunks. Repeated inputs change after capture, and one rank is delayed.
inputs = [torch.empty(size // 2, device=rank, dtype=torch.bfloat16) for size in [5 * 2**20 + 514, 20 * 2**20, 80 * 2**20]]
for tensor in inputs:
    tensor.fill_(rank + 1)
side = torch.cuda.Stream()
side.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(side):
    for tensor in inputs:
        comm.all_reduce(tensor)
side.synchronize()
dist.barrier()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, stream=side):
    outputs = [comm.all_reduce(tensor) for tensor in inputs]
for iteration in range(24):
    for tensor in inputs:
        tensor.fill_(iteration + rank)
    references = [adapter._original_reduce(comm, tensor) for tensor in inputs]
    torch.cuda.synchronize()
    dist.barrier()
    if rank == iteration % 2:
        time.sleep(0.02)
    graph.replay()
    for output, reference in zip(outputs, references):
        compare(output, reference)

eligible_sizes = {n for n in sizes if 5 * 2**20 <= n <= 80 * 2**20}
assert comm._host_staged_observed_sizes == eligible_sizes

small = torch.full((32,), rank + 1, device=rank, dtype=torch.float32)
assert torch.all(comm.all_reduce(small, op=dist.ReduceOp.MAX) == 2)
comm.disabled = True
assert comm.all_reduce(small) is None
comm.disabled = False
print(json.dumps({'rank': rank, 'bitwise_checks': checks, 'graphs_replayed': 24, 'status': 'PASS'}), flush=True)
dist.barrier()
dist.destroy_process_group()
