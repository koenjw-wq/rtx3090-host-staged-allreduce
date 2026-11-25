# Host-staged BF16 all-reduce on two RTX 3090s

**Project Prometheus — AI-directed GPU performance research**

Overlapping CUDA copies through pinned system memory reduced the time for large, two-GPU BF16 all-reduces by **23–26% versus default NCCL 2.30.7** in standalone benchmarks across two different hardware topologies.

The original experiment used two RTX 3090s on the same NUMA node with PCIe Gen3 x8 connections. A subsequent replication on an HPE ProLiant DL360 Gen10 used two RTX 3090s on separate physical CPU sockets with PCIe Gen3 x16 connections.

The performance improvement was reproduced on both systems, with all reported output correctness checks passing.

**No native CUDA peer-to-peer access is required.**

## Sonar integration: measured Qwen TTFT gains — 25 November 2026

The current two-worker adapter in Sonar 0.26.0 selects eligible contiguous BF16 SUM collectives of **5–80 MiB** on the tested dual RTX 3090 configuration. Other operations, including small decode reductions, retain NCCL.

### Medium-prompt update

The table compares the new 5 MiB cutoff with the previous 40 MiB cutoff after a full Sonar service restart on each server.

| GPU topology | Prompt tokens | Previous TTFT | New TTFT | Less time |
| --- | ---: | ---: | ---: | ---: |
| Same NUMA node | 2,048 | 1.208 s | 1.133 s | +6.17% |
| Same NUMA node | 3,072 | 1.795 s | 1.639 s | +8.70% |
| Cross-socket | 2,048 | 1.014 s | 0.961 s | +5.18% |
| Cross-socket | 3,072 | 1.482 s | 1.390 s | +6.25% |

All 54 paired API responses per server matched by text digest and token counts. Both GPU workers exercised the expanded range in live inference, and four concurrent requests passed on each server. [Medium-prompt methodology and raw evidence](results/sonar-medium-reductions/REPORT.md)

### Original 40 MiB cutoff results

The following historical table uses the original NCCL baseline and the first 40–80 MiB adapter. The two tables use different baselines and prompt sets.

| System | Prompt tokens | Baseline median TTFT | Adapter after restart | Less elapsed time |
| --- | ---: | ---: | ---: | ---: |
| Dual socket, PCIe Gen3 x16 | 5,000 | 2.465 s | 2.262 s | 8.24% |
| Dual socket, PCIe Gen3 x16 | 9,000 | 4.323 s | 4.055 s | 6.22% |
| Single socket, PCIe Gen3 x8 | 5,000 | 2.952 s | 2.661 s | 9.9% |
| Single socket, PCIe Gen3 x8 | 9,000 | 5.264 s | 4.921 s | 6.5% |

The dual-socket integration handover includes comparison data and reports 120 bitwise checks per rank against NCCL. The integrated two-process collective benchmark measured approximately 17–18% less time than earlier NCCL references on that host. The initial API comparison measured 8.22% and 6.55% lower TTFT at 5,000 and 9,000 tokens respectively.

On both systems, all eight benchmark completion texts reportedly matched baseline in both runs, four simultaneous requests passed, and both workers loaded and exercised the adapter after service restart. The single-socket results come from the subsequent deployment report; raw timing files for that machine are not yet included in the supplied handover.

The adapter is persistent across Sonar service starts. On the single-socket system Sonar itself remains manually started to preserve ComfyUI's existing boot behavior. On the dual-socket system service boot enablement was verified; an actual hardware reboot was not qualified.

These are controlled TTFT results, not measured sustained decode throughput gains. Long-context, arbitrary concurrent collective streams and other engine versions require further qualification.

[Sonar integration results and limitations](results/sonar-integration/REPORT.md)

**Reproduce the integration:** [Sonar build, validation, installation and rollback guide](sonar-integration/README.md). The two-worker CUDA backend, current Python adapter, startup hook, validation and API benchmark scripts are published with source checksums. Build the native library locally.

## Headline results

| Hardware topology | Payload per GPU | Default NCCL | Host pipeline | Less elapsed time |
| --- | ---: | ---: | ---: | ---: |
| Single socket, PCIe Gen3 x8 | 40 MiB | 10.754 ms | 8.079 ms | 24.87% |
| Single socket, PCIe Gen3 x8 | 80 MiB | 21.654 ms | 15.974 ms | 26.23% |
| Dual socket, PCIe Gen3 x16 | 40 MiB | 7.807 ms | 5.965 ms | 23.60% |
| Dual socket, PCIe Gen3 x16 | 80 MiB | 15.505 ms | 11.778 ms | 24.04% |

These are the repeat-run medians. Each median uses seven samples.

The original single-socket initial run measured 10.704 → 8.218 ms and 21.422 → 16.181 ms, respectively.

The dual-socket initial run measured 7.818 → 5.968 ms and 15.508 → 11.773 ms, respectively.

Every tested prototype output matched its BF16 reference bit for bit on both GPUs.

## Key findings

- 23–26% less collective execution time compared with the installed NCCL path.
- Repeatable improvements on two separate physical machines.
- Successful validation across single-socket and dual-socket CPU topologies.
- Successful validation with NVIDIA drivers 595.99.02 and 615.71.09.
- All reported prototype output correctness checks passed.
- Performance gains persisted across four requested NUMA placement policies.
- No native CUDA P2P, NVLink, or additional hardware required for this algorithm.
- No driver, firmware, or BIOS modifications are required by the communication algorithm itself.

These findings describe standalone collective workloads. Separate controlled Qwen TTFT improvements are documented in the Sonar integration section above.

## Hardware and software configurations

| Specification | Original experimental server | Dual-socket replication server |
| --- | --- | --- |
| GPUs | 2× RTX 3090 24 GB | 2× RTX 3090 24 GB |
| CPU topology | Single socket | Dual socket |
| GPU NUMA placement | Same NUMA node | Separate NUMA nodes |
| NVIDIA topology | PHB | SYS |
| PCIe connectivity | Gen3 x8 per GPU | Gen3 x16 per GPU |
| NVIDIA driver | 595.99.02 | 615.71.09 open driver |
| GPU power limits | 300 W | 420 W |
| NCCL | 2.30.7 | 2.30.7 |
| Native CUDA P2P | Unavailable | Unavailable |

The second server is an HPE ProLiant DL360 Gen10 equipped with:

- Two Intel Xeon Silver 4110 processors.
- 32 logical CPUs.
- Two NUMA nodes.
- Approximately 192 GiB of system memory.
- Linux kernel 7.0.0-42-generic.
- NVIDIA open driver and GSP firmware 615.71.09.
- Sonar 0.26.0.
- PyTorch 2.14.0+cu130.
- NCCL 2.30.7.
- CUDA compiler 13.0.88.
- Two RTX 3090 GPUs, one attached to each CPU socket.
- 256 MiB BAR1 apertures on both GPUs.
- Existing BF16 performance unlock active during testing.

CUDA reported direct peer access unavailable in both directions.

NCCL transport logs identified SHM/direct communication in both directions, meaning that the installed NCCL path used shared host memory.

Absolute timing comparisons between the two machines include differences in CPU architecture, PCIe topology, driver versions, power limits and other platform characteristics. The measurements do not isolate which individual platform difference accounts for the lower absolute timings on the dual-socket server.

## Cross-socket replication — 25 November 2026

The original published collective improvement was reproduced on a second physical server with a substantially different PCIe and NUMA topology.

The replication used the original source pinned to commit:

`a966f728545160edc65fab918bdb99c2aca64382`

The source working tree was clean. The original CUDA algorithm and measurement harnesses were executed directly.

[Original published revision](https://github.com/koenjw-wq/rtx3090-host-staged-allreduce/tree/a966f728545160edc65fab918bdb99c2aca64382)

### Default and NUMA-policy repeat results

Every value is the median of seven timed samples.

The candidate uses upstream mode 3, with 1 MiB chunks and one collective per CUDA graph replay.

Each repeat pipeline result is paired with a fresh PyNCCL baseline.

| Requested process placement | MiB per GPU | NCCL ms | Pipeline ms | Less elapsed time |
| --- | ---: | ---: | ---: | ---: |
| Default | 40 | 7.807 | 5.965 | 23.60% |
| Default | 80 | 15.505 | 11.778 | 24.04% |
| Node 0 | 40 | 7.807 | 5.934 | 23.98% |
| Node 0 | 80 | 15.550 | 11.719 | 24.64% |
| Node 1 | 40 | 7.824 | 5.958 | 23.86% |
| Node 1 | 80 | 15.548 | 11.778 | 24.25% |
| Interleave | 40 | 7.845 | 5.915 | 24.61% |
| Interleave | 80 | 15.446 | 11.648 | 24.59% |

Across both repetitions and all four requested placement policies, the pipeline reduced all-reduce execution time by **23.60–25.04%**.

### All matched replication runs

Run A uses the original `--repeat` option with seed 6290 and is paired with NCCL-before.

Run B uses seed 3090 and is paired with NCCL-after.

| Placement | Run | MiB per GPU | NCCL ms | Pipeline ms | Less elapsed time |
| --- | --- | ---: | ---: | ---: | ---: |
| Default | A | 40 | 7.818 | 5.968 | 23.66% |
| Default | A | 80 | 15.508 | 11.773 | 24.08% |
| Default | B | 40 | 7.807 | 5.965 | 23.60% |
| Default | B | 80 | 15.505 | 11.778 | 24.04% |
| Node 0 | A | 40 | 7.826 | 5.944 | 24.05% |
| Node 0 | A | 80 | 15.484 | 11.729 | 24.25% |
| Node 0 | B | 40 | 7.807 | 5.934 | 23.98% |
| Node 0 | B | 80 | 15.550 | 11.719 | 24.64% |
| Node 1 | A | 40 | 7.845 | 5.957 | 24.06% |
| Node 1 | A | 80 | 15.638 | 11.762 | 24.78% |
| Node 1 | B | 40 | 7.824 | 5.958 | 23.86% |
| Node 1 | B | 80 | 15.548 | 11.778 | 24.25% |
| Interleave | A | 40 | 7.813 | 5.910 | 24.36% |
| Interleave | A | 80 | 15.527 | 11.639 | 25.04% |
| Interleave | B | 40 | 7.845 | 5.915 | 24.61% |
| Interleave | B | 80 | 15.446 | 11.648 | 24.59% |

### NUMA placement methodology

Four process-startup policies were tested:

| Policy | Configuration |
| --- | --- |
| Default | Inherited CPU and memory policy |
| Node 0 | `numactl --cpunodebind=0 --membind=0` |
| Node 1 | `numactl --cpunodebind=1 --membind=1` |
| Interleave | `numactl --interleave=all` |

These flags describe the requested Linux process-placement policies.

Exact physical placement of every CUDA pinned host buffer was outside the captured measurements. NCCL may also manage its worker affinities internally.

For each policy, the execution order was:

1. NCCL baseline before.
2. Pipeline with seed 6290.
3. Pipeline with seed 3090.
4. Fresh NCCL baseline after.

Inherited `NCCL_*` settings were removed to establish default tuning. A separate INFO-level invocation captured NCCL transport selection, while primary timings used normal logging.

All 18 subprocess invocations completed successfully.

### Additional portable NCCL validation

A separate portable PyTorch NCCL benchmark checked both ranks after every timed replay.

| Payload per GPU | Portable PyTorch NCCL |
| --- | ---: |
| 40 MiB | 7.824 ms |
| 80 MiB | 15.475 ms |

These timings are consistent with the primary PyNCCL reference measurements.

The default-placement repeat corresponds to approximately 1.309× and 1.316× collective speedups at 40 MiB and 80 MiB, respectively.

## How it works

Each GPU sends its local input to a pinned host buffer and receives the other GPU's input into local VRAM.

Independent CUDA send and receive streams let the source copy the next chunk out while the destination copies the previous chunk in.

A CUDA event marks each chunk ready for the receiving stream. A second event tells the sender when the peer has finished reading its host buffer.

A local BF16 addition produces the sum on each GPU.

Instead of serialising the complete outgoing and incoming transfers, the pipeline overlaps communication in smaller chunks.

```text
Traditional serial transfer:

Outgoing: [Chunk 1][Chunk 2][Chunk 3][Chunk 4]
Incoming:                                   [Chunk 1][Chunk 2][Chunk 3][Chunk 4]


Overlapping host pipeline:

Outgoing: [Chunk 1][Chunk 2][Chunk 3][Chunk 4]
Incoming:          [Chunk 1][Chunk 2][Chunk 3][Chunk 4]
```

This is a conceptual illustration, not a measured execution timeline.

The headline benchmark uses 1 MiB chunks and one all-reduce per CUDA graph launch. It measures wall-clock time from graph submission through completion.

The serial comparison uses the same local addition after copying the full payload through host memory.

This is ordinary CUDA software. It works with host staging and requires no changes to the NVIDIA driver, GPU firmware, or BIOS settings.

## Numerical correctness

The CUDA benchmark:

- Poisons output buffers before every replay.
- Regenerates deterministic seeded inputs between rounds.
- Checks the complete output on both GPUs.
- Compares every BF16 output value against the reference.
- Exits with an error if a mismatch is detected.

Every reported prototype replay passed full bitwise BF16 verification on both GPUs.

The original PyNCCL harness validates warmup and initial graph output and retains fixed inputs during timed replays.

The separate portable PyTorch NCCL harness validates both ranks after every timed replay.

These correctness results apply to the tested workloads and do not establish correctness under every possible application or concurrency pattern.

## Reproduce the CUDA test

Use Linux with two visible CUDA GPUs, a CUDA toolkit with Ampere BF16 support, and at least 1 GiB of free VRAM per GPU for benchmark buffers and CUDA contexts.

The 80 MiB case also uses 160 MiB of pinned system memory.

Keep other GPU workloads idle for timing.

```bash
nvcc -O3 -std=c++17 -arch=sm_86 staged.cu -o staged
export CUDA_DEVICE_ORDER=PCI_BUS_ID

./staged --prefill --repeat > my-prefill-initial.csv
./staged --prefill > my-prefill-repeat.csv

python3 summarize.py my-prefill-repeat.csv prefill-nccl-repeat.jsonl
```

Use `--sweep` for the smaller sizes and chunk-size sweep.

Add `--single` to time one reduction per graph in the sweep.

The two prefill commands use different deterministic seeds.

The program poisons the output before every replay, changes the input each round, and checks the complete result on both GPUs.

It exits with an error on a mismatch.

## Reproduce the NCCL comparison

The **headline reference** runs the same PyNCCL wrapper and NCCL 2.30.7 library used by the local Sonar installation.

Set `SONAR_PYTHON` to that environment's Python executable.

Run with default NCCL tuning and one process per GPU.

```bash
OMP_NUM_THREADS=1 "$SONAR_PYTHON" -m torch.distributed.run \
  --standalone --nproc-per-node=2 measure_pynccl.py --prefill \
  --mode nccl_prefill > my-nccl.jsonl

python3 summarize.py my-prefill-repeat.csv my-nccl.jsonl
```

If you have CUDA-enabled PyTorch with NCCL, use the **portable comparison** below.

It uses PyTorch's NCCL process group, reports its NCCL version, and checks both ranks after every timed replay.

Its timings are separate from the headline PyNCCL results.

The default Python environment for this command should have PyTorch installed.

```bash
OMP_NUM_THREADS=1 python3 -m torch.distributed.run \
  --standalone --nproc-per-node=2 measure_nccl.py > my-torch-nccl.jsonl

python3 summarize.py my-prefill-repeat.csv my-torch-nccl.jsonl
```

## Reproduce the NUMA placement comparison

On a Linux machine with two NUMA nodes and the `numactl` utility installed, the pipeline can be tested with different requested process-placement policies.

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1

numactl --cpunodebind=0 --membind=0 \
  ./staged --prefill > pipeline-node0.csv

numactl --cpunodebind=1 --membind=1 \
  ./staged --prefill > pipeline-node1.csv

numactl --interleave=all \
  ./staged --prefill > pipeline-interleave.csv
```

Apply the same `numactl` prefix to the NCCL reference when comparing a specific policy.

Clear inherited NCCL tuning settings before running the default NCCL reference.

The complete replication runner brackets both seeded pipeline runs with reference measurements and records exact commands, durations and exit codes.

The replication evidence package includes the runner, analyzer, hardware inventory, raw measurements, source revision and checksums.

When using the replication runner, adjust its absolute path constants for your local source checkout, Python environment and output directory. Use a fresh output directory to preserve the original evidence.

## What was measured

The input width was 5,120 BF16 values, matching the local Qwen model.

The large cases have 4,096 and 8,192 rows. They are synthetic inputs sized like model prefill activations.

| Rows | BF16 values per row | Payload per GPU |
| ---: | ---: | ---: |
| 4,096 | 5,120 | 40 MiB |
| 8,192 | 5,120 | 80 MiB |

The CUDA prototype uses two devices in one process.

Sonar's NCCL baseline uses one process per GPU and takes the slower rank's wall-clock time.

Timing includes graph submission and completion in both harnesses, with setup and output checks outside the measured region.

The original server used NVIDIA driver 595.99.02 and 300 W limits.

The dual-socket replication used NVIDIA open driver 615.71.09 and 420 W limits with normal automatic clock scaling.

The replication recorded a maximum sampled GPU temperature of 39°C. Telemetry samples covered the full benchmark processes, including CPU input generation and verification.

Per-collective energy consumption was not measured.

## Separate NCCL small-message tuning experiment

A separate Project Prometheus experiment investigated NCCL protocol selection for small tensor-parallel all-reduces relevant to speculative decoding.

| Tokens | Stock latency | Tuned latency |
| ---: | ---: | ---: |
| 1 | 25.46 µs | 14.40 µs |
| 2 | 31.41 µs | 18.77 µs |
| 4 | 34.28 µs | 27.74 µs |
| 5 | 40.13 µs | 32.42 µs |
| 6 | 41.30 µs | 38.24 µs |

Another stock baseline achieved 14.83 µs for the one-token case, indicating that automatic NCCL protocol selection influences the measured improvement.

The tuning experiment was also evaluated against actual Qwen streaming decode with MTP4 active.

| Configuration | Median decode speed |
| --- | ---: |
| Stock before | 99.81 tokens/s |
| Tuned | 99.79 tokens/s |
| Stock restored | 99.69 tokens/s |

Output digests matched for every tested prompt.

Despite improved standalone small-message latency, no measurable short-prompt single-request decode improvement was observed.

The original Sonar settings were restored.

These negative application-level results are included because faster collective communication alone does not necessarily improve end-to-end inference performance.

## Scientific significance

The replication demonstrates that the host-staged pipeline's measured improvement is not limited to the original single-socket motherboard.

Comparable relative performance gains were observed:

- Across two physical machines.
- Across different PCIe link widths.
- Across single-socket and dual-socket GPU placement.
- Across different NVIDIA driver versions.
- Across multiple requested NUMA policies.
- Across separate benchmark runs and deterministic input seeds.

The results provide evidence that overlapping host-staged transfers is a promising alternative for large BF16 collectives on these tested dual-GPU configurations where native CUDA peer access is unavailable.

The underlying principles of pipelining, pinned host memory and asynchronous CUDA communication are established techniques. This project contributes a specific implementation and reproducible measurements for the tested GPU configurations.

## Limitations and future work

- These are standalone synthetic collective communication benchmarks.
- The CUDA prototype uses one process controlling both GPUs.
- The NCCL reference uses separate worker processes.
- The original standalone comparison has different process models; a subsequent two-worker adapter benchmark provides additional evidence, but a fully matched comparison remains desirable.
- The two-worker integration is qualified for the tested Sonar 0.26.0 configuration; broader compatibility remains unverified.
- Controlled Qwen TTFT improvements have been measured; isolated prefill throughput and longer-context gains remain unverified.
- End-to-end Qwen decode improvements remain unmeasured.
- Long-context and concurrent inference correctness remain unverified.
- Long-running application-level stability remains unverified.
- Per-collective energy consumption remains unmeasured.
- Exact physical placement of pinned host allocations was not independently verified for each requested NUMA policy.
- The results do not establish superiority over NCCL across all hardware configurations, payload sizes or workloads.
- The implementation has not been validated with native CUDA P2P enabled.
- Differences in driver versions, CPU platforms and power limits prevent attributing absolute cross-machine timing differences to a single factor.

Future research may investigate broader inference-engine compatibility, long-context qualification, additional GPU architectures, different communication topologies and further scheduling optimisations.

## Original experiment files and evidence

| Files | Contents |
| --- | --- |
| [staged.cu](staged.cu) | Buildable CUDA implementation, benchmark, and correctness checks |
| [measure_pynccl.py](measure_pynccl.py), [measure_nccl.py](measure_nccl.py) | Original Sonar PyNCCL baseline and portable PyTorch baseline |
| [summarize.py](summarize.py) | Benchmark result summarisation |
| [prefill.csv](prefill.csv), [prefill-repeat.csv](prefill-repeat.csv) | Raw CUDA timings for the initial and repeat large-payload runs |
| [prefill-nccl.jsonl](prefill-nccl.jsonl), [prefill-nccl-repeat.jsonl](prefill-nccl-repeat.jsonl) | Raw PyNCCL timings used for the headline comparison |
| [prefill-torch-nccl.jsonl](prefill-torch-nccl.jsonl) | Separate portable comparison |
| [overlap-sweep.csv](overlap-sweep.csv), [overlap-single.csv](overlap-single.csv) | Chunk-size sweeps with batched and single-call graphs |
| [api-stock-before.jsonl](api-stock-before.jsonl), [api-tuned.jsonl](api-tuned.jsonl), [api-stock-after.jsonl](api-stock-after.jsonl) | Qwen A/B/A decode timings for the separate NCCL tuning experiment |

CUDA CSV fields are:

`tokens,mode,round,microseconds`

Benchmark modes:

| Mode | Implementation |
| ---: | --- |
| 0 | Serial copies |
| 1 | Mapped host reads |
| 2 | Pipelined 256 KiB copies |
| 3 | Pipelined 1 MiB copies |
| 4–8 | Copy-only pipelines with 32, 64, 128, 512 and 2,048 KiB chunks |
| 9–11 | Overlapping reduction kernels with 64, 256 and 512 KiB chunks |

Use mode 3 for the headline comparison.

## Dual-socket replication evidence

[Cross-socket replication results summary](results/dual-socket-replication/REPORT.md) — all 16 matched comparisons and methodological limitations. The table below describes the supplied evidence archive; it does not imply that every listed file has been uploaded to GitHub.

The replication evidence package contains:

| File or directory | Contents |
| --- | --- |
| `REPORT.md` | Complete replication report |
| `results.json` | Machine-readable timing results |
| `results.csv` | Summary of matched benchmark results |
| `commands.jsonl` | Exact execution commands, durations and exit codes |
| `run-comparison.py` | Replication benchmark runner |
| `analyze.py` | Result verification and analysis |
| `gpus.txt` | GPU hardware inventory |
| `topology.txt` | NVIDIA GPU topology |
| `cpu.txt` | CPU hardware inventory |
| `numa.txt` | NUMA configuration |
| `peer.txt` | CUDA peer-access capability |
| `nvcc.txt` | CUDA compiler version |
| `telemetry.jsonl` | GPU telemetry |
| `source_commit.txt` | Pinned original source commit |
| `source_status.txt` | Original source working-tree status |
| `nccl-library.sha256` | NCCL library checksum |
| `SHA256SUMS` | Evidence integrity checksums |
| `default/` | Default-placement raw measurements |
| `node0/` | Node 0 placement raw measurements |
| `node1/` | Node 1 placement raw measurements |
| `interleave/` | Interleaved placement raw measurements |
| `upstream-source/` | Original pinned source and licence |

The complete evidence package is intended to allow independent inspection and reproduction of the cross-socket results.

## Project Prometheus

Project Prometheus is an experimental research programme investigating underutilised capabilities and performance opportunities in consumer GPU hardware and software.

Rather than assigning a predetermined optimisation, the research agent is given access to physical hardware and encouraged to independently identify research opportunities, formulate hypotheses, implement experiments and validate results.

Research principles:

- Establish controlled baselines.
- Verify numerical correctness.
- Investigate unexpected results.
- Repeat experiments.
- Preserve raw evidence.
- Publish reproducible implementations.
- Acknowledge prior art.
- Document negative results and limitations.
- Distinguish measured findings from speculation.

GPT-6.2 Sol (Codex) independently selected the research direction, implemented the CUDA prototype and conducted the original experiments.

Local Qwen provided technical review of the synchronisation logic.

The human laboratory collaborator provided the physical hardware, experimental environment and publication support.

The subsequent cross-socket experiment was performed on a second physical server using the original pinned implementation.

### Related Project Prometheus research

[Wan 2.2 Text K/V Cache](https://github.com/koenjw-wq/wan22-text-kv-cache)

A reproducible ComfyUI optimisation that reduced Wan 2.2 denoising time by approximately 4.25% under CPU weight streaming, with all 17 decoded frames matching bit for bit in the reported tests.

## Technical references

- [CUDA multi-GPU programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html)
- [NCCL 2.30.7 documentation](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2307/user-guide/docs/env.html)
- [Original published benchmark revision](https://github.com/koenjw-wq/rtx3090-host-staged-allreduce/tree/a966f728545160edc65fab918bdb99c2aca64382)

## Licence

The benchmark code is available under [GPL-3.0](LICENSE).

---

**Project Prometheus — AI-directed research, physical hardware, reproducible evidence.**
