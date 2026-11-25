# Dual-socket replication — 25 November 2026

Publication summary of the supplied replication report; not the original raw evidence archive.

The original host-staged BF16 all-reduce was reproduced on an HPE ProLiant DL360 Gen10 with two Xeon Silver 4110 sockets and two RTX 3090s on separate NUMA nodes (SYS topology, PCIe Gen3 x16). NVIDIA open driver/GSP 615.71.09, NCCL 2.30.7, Sonar 0.26.0, kernel 7.0.0-42-generic, 420 W power limits. CUDA peer access was unavailable in both directions; NCCL reported SHM/direct transport.

Source revision: a966f728545160edc65fab918bdb99c2aca64382.

| Requested placement | Run | MiB per GPU | NCCL ms | Pipeline ms | Less time |
| --- | --- | ---: | ---: | ---: | ---: |
| default | A | 40 | 7.818 | 5.968 | 23.66% |
| default | A | 80 | 15.508 | 11.773 | 24.08% |
| default | B | 40 | 7.807 | 5.965 | 23.60% |
| default | B | 80 | 15.505 | 11.778 | 24.04% |
| node0 | A | 40 | 7.826 | 5.944 | 24.05% |
| node0 | A | 80 | 15.484 | 11.729 | 24.25% |
| node0 | B | 40 | 7.807 | 5.934 | 23.98% |
| node0 | B | 80 | 15.550 | 11.719 | 24.64% |
| node1 | A | 40 | 7.845 | 5.957 | 24.06% |
| node1 | A | 80 | 15.638 | 11.762 | 24.78% |
| node1 | B | 40 | 7.824 | 5.958 | 23.86% |
| node1 | B | 80 | 15.548 | 11.778 | 24.25% |
| interleave | A | 40 | 7.813 | 5.910 | 24.36% |
| interleave | A | 80 | 15.527 | 11.639 | 25.04% |
| interleave | B | 40 | 7.845 | 5.915 | 24.61% |
| interleave | B | 80 | 15.446 | 11.648 | 24.59% |

Each median uses seven samples. For each requested policy, fresh NCCL baselines bracketed the two pipeline runs (seeds 6290 and 3090). Mode 3 uses 1 MiB chunks and one all-reduce per graph launch. Full BF16 prototype outputs matched the reference bit for bit on both GPUs. All 18 subprocesses completed successfully. Portable PyTorch NCCL measured 7.824 and 15.475 ms.

The requested policies do not establish actual physical placement of every pinned buffer. The candidate uses one process while NCCL uses two. Driver, platform and power differences confound absolute cross-machine comparisons. No end-to-end Qwen speedup has been established.

See the [main README](../../README.md) for reproduction commands and limitations.
