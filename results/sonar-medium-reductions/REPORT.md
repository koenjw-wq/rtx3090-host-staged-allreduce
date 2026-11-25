# Sonar 5 MiB cutoff qualification

25 November 2026

The Sonar 0.26.0 two-worker adapter now accepts eligible BF16 SUM payloads from 5 through 80 MiB. The prior release began at 40 MiB. The CUDA backend still uses 1 MiB chunks, and small reductions retain NCCL. Qualification covers two RTX 3090s, TP=2, same-host ordered collectives, and the tested runtime versions.

| System | GPU topology | NVIDIA driver | Power limit per GPU |
| --- | --- | --- | ---: |
| Same NUMA node | PHB | 595.99.02 | 300 W |
| Cross-socket | SYS | 615.71.09 | 420 W |

## Live comparison after a Sonar service restart

| GPU topology | Prompt tokens | Previous TTFT | New TTFT | Less time |
| --- | ---: | ---: | ---: | ---: |
| Same NUMA node | 512 | 0.338 s | 0.321 s | +4.99% |
| Same NUMA node | 2,048 | 1.208 s | 1.133 s | +6.17% |
| Same NUMA node | 3,072 | 1.795 s | 1.639 s | +8.70% |
| Cross-socket | 512 | 0.290 s | 0.286 s | +1.23% |
| Cross-socket | 2,048 | 1.014 s | 0.961 s | +5.18% |
| Cross-socket | 3,072 | 1.482 s | 1.390 s | +6.25% |

## First live comparison

| GPU topology | Prompt tokens | Previous TTFT | New TTFT | Less time |
| --- | ---: | ---: | ---: | ---: |
| Same NUMA node | 256 | 0.192 s | 0.198 s | -3.46% |
| Same NUMA node | 512 | 0.338 s | 0.327 s | +3.30% |
| Same NUMA node | 1,024 | 0.622 s | 0.573 s | +7.76% |
| Same NUMA node | 2,048 | 1.200 s | 1.108 s | +7.62% |
| Same NUMA node | 3,072 | 1.795 s | 1.623 s | +9.56% |
| Same NUMA node | 5,000 | 2.691 s | 2.633 s | +2.12% |
| Same NUMA node | 9,000 | 4.921 s | 4.733 s | +3.83% |
| Cross-socket | 256 | 0.214 s | 0.203 s | +5.17% |
| Cross-socket | 512 | 0.289 s | 0.287 s | +0.77% |
| Cross-socket | 1,024 | 0.516 s | 0.491 s | +4.84% |
| Cross-socket | 2,048 | 1.014 s | 0.951 s | +6.22% |
| Cross-socket | 3,072 | 1.482 s | 1.395 s | +5.91% |
| Cross-socket | 5,000 | 2.263 s | 2.269 s | -0.27% |
| Cross-socket | 9,000 | 4.054 s | 3.985 s | +1.70% |

Each row is a median, excluding the warmup. The initial arm used five timed requests per prompt length. The restart arm used three timed requests, compared with the same prompt IDs from the old-cutoff run. Each request used a fresh prefix-cache salt and returned 24 tokens. The baseline already had the 40–80 MiB adapter. The comparisons report time to first streamed text. Sustained decode throughput was outside this test.

Both passes matched all 54 paired responses per server by text digest and token counts. Both workers logged newly eligible sizes after API startup. Each rank passed 152 bitwise comparisons against NCCL and 24 changing-input CUDA graph replays per server. Four concurrent requests completed on each server after restart. Both installed services passed the startup integrity checks with zero automatic restarts during qualification.

The same-NUMA service remains manually started to preserve its ComfyUI boot setup. The cross-socket service retains its existing boot enablement. Qualification used service restarts. Neither machine was rebooted for this test.

## Raw evidence

- [Same-NUMA logs](same-numa/) and [cross-socket logs](cross-numa/) contain paired request timings, response digests, batching and verification records.
- The `collective-*.jsonl` files contain the separate two-worker CUDA graph sweep. The same-NUMA test sampled 512 KiB, 1 MiB, and 2 MiB chunk sizes. The cross-socket test confirmed the selected 1 MiB chunk size.
- The [build and reproduction guide](../../sonar-integration/README.md) includes the qualification, service integration, and rollback sequence.
