# Sonar 0.26.0 integration — 25 November 2026

The host-staged BF16 all-reduce has progressed from a single-process prototype to a two-worker Sonar adapter. It selects contiguous BF16 SUM collectives of 40–80 MiB inclusive on the qualified two-GPU configuration; other sizes, dtypes and operations retain the original NCCL path. Small decode reductions continue to use NCCL.

## Cross-socket server (PCIe Gen3 x16, SYS)

The supplied integration handover records 120 bitwise comparisons per rank against NCCL, including in-place/out-of-place operations, boundary sizes, fallback cases and changing-input CUDA graph replays. The integrated two-process microbenchmark measured 6.504 ms at 40 MiB and 12.698 ms at 80 MiB, versus earlier NCCL measurements of approximately 7.81 and 15.51 ms (approximately 17–18% less time).

| Prompt tokens | Baseline median TTFT | Initial adapter TTFT | Initial reduction | After restart TTFT | Reduction after restart |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5,000 | 2.465 s | 2.263 s | 8.22% | 2.262 s | 8.24% |
| 9,000 | 4.323 s | 4.040 s | 6.55% | 4.055 s | 6.22% |

One warmup and three measured requests per size, seeded synthetic prompts, 16 generated tokens. All eight completion texts (including warmups) matched baseline in both comparisons. Four simultaneous distinct 9,000-token requests with 64 generated tokens completed successfully. Both workers loaded and exercised the adapter following service restart; zero automatic restarts were recorded during qualification. Boot enablement was verified, but an actual server reboot was not tested.

## Single-socket server (PCIe Gen3 x8, same NUMA node)

A subsequent deployment report supplied by the owner records:

| Prompt tokens | Baseline TTFT | After restart TTFT | Reduction |
| ---: | ---: | ---: | ---: |
| 5,000 | 2.952 s | 2.661 s | 9.9% |
| 9,000 | 5.264 s | 4.921 s | 6.5% |

Both workers loaded and exercised the adapter during API requests. All eight benchmark responses matched baseline in both runs, four simultaneous requests passed, and zero automatic restarts were reported. The adapter loads whenever Sonar starts; Sonar itself remains manually started on this machine so the existing ComfyUI boot behavior is preserved.

These single-socket figures are from the deployment status report, not independently verified raw timing files. The supplied handover contains the cross-socket integration evidence.

## Implementation and limitations

The backend uses shared pinned host memory, 1 MiB chunks, independent copy streams, GPU-visible progress flags and local BF16 addition. Each rank allocates an 80 MiB receive buffer; the shared host mapping contains two 80 MiB slabs plus a header. The adapter is opt-in and version-guarded for Sonar 0.26.0. The backend uses a 60-second device watchdog; peer failure may terminate the CUDA context.

Qualification is limited to the tested ordered collective execution model. Arbitrary independent collective streams, dual batch overlap, additional ranks, multi-node execution, engine upgrades, long-context inference and sustained decode performance need separate validation. Service restart is not equivalent to full hardware reboot qualification. TTFT reductions are not decode throughput gains or universal prefill speedups.

This public summary excludes private deployment paths, credentials, service configuration, model metadata and compiled binaries. The full integration source and raw evidence are not included in this summary.

See the [main README](../../README.md) for the original standalone experiments.
