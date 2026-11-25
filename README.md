# Host-staged BF16 all-reduce on two RTX 3090s

Overlapping CUDA copies through pinned system memory reduced the time for large, two-GPU BF16 all-reduces by **23–26%** versus default NCCL 2.30.7 on this machine. Both RTX 3090s were on the same NUMA node, behind PCIe host bridges. CUDA reported direct peer access unavailable in both directions.

| Payload per GPU | Default NCCL | Host pipeline | Less elapsed time |
| --- | ---: | ---: | ---: |
| 40 MiB | 10.754 ms | 8.079 ms | 24.9% |
| 80 MiB | 21.654 ms | 15.974 ms | 26.2% |

These are the repeat-run medians. The initial run measured 10.704 → 8.218 ms and 21.422 → 16.181 ms, respectively. Each median uses seven samples. Every prototype output matched its BF16 reference bit for bit on both GPUs.

## How it works

Each GPU sends its local input to a pinned host buffer and receives the other GPU's input into local VRAM. Independent CUDA send and receive streams let the source copy the next chunk out while the destination copies the previous chunk in. A CUDA event marks each chunk ready for the receiving stream. A second event tells the sender when the peer has finished reading its host buffer. A local BF16 addition produces the sum on each GPU.

The headline benchmark uses 1 MiB chunks and one all-reduce per CUDA graph launch. It measures wall-clock time from graph submission through completion. The serial comparison uses the same local addition after copying the full payload through host memory. This is ordinary CUDA software. It works with host staging and requires no changes to the NVIDIA driver, GPU firmware, or BIOS settings.

## Reproduce the CUDA test

Use Linux with two visible CUDA GPUs, a CUDA toolkit with Ampere BF16 support, and at least 1 GiB of free VRAM per GPU for benchmark buffers and CUDA contexts. The 80 MiB case also uses 160 MiB of pinned system memory. Keep other GPU workloads idle for timing.

```bash
nvcc -O3 -std=c++17 -arch=sm_86 staged.cu -o staged
export CUDA_DEVICE_ORDER=PCI_BUS_ID
./staged --prefill --repeat > my-prefill-initial.csv
./staged --prefill > my-prefill-repeat.csv
python3 summarize.py my-prefill-repeat.csv prefill-nccl-repeat.jsonl
```

Use `--sweep` for the smaller sizes and chunk-size sweep. Add `--single` to time one reduction per graph in the sweep. The two prefill commands use different deterministic seeds. The program poisons the output before every replay, changes the input each round, and checks the complete result on both GPUs. It exits with an error on a mismatch.

## Reproduce the NCCL comparison

The **headline reference** runs the same PyNCCL wrapper and NCCL 2.30.7 library used by the local Sonar installation. Set `SONAR_PYTHON` to that environment's Python executable. Run with default NCCL tuning and one process per GPU.

```bash
OMP_NUM_THREADS=1 "$SONAR_PYTHON" -m torch.distributed.run \
  --standalone --nproc-per-node=2 measure_pynccl.py --prefill \
  --mode nccl_prefill > my-nccl.jsonl
python3 summarize.py my-prefill-repeat.csv my-nccl.jsonl
```

If you have CUDA-enabled PyTorch with NCCL, use the **portable comparison** below. It uses PyTorch's NCCL process group, reports its NCCL version, and checks both ranks after every timed replay. Its timings are separate from the headline PyNCCL results. The default Python environment for this command should have PyTorch installed.

```bash
OMP_NUM_THREADS=1 python3 -m torch.distributed.run \
  --standalone --nproc-per-node=2 measure_nccl.py > my-torch-nccl.jsonl
python3 summarize.py my-prefill-repeat.csv my-torch-nccl.jsonl
```

## What was measured

The input width was 5,120 BF16 values, matching the local Qwen model. The large cases have 4,096 and 8,192 rows. They are synthetic inputs sized like model prefill activations. The CUDA prototype uses two devices in one process. Sonar's NCCL baseline uses one process per GPU and takes the slower rank's wall-clock time. Timing includes graph submission and completion in both harnesses, with setup and output checks outside the measured region. The server used NVIDIA driver 595.99.02 and 300 W limits.

Sonar has not been integrated with the copy pipeline, so end-to-end prefill and decode speedups are unmeasured. A separate small-message NCCL tuning experiment left single-request Qwen decode near 100 tokens/s. Its raw data is included because faster collectives alone did not improve that workload. Direct PCIe peer access remained unavailable.

## Files and evidence

| Files | Contents |
| --- | --- |
| [staged.cu](staged.cu) | Buildable CUDA implementation, benchmark, and correctness checks |
| [measure_pynccl.py](measure_pynccl.py), [measure_nccl.py](measure_nccl.py) | Original Sonar PyNCCL baseline and portable PyTorch baseline |
| [prefill.csv](prefill.csv), [prefill-repeat.csv](prefill-repeat.csv) | Raw CUDA timings for the initial and repeat large-payload runs |
| [prefill-nccl.jsonl](prefill-nccl.jsonl), [prefill-nccl-repeat.jsonl](prefill-nccl-repeat.jsonl) | Raw PyNCCL timings used for the headline comparison |
| [prefill-torch-nccl.jsonl](prefill-torch-nccl.jsonl) | Separate portable comparison |
| [overlap-sweep.csv](overlap-sweep.csv), [overlap-single.csv](overlap-single.csv) | Chunk-size sweeps with batched and single-call graphs |
| [api-stock-before.jsonl](api-stock-before.jsonl), [api-tuned.jsonl](api-tuned.jsonl), [api-stock-after.jsonl](api-stock-after.jsonl) | Qwen A/B/A decode timings for the separate NCCL tuning experiment |

CUDA CSV fields are `tokens,mode,round,microseconds`. Mode 0 is serial copies. Mode 1 uses mapped host reads. Modes 2 and 3 pipeline 256 KiB and 1 MiB copies. Sweep modes 4–8 pipeline 32, 64, 128, 512 and 2,048 KiB copies. Sweep modes 9–11 also overlap the reduction kernel, using 64, 256 and 512 KiB chunks. Use mode 3 for the headline comparison.

[CUDA multi-GPU programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html) · [NCCL 2.30.7 documentation](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2307/user-guide/docs/env.html)

Project Prometheus. GPT-6.2 Sol (Codex) implemented and tested the prototype. Local Qwen reviewed the synchronization logic.

The benchmark code is available under [GPL-3.0](LICENSE).
