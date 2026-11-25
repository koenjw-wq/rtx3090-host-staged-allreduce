# Reproduce the Sonar two-worker integration

Experimental, opt-in integration for Sonar 0.26.0 on a single Linux host with two RTX 3090 GPUs (SM 8.6), TP=2, ordered collective execution. Read the [measured results and limitations](../results/sonar-integration/REPORT.md) before deployment. Native CUDA P2P is not required. The backend uses POSIX shared memory and CUDA pinned memory; it is not an NCCL replacement for arbitrary workloads.

## 1. Requirements and preflight

Use your existing working Sonar 0.26.0 Python environment, CUDA toolkit with BF16 support (the dual-socket build used nvcc 13.0.88), compatible NVIDIA driver and matching Torch/CUDA/NCCL libraries. Both tested systems used NCCL 2.30.7. Check the driver, GPU topology and GPU memory first. The backend allocates two 80 MiB shared host slabs plus a 4 KiB header per communicator, registers the mapping in each worker and allocates an 80 MiB receive buffer per GPU per communicator. Allow additional VRAM, /dev/shm capacity and memory-lock allowance. Both workers must see the same shared-memory namespace.

Run in an isolated environment or approved maintenance window. Stop other GPU jobs before validation. Save the effective service configuration, environment, original PYTHONPATH and any existing sitecustomize hook privately. Do not replace an existing sitecustomize hook or PYTHONPATH without merging its required behavior. Do not enable dual batch overlap or independent concurrent collective streams; this backend assumes ordered execution.

```bash
git clone https://github.com/koenjw-wq/rtx3090-host-staged-allreduce.git
cd rtx3090-host-staged-allreduce/sonar-integration
export SONAR_PYTHON=/absolute/path/to/sonar-0.26.0/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1
nvidia-smi
nvidia-smi topo -m
df -h /dev/shm
ulimit -l
"$SONAR_PYTHON" -c 'import torch; from importlib.metadata import version; print(version("sonar"), torch.__version__, torch.version.cuda, torch.cuda.nccl.version())'
sha256sum -c SHA256SUMS.source
```

## 2. Record the stock API baseline

These scripts use a local API on port 2242 and served model name qwen3.8-27b. Adapt the endpoint/model constants in copies of the benchmark scripts if your service differs. Keep any API credentials local. The original tests used an unauthenticated local endpoint; do not expose it publicly. The scripts write synthetic completion text and timing records locally, and refuse to overwrite prior evidence.

For a comparable Qwen configuration, use the same model weights, tokenizer, service settings and workload for all A/B runs. The reference deployment used Qwen 3.8 27B GPTQ W4A16, TP2, BF16 compute, FP8 KV cache, native MTP4, max model length 262144, max sequences 16, max batched tokens 8192, GPU memory utilization 0.90, prefix caching and chunked prefill. The production launch also set SONAR_USE_STANDALONE_COMPILE=0 and SONAR_USE_MEGA_AOT_ARTIFACT=0. These are recorded settings, not universal prerequisites. No weights are included in this repository.

Start your existing Sonar service with the adapter disabled and no unrelated tuning changes. Use a fresh service start for each A/B variant to reset prefix-cache state.

```bash
mkdir -p inference
python3 inference-benchmark.py stock
```

## 3. Build and validate locally

Stop Sonar and other GPU workloads before running the standalone validation.

```bash
nvcc -O3 -std=c++17 -arch=sm_86 --shared -Xcompiler -fPIC backend.cu -o libhoststaged.so
SONAR_HOST_STAGED_ALLREDUCE=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  "$SONAR_PYTHON" -m torch.distributed.run \
  --standalone --nproc-per-node=2 validate.py \
  > validation-local.jsonl 2> validation-local.stderr
cat validation-local.jsonl
```

Expect PASS on both ranks, 152 bitwise checks per rank and 24 graph replays. The test covers in-place/out-of-place SUM, boundary and partial-chunk sizes, fallback cases, changing inputs, graph replay and delayed ranks. If the command fails or correctness differs, do not install. Check memory availability, driver errors and environment compatibility.

## 4. Activate the adapter

Python imports sitecustomize at startup. When SONAR_HOST_STAGED_ALLREDUCE=1, it installs the adapter into PyNcclCommunicator before worker initialization. Only compatible contiguous BF16 SUM calls from 5 to 80 MiB inclusive use the custom backend; all other calls delegate to the original method. Startup initialization failure is fatal rather than silently continuing. A peer timeout (60 seconds) may terminate the CUDA context.

For a temporary manual test, start the existing Sonar launch command from the same environment with SONAR_HOST_STAGED_ALLREDUCE=1 and PYTHONPATH containing the absolute integration directory. Ensure that both workers inherit these variables. Do not change model or scheduler settings between runs. Run the API checks below before persistent installation.

For a systemd service, the following is an example of the qualified installation pattern. Replace SERVICE with your actual unit name. If the target directory or override already exists, stop and back it up; do not overwrite another installation. Keep service-user access and existing environment hooks intact.

```bash
SERVICE=your-sonar.service
sudo install -d -o root -g root -m 755 /opt/rtx3090-host-staged-allreduce
sudo install -o root -g root -m 644 adapter.py backend.cu sitecustomize.py libhoststaged.so ../LICENSE /opt/rtx3090-host-staged-allreduce/
```

Generate the installed manifest with absolute paths:

```bash
sudo sh -c 'cd /opt/rtx3090-host-staged-allreduce && sha256sum "$PWD/adapter.py" "$PWD/backend.cu" "$PWD/sitecustomize.py" "$PWD/libhoststaged.so" "$PWD/LICENSE" > SHA256SUMS'
sudo sha256sum --check /opt/rtx3090-host-staged-allreduce/SHA256SUMS
sudo install -d -m 755 "/etc/systemd/system/$SERVICE.d"
```

Review [the example drop-in](95-host-staged-allreduce.conf.example) before installing. If the service already needs PYTHONPATH entries, edit the example to include those literal paths; systemd Environment= is not shell expansion. The example assumes no conflicting sitecustomize module. Preserve existing startup arguments and boot enablement. Install the reviewed drop-in and restart only in an approved window:

```bash
sudo install -m 644 95-host-staged-allreduce.conf.example "/etc/systemd/system/$SERVICE.d/95-host-staged-allreduce.conf"
sudo systemctl daemon-reload
sudo systemctl restart "$SERVICE"
systemctl is-active "$SERVICE"
sudo journalctl -u "$SERVICE" --since today | grep -E 'HOST_STAGED_READY|HOST_STAGED_FIRST_USE'
curl --fail http://127.0.0.1:2242/health
```

Model loading can take several minutes. Look for HOST_STAGED_READY on both ranks, then HOST_STAGED_FIRST_USE on both ranks when an eligible collective runs. HOST_STAGED_SIZE records up to 16 distinct eligible sizes per communicator. Verify that both ranks actually use a size below 40 MiB during live requests when comparing the lower cutoff. A local opt-in variable warning from Sonar may be expected because sitecustomize handles it.

## 5. Compare real API performance and restart behavior

```bash
python3 inference-benchmark.py adapter
python3 batched-api-check.py
# Restart the service in an approved window, then:
python3 inference-benchmark.py after-restart
```

The benchmark sends seeded 5,000- and 9,000-token prompts, with one warmup and three measured requests per size and 16 generated tokens each. Compare median time to first streamed text and exact completion text across all eight records. The batch test sends four simultaneous distinct 9,000-token prompts with 64 generated tokens. Keep the service and hardware policies identical. The adapter should actually be used on both ranks; a faster result without that evidence does not establish the adapter's effect.

```bash
python3 - <<'PY'
import json
from pathlib import Path
from statistics import median
def load(label):
    return [json.loads(s) for s in Path("inference", label+".jsonl").read_text().splitlines()]
a=load("stock")
for label in ("adapter","after-restart"):
    b=load(label)
    key=lambda x:(x["prompt_tokens"],x["iteration"],x["warmup"])
    a=sorted(a,key=key)
    b=sorted(b,key=key)
    assert len(a)==len(b)==8
    assert [key(x) for x in a]==[key(x) for x in b]
    assert all(x["completion_text"]==y["completion_text"] for x,y in zip(a,b))
    for n in (5000,9000):
        p=median(x["ttft_seconds"] for x in a if not x["warmup"] and x["prompt_tokens"]==n)
        q=median(x["ttft_seconds"] for x in b if not x["warmup"] and x["prompt_tokens"]==n)
        print(label,n,p,q,100*(p-q)/p)
PY
```

This comparison helper is provided for convenience; it is not a new GPU measurement. Verify output equality, worker logs, service state and automatic restart counters. Do not infer sustained decode throughput from 16-token completions.

## 6. Rollback

Disable the adapter by moving only its drop-in out of the service directory, then reload and restart. Restore any manually changed environment or sitecustomize configuration to the saved baseline as well.

```bash
sudo install -d -m 700 /root/host-staged-allreduce-disabled
sudo mv "/etc/systemd/system/$SERVICE.d/95-host-staged-allreduce.conf" /root/host-staged-allreduce-disabled/
sudo systemctl daemon-reload
sudo systemctl restart "$SERVICE"
curl --fail http://127.0.0.1:2242/health
```

Choose a fresh backup location if the destination already exists. Confirm that the adapter startup markers no longer appear in the new service invocation, and verify normal API behavior. The native library and original source may be retained for later review.

## Scope and provenance

The six source scripts and example service override are copied byte-for-byte from the original integration handover. The source SHA-256 manifest permits verification. Build locally; no precompiled native library, private service configuration, credentials or model files are distributed. The root [GPL-3.0 license](../LICENSE) accompanies the implementation. See [results](../results/sonar-integration/REPORT.md) for measured findings. Results depend on the hardware, model, driver, workload and runtime; broader engine versions, long contexts, arbitrary collective streams and multi-node systems remain unqualified.


## Compare the 5 MiB cutoff with a previous 40 MiB deployment

Preserve the old installation and match the model, scheduler, hardware, and service settings across both arms. Run the baseline with the old cutoff and an idle model. Use a separate installation directory for the candidate, back up the existing drop-in, and adapt both the Python path and integrity-check path to that directory. Keep the existing startup hook behavior. On Ubuntu, the same-NUMA qualification chained the system Python hook before loading the adapter. Use the default hook on installations without an earlier hook to preserve.

```bash
# Run under the original 40 MiB cutoff first.
python3 api_bench.py incumbent --sizes 256 512 1024 2048 3072 5000 9000

# Switch the service to the new adapter and wait for the health endpoint.
python3 api_bench.py candidate --sizes 256 512 1024 2048 3072 5000 9000
python3 compare_api.py candidate --qualify

# Restart the candidate service, wait for health, then repeat the shorter sweep.
python3 api_bench.py restart --sizes 512 2048 3072 --rounds 3
python3 compare_api.py restart
python3 test-batch.py
```

The benchmark assigns a distinct prefix-cache salt to every request and verifies identical outputs by SHA-256. See [the two-server evidence](../results/sonar-medium-reductions/REPORT.md) for the measured gains and limits.
