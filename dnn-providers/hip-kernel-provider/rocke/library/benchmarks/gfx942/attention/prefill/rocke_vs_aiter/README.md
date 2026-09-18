# Four-Way rocKE, AITER, and CK Attention Benchmark — MI300X

This benchmark compares four forward-attention paths:

* **rocKE `attention_dense`**
* **rocKE auto-unified attention**
* **AITER FMHA-v3 ASM**
* **Composable Kernel (CK) Tile FMHA**

The benchmark targets **AMD Instinct MI300X (`gfx942`)** and evaluates BF16 causal grouped-query attention (GQA) across different batch sizes, sequence lengths, and key/value-head counts.

## Benchmark Environment

Each benchmark run records the hardware and software environment used to produce its results. The benchmark is intended for an AMD Instinct MI300X-class `gfx942` system with the following software stack:

```text
GPU:                AMD Instinct MI300X
GPU architecture:   gfx942
ROCm wheel version: 10.0.0
PyTorch:            2.13.0+rocm10.0.0
HIP:                7.15.26333
Triton:             3.8.0+git4cff872c.rocm10.0.0
Python:             3.12
```

The setup and benchmark scripts additionally record the exact environment used for each run, including:

```text
ROCm Libraries commit
ROCm Libraries dirty state
AITER commit
AITER dirty state
Composable Kernel commit
System ROCm version
CK compiler version
GPU name
GPU architecture
Compute-unit count
PyTorch version
HIP version
Triton version
```

This metadata is written to the benchmark output directory so that each result can be associated with the exact software and hardware configuration that produced it.

## Setup

Create a workspace:

```bash
mkdir -p ~/gpu-bench
cd ~/gpu-bench
```

Clone this ROCm Libraries fork:

```bash
git clone https://github.com/Parthiv911/rocm-libraries.git
```

Move to the benchmark directory:

```bash
cd rocm-libraries/dnn-providers/hip-kernel-provider/rocke/library/benchmarks/gfx942/attention/prefill/rocke_vs_aiter
```

Make the setup and benchmark scripts executable:

```bash
chmod +x setup_mi300x_attention_bench.sh
chmod +x run_mi300x_attention_bench_updated.sh
```

Run the setup script:

```bash
./setup_mi300x_attention_bench.sh
```

Then run the benchmark:

```bash
./run_mi300x_attention_bench_updated.sh
```

The setup script clones and prepares the external AITER dependency next to the `rocm-libraries` checkout. Composable Kernel is taken from the CK submodule pinned by the selected AITER revision.

The resulting workspace is approximately:

```text
~/gpu-bench/
├── rocm-libraries/
├── aiter/
│   └── 3rdparty/
│       └── composable_kernel/
├── env/
└── results/
```

## Benchmark Configuration

The benchmark compares the same logical forward-attention workload across four independently reported arms.

Common configuration:

```text
GPU architecture: gfx942
Target GPU:       AMD Instinct MI300X
Datatype:         BF16
Head dimension:   128
Query heads:      32
Attention:        Causal
Sq:               Sk
Logical layout:   contiguous dense BSHD
Warmup runs:      10
Measured runs:    50
Reported latency: Average latency per kernel launch
```

The auto-unified arm uses page-backed K/V internally, but it does **not** benchmark a paged-attention workload. It performs one identity-ordered conversion from the contiguous dense BSHD K/V tensors into page caches before timing; that conversion is outside the timed region. The logical operation for every arm remains non-paged dense BSHD causal attention.

The tested configurations vary:

* Batch size (`B`)
* Sequence length (`S`)
* Number of key/value heads (`Hkv`)
* GQA ratio

The query-head count remains fixed at:

```text
Hq = 32
```

The benchmark evaluates the following 10 shapes:

|  # |  B |     S | Hq | Hkv | GQA |
| -: | -: | ----: | -: | --: | --: |
|  1 |  1 |  4096 | 32 |   8 | 4:1 |
|  2 |  1 |  4096 | 32 |  16 | 2:1 |
|  3 |  1 |  8192 | 32 |   8 | 4:1 |
|  4 |  1 |  8192 | 32 |  16 | 2:1 |
|  5 |  1 | 16384 | 32 |   8 | 4:1 |
|  6 | 16 |  4096 | 32 |   8 | 4:1 |
|  7 | 16 |  8192 | 32 |   8 | 4:1 |
|  8 | 16 |  4096 | 32 |  16 | 2:1 |
|  9 | 64 |  4096 | 32 |   8 | 4:1 |
| 10 | 64 |  8192 | 32 |   8 | 4:1 |

## Implementations and Kernels

### rocKE `attention_dense`

The `attention_dense` arm runs directly from the ROCm Libraries checkout containing this benchmark. It uses the production rocKE dense dispatch path:

```text
dense_request
    ↓
resolve_dense_spec
    ↓
run
```

For every configuration, the benchmark resolves the dense kernel specification with `resolve_dense_spec(...)` and records the selected kernel and settings in `rocke_dense.tsv`.

Configuration 10 (`B=64`, `S=8192`, `Hkv=8`) is expected `attention_dense` coverage: its 32-bit extent is reported as `UNSUPPORTED`. This is a result for the dense arm only; the auto-unified, AITER, and CK arms are still executed and reported independently.

### rocKE auto-unified attention

The auto-unified arm runs rocKE's normal automatic route with the identity-paged K/V representation described above. It records its selected path, concrete kernel, and settings in `rocke_unified.tsv`; it does not substitute a paged logical workload for the dense BSHD operation.

The two rocKE arms share one deterministic BF16 Q/K/V fixture for each configuration. Before timing, each arm produces one output and gates its timed record against the same FP32 causal GQA scaled dot-product attention (SDPA) reference. A rocKE row passes only when `max_abs < 4e-2`; a failed or errored row has no passing timing for downstream comparisons.

### Composable Kernel

Composable Kernel is benchmarked using the standalone **CK Tile FMHA forward** implementation.

The benchmark setup intentionally builds only the CK kernel instance used by this comparison:

```text
fmha_fwd_d128_bf16_batch_b128x128x32x128x32x128_r4x1x1_r4x1x1_w32x32x16_w32x32x16_qr_async_vr_psddv_nlogits_nbias_mask_nlse_ndropout_nskip_nqscale_ntrload_nsink
```

Relevant configuration:

```text
Datatype:       BF16
Head dimension: 128
Causal:         Yes
Layout:         BSHD
V layout:       Row-major
num_splits:     1
```

The benchmark script verifies that the expected CK kernel is actually selected before accepting the timing result.

### AITER

AITER is benchmarked using the optimized **FMHA v3 assembly path** for `gfx942`.

The AITER checkout used by the setup script is pinned to:

```text
cdf6ee88a128c2c160b0512fe37cfb67be161b1d
```

Relevant configuration:

```text
FMHA path:       v3 ASM
Architecture:    gfx942
Datatype:        BF16
Head dimension:  128
Causal:          Yes
Layout:          BSHD
BF16 conversion: RTZ
```

The expected assembly kernel contains:

```text
fmha_fwd_hd128_bf16_causal_rtz
```

The benchmark performs an AITER FMHA-v3 support check separately from the timed run and verifies that the expected assembly kernel was loaded.

The synthetic timing line produced by AITER's `is_v3_check` path is **not** used as a benchmark result.

## Benchmark Methodology

Each arm executes the same logical BF16 causal GQA forward-attention operation for each tested shape. The rocKE pair shares a deterministic BF16 Q/K/V fixture. AITER and CK run in their native GPU benchmark environments, so they cannot consume that in-process rocKE fixture.

Before timing:

1. rocKE `attention_dense` and auto-unified each run once and compare their output with an FP32 causal GQA SDPA reference. Each requires `max_abs < 4e-2`.
2. AITER checks FMHA-v3 ASM support and confirms the expected assembly kernel. CK confirms the expected Tile FMHA kernel.
3. AITER and CK run their original native GPU timer commands with `-v=0`.

### External numeric-validation limitation

The native external `-v=2` GPU validator is known invalid for this BSHD workload. On reproduced configuration 1, AITER FMHA-v3 ASM and CK produced bit-identical outputs—starting `out[0]=0.3535156`, `out[1]=0.2597656`—while only the validator reference diverged (`ref[0]=0.4980469`, `ref[1]=0.4902344`) and both commands returned exit `254`. An independent AITER MHA reproduction with `Hq=Hkv=32` also returns `254`. This cross-kernel agreement identifies the experimental BSHD reference as faulty, not the timed kernels; the normal runner therefore never invokes `-v=2` or writes external validation logs.

External timing rows use `validation_status=UNAVAILABLE` and record this evidence as their reason. An external `status=PASS` means the selected kernel timed successfully; it is **not** an external numeric-output certification. rocKE remains FP32-gated as described above.

The timing procedure is:

```text
Warmup iterations: 10
Measured iterations: 50
```

Reported latency is the average execution time of one forward-attention invocation as reported by the respective benchmark path. Compilation, environment setup, rocKE's one-time identity-paged conversion, and AITER's support check are outside the timed latency.

Each arm is recorded independently. A status of `PASS`, `FAIL`, `ERROR`, `UNSUPPORTED`, or a missing row is preserved rather than inferred from another arm.

## Results

Lower latency is better. This README intentionally contains no measured performance table: run-specific results belong to the timestamped output directory and must not be generalized across MI300X systems, software revisions, or benchmark runs.

`results.csv` and `benchmark_results.md` merge the four independently recorded arms:

* rocKE `attention_dense`
* rocKE auto-unified attention
* AITER FMHA-v3 ASM
* CK Tile FMHA

For each configuration, the report first selects the fastest **passing** rocKE path between `attention_dense` and auto-unified. Only then does it calculate external ratios against AITER and CK. A ratio is emitted only when both input timings are finite, positive, and have `status=PASS`; `validation_status=UNAVAILABLE` permits external timing rows but remains visible in the report with its reason. The report includes each arm's status, validation status where available, selected kernel, and recorded reason, so it does not imply numeric certification for external rows.

The rocKE-internal comparison is:

```text
dense vs unified = attention_dense latency / auto-unified latency
```

The external comparisons use the selected passing rocKE latency:

```text
AITER vs best rocKE = best passing rocKE latency / AITER latency
CK vs best rocKE    = best passing rocKE latency / CK latency
```

For configuration 10 (`B=64`, `S=8192`, `Hkv=8`), `attention_dense` is expected to appear as the 32-bit-extent `UNSUPPORTED` coverage result. The report retains that dense status while independently preserving the auto-unified, AITER, and CK outcomes; it does not fabricate a dense timing or a ratio that requires one.

## Benchmark Outputs

Each benchmark run creates a timestamped results directory under:

```text
~/gpu-bench/results/
```

The output contains environment metadata, the four arm TSVs, merged machine- and human-readable reports, and per-arm logs:

```text
results/
└── attention_<timestamp>/
    ├── environment.txt
    ├── configs.tsv
    ├── rocke_dense.tsv
    ├── rocke_unified.tsv
    ├── aiter.tsv
    ├── ck.tsv
    ├── results.csv
    ├── benchmark_results.md
    └── logs/
        ├── rocke/
        │   └── all.log
        ├── aiter/
        │   ├── all.log
        │   ├── config_XX.support.log
        │   └── config_XX.log
        └── ck/
            ├── all.log
            └── config_XX.log
```

`rocke_dense.tsv` and `rocke_unified.tsv` include each rocKE arm's status, `max_abs`, selected kernel, path, settings, and reason. The shared-fixture rocKE validation records appear in those TSVs and in `logs/rocke/all.log`.

`aiter.tsv` and `ck.tsv` include each external arm's timing status, `validation_status`, selected kernel, and reason. Passing external rows report `validation_status=UNAVAILABLE`: the BSHD `-v=2` reference diverged while the AITER and CK outputs matched bit-for-bit. Their `config_XX.log` files preserve the `-v=0` timing output. AITER also writes `config_XX.support.log`. `results.csv` is the four-arm machine-readable merge, and `benchmark_results.md` is the corresponding Markdown report.

## Reproduction

From a fresh MI300X machine:

```bash
mkdir -p ~/gpu-bench
cd ~/gpu-bench

git clone https://github.com/Parthiv911/rocm-libraries.git

cd rocm-libraries/dnn-providers/hip-kernel-provider/rocke/library/benchmarks/gfx942/attention/prefill/rocke_vs_aiter

chmod +x setup_mi300x_attention_bench.sh
chmod +x run_mi300x_attention_bench_updated.sh

./setup_mi300x_attention_bench.sh
./run_mi300x_attention_bench_updated.sh
```

The setup script:

1. Uses the current ROCm Libraries checkout containing this benchmark.
2. Creates the required ROCm/Python environments.
3. Clones AITER at the benchmarked revision.
4. Initializes AITER's pinned Composable Kernel submodule.
5. Builds AITER FMHA-v3 ASM.
6. Builds the required CK Tile FMHA kernel.
7. Records repository and environment metadata.

The benchmark script then:

1. Records the runtime environment.
2. Runs both rocKE arms from a shared deterministic fixture for each of the 10 configurations.
3. Runs the same 10 configurations through AITER FMHA-v3 ASM with its `-v=0` GPU timer, then verifies ASM support and the expected assembly kernel.
4. Runs the same 10 configurations through CK Tile FMHA with its `-v=0` GPU timer and verifies the expected CK kernel.
5. Saves the four raw per-arm result TSVs and timing/support logs.
6. Produces the four-arm CSV and Markdown reports, selecting the fastest passing rocKE path before external ratios.

## Benchmark Files

```text
rocke_vs_aiter/
├── README.md
├── setup_mi300x_attention_bench.sh
└── run_mi300x_attention_bench_updated.sh
```

`setup_mi300x_attention_bench.sh` prepares the environments, dependencies, and kernel builds required by the benchmark.

`run_mi300x_attention_bench_updated.sh` executes rocKE `attention_dense`, rocKE auto-unified, CK, and AITER across the same 10 logical dense BSHD attention workloads and records the environment, selected kernels, rocKE FP32 validation evidence, external numeric-validation availability, raw measurements, and four-arm comparison results.
