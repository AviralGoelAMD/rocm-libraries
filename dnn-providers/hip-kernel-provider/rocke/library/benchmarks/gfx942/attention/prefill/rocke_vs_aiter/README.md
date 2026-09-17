# ROCKE vs CK vs AITER Attention Benchmark — MI300X

This benchmark compares forward-attention performance across:

* **ROCKE Dense Attention**
* **Composable Kernel (CK) Tile FMHA**
* **AITER FMHA v3 ASM**

The benchmark targets **AMD Instinct MI300X (`gfx942`)** and evaluates BF16 causal attention across different batch sizes, sequence lengths, and GQA configurations.

## Benchmark Environment

The reported results were collected on an AMD Instinct MI300X-class `gfx942` system using the following software stack:

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

The benchmark compares equivalent forward-attention workloads across all three implementations.

Common configuration:

```text
GPU architecture: gfx942
Target GPU:       AMD Instinct MI300X
Datatype:         BF16
Head dimension:   128
Query heads:      32
Attention:        Causal
Sq:               Sk
Tensor layout:    BSHD
Warmup runs:      10
Measured runs:    50
Reported latency: Average latency per kernel launch
```

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

### ROCKE

ROCKE Dense Attention is run directly from the ROCm Libraries checkout containing this benchmark.

The benchmark uses the production ROCKE dispatch path:

```text
dense_request
    ↓
resolve_dense_spec
    ↓
run
```

For every configuration, the resolved ROCKE kernel specification is obtained through:

```text
describe_dense_spec(...)
```

and recorded in the generated benchmark TSV/log files.

This is important because ROCKE may select a kernel based on the requested workload rather than relying on a single hard-coded kernel name.

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

Each implementation executes the same logical forward-attention workload for each tested shape.

The default timing procedure is:

```text
Warmup iterations: 10
Measured iterations: 50
```

Reported latency is the average execution time of one forward-attention invocation as reported by the respective benchmark path.

The benchmark does not include compilation or environment setup time in the reported kernel latency.

ROCKE, CK, and AITER are executed separately using their respective benchmark environments.

The benchmark script also saves the raw output for each configuration so that selected kernel names and reported timings can be inspected independently.

## Results

Lower latency is better.

|  # |  B |     S | Hq | Hkv | GQA |    ROCKE ms |   CK ms | AITER ASM ms | CK vs ROCKE | AITER vs ROCKE | AITER vs CK |
| -: | -: | ----: | -: | --: | --: | ----------: | ------: | -----------: | ----------: | -------------: | ----------: |
|  1 |  1 |  4096 | 32 |   8 | 4:1 |      0.6490 |  0.3490 |       0.2860 |       1.86× |          2.27× |       1.22× |
|  2 |  1 |  4096 | 32 |  16 | 2:1 |      0.6471 |  0.3560 |       0.2760 |       1.82× |          2.34× |       1.29× |
|  3 |  1 |  8192 | 32 |   8 | 4:1 |      2.0028 |  1.1870 |       0.9890 |       1.69× |          2.03× |       1.20× |
|  4 |  1 |  8192 | 32 |  16 | 2:1 |      2.0040 |  1.1940 |       0.9940 |       1.68× |          2.02× |       1.20× |
|  5 |  1 | 16384 | 32 |   8 | 4:1 |      7.6924 |  4.5400 |       3.9500 |       1.69× |          1.95× |       1.15× |
|  6 | 16 |  4096 | 32 |   8 | 4:1 |      7.2652 |  5.1780 |       4.2010 |       1.40× |          1.73× |       1.23× |
|  7 | 16 |  8192 | 32 |   8 | 4:1 |     28.0588 | 19.2430 |      15.5580 |       1.46× |          1.80× |       1.24× |
|  8 | 16 |  4096 | 32 |  16 | 2:1 |      7.7335 |  5.6500 |       4.2320 |       1.37× |          1.83× |       1.34× |
|  9 | 64 |  4096 | 32 |   8 | 4:1 |     31.0322 | 20.9060 |      16.8720 |       1.48× |          1.84× |       1.24× |
| 10 | 64 |  8192 | 32 |   8 | 4:1 | unsupported | 78.4520 |      63.4220 |           — |              — |       1.24× |

## Speedup Calculation

Speedups are calculated from measured latency:

```text
CK vs ROCKE       = ROCKE latency / CK latency
AITER vs ROCKE    = ROCKE latency / AITER latency
AITER vs CK       = CK latency / AITER latency
```

For example:

```text
B   = 1
S   = 4096
Hq  = 32
Hkv = 8
```

Measured latency:

```text
ROCKE:     0.649 ms
CK:        0.349 ms
AITER ASM: 0.286 ms
```

Therefore:

```text
CK vs ROCKE:
0.649 / 0.349 = 1.86x

AITER vs ROCKE:
0.649 / 0.286 = 2.27x

AITER vs CK:
0.349 / 0.286 = 1.22x
```

## Observations

Across the tested configurations, both CK and AITER have lower measured latency than the current ROCKE Dense Attention implementation.

AITER FMHA v3 ASM has the lowest measured latency for every supported configuration in this benchmark.

Across the tested shapes:

```text
CK vs ROCKE:
1.37x – 1.86x

AITER vs ROCKE:
1.73x – 2.34x

AITER vs CK:
1.15x – 1.34x
```

The relative latency difference between ROCKE and the optimized implementations generally becomes smaller at larger batch sizes.

For:

```text
B=64
S=8192
Hq=32
Hkv=8
```

the ROCKE implementation reports the workload as unsupported, while the tested CK and AITER implementations execute successfully.

## Benchmark Outputs

Each benchmark run creates a timestamped results directory under:

```text
~/gpu-bench/results/
```

The output contains the environment metadata, raw measurements, generated comparison table, and per-implementation logs.

Conceptually:

```text
results/
└── attention_<timestamp>/
    ├── environment.txt
    ├── configs.tsv
    ├── rocke.tsv
    ├── ck.tsv
    ├── aiter.tsv
    └── logs/
        ├── rocke/
        ├── ck/
        └── aiter/
```

`environment.txt` records the hardware/software environment and exact repository revisions used for the run.

The TSV files record per-shape measurements, status, and selected kernel information.

The log directories contain the raw benchmark output and commands for inspection.

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
2. Runs the 10 ROCKE configurations.
3. Runs the same 10 configurations through AITER FMHA-v3 ASM.
4. Runs the same 10 configurations through CK Tile FMHA.
5. Verifies the expected CK/AITER kernel paths.
6. Saves raw per-implementation results and logs.
7. Produces the three-way latency comparison.

## Benchmark Files

```text
rocke_vs_aiter/
├── README.md
├── setup_mi300x_attention_bench.sh
└── run_mi300x_attention_bench_updated.sh
```

`setup_mi300x_attention_bench.sh` prepares the environments, dependencies, and kernel builds required by the benchmark.

`run_mi300x_attention_bench_updated.sh` executes ROCKE, CK, and AITER across the same 10 attention workloads and records the environment, selected kernels, raw measurements, and comparison results.
