# ROCKE vs CK vs AITER Attention Benchmark — MI300X

This benchmark compares forward attention performance across:

* **ROCKE Dense Attention**
* **Composable Kernel (CK) FMHA**
* **AITER FMHA v3 ASM**

The benchmark targets **AMD Instinct MI300X (`gfx942`)** and evaluates BF16 causal attention across different batch sizes, sequence lengths, and GQA configurations.

## Setup

Create a workspace:

```bash
mkdir -p ~/gpu-bench
cd ~/gpu-bench
```

Clone the ROCm libraries fork:

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

## Benchmark Configuration

The benchmark compares equivalent forward-attention workloads across the three implementations.

Common configuration:

```text
GPU architecture: gfx942
Target GPU:       AMD Instinct MI300X
Datatype:         BF16
Head dimension:   128
Query heads:      32
Attention:        Causal
Sq:               Sk
Warmup runs:      10
Measured runs:    50
```

The tested configurations vary:

* Batch size (`B`)
* Sequence length (`S`)
* Number of key/value heads (`Hkv`)
* GQA ratio

The query-head count remains fixed at `Hq = 32`.

## Implementations

### ROCKE

ROCKE Dense Attention is run from the ROCm Libraries tree containing this benchmark.

### Composable Kernel

The CK result uses the Composable Kernel FMHA implementation configured for the same BF16 causal-attention workload.

### AITER

The AITER result uses the optimized **FMHA v3 assembly kernel** for `gfx942`.

The setup script installs/clones the external dependencies needed by the benchmark so the benchmark can be launched from this directory.

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

For example, for:

```text
B=1
S=4096
Hq=32
Hkv=8
```

the measured latencies are:

```text
ROCKE:      0.649 ms
CK:         0.349 ms
AITER ASM:  0.286 ms
```

giving:

```text
CK vs ROCKE:      0.649 / 0.349 = 1.86x
AITER vs ROCKE:   0.649 / 0.286 = 2.27x
AITER vs CK:      0.349 / 0.286 = 1.22x
```

## Observations

Across the tested configurations, both CK and AITER outperform the current ROCKE Dense Attention implementation.

AITER FMHA v3 ASM gives the lowest measured latency for every supported configuration in this benchmark.

For the tested configurations:

```text
CK vs ROCKE:
1.37x – 1.86x

AITER vs ROCKE:
1.73x – 2.34x

AITER vs CK:
1.15x – 1.34x
```

The relative difference between ROCKE and the optimized implementations generally becomes smaller at larger batch sizes.

The `B=64, S=8192, Hq=32, Hkv=8` ROCKE configuration is marked as unsupported, while CK and AITER successfully execute the workload.

## Reproduction

From a fresh machine:

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

The setup script prepares the required ROCKE, CK, and AITER dependencies. The benchmark script then runs the same workload configurations across the available implementations and reports their measured execution times.

## Benchmark Files

```text
rocke_vs_aiter/
├── README.md
├── setup_mi300x_attention_bench.sh
└── run_mi300x_attention_bench_updated.sh
```

`setup_mi300x_attention_bench.sh` prepares the benchmark environment and dependencies.

`run_mi300x_attention_bench_updated.sh` executes the ROCKE, CK, and AITER attention benchmarks using the configurations described above.
