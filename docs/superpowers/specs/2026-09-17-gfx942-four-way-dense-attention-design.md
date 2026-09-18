# gfx942 Four-Way Dense Attention Benchmark Design

**Status:** User-approved

## Goal

Extend `dnn-providers/hip-kernel-provider/rocke/library/benchmarks/gfx942/attention/prefill/rocke_vs_aiter/` so every existing `(B, S, Hkv)` row reports four dense causal BF16-D128 implementations on gfx942:

1. rocKE `attention_dense`, selected through `dense_request -> resolve_dense_spec`.
2. rocKE normal unified attention, selected through `run_unified_attention_torch(backend="auto")`.
3. AITER FMHA-v3 ASM, requiring `fmha_fwd_hd128_bf16_causal_rtz`.
4. CK Tile FMHA, requiring the existing fixed generated instance.

The benchmark answers whether rocKE `attention_dense` is faster than rocKE's normal unified route while preserving the external AITER and CK comparisons.

## Workload Contract

Every row is dense, contiguous, forward causal attention:

```text
BF16; D=128; Hq=32; Hkv in {8,16}; Sq=Sk; BSHD; GQA; no paging;
no bias; no dropout; no LSE output; softmax scale = 1/sqrt(128).
```

The ten existing rows remain unchanged. `B=64, S=8192, Hkv=8` remains a required coverage row: `attention_dense` must report its structured 32-bit extent rejection rather than fabricate a timing, while unified/AITER/CK continue when supported.

## rocKE Comparison Design

A dedicated Python runner receives each row once and creates deterministic BF16 `q`, `k`, and `v` tensors from one seed. It uses those exact tensors for both rocKE paths.

`attention_dense` receives contiguous `[B, S, H, D]` tensors directly. The unified path receives the same logical K/V through a one-time identity paging conversion outside every timed region:

- page size: 64;
- cache shape: `[B * (S / 64), 64, Hkv, D]`;
- block table: contiguous page IDs for each sequence;
- flattened query: `[B * S, Hq, D]`;
- `cu_seqlens_q`: `[0, S, 2S, ...]`;
- `seqused_k`: `S` for every sequence.

The runner uses `run_unified_attention_torch(..., backend="auto")`, so the benchmark measures rocKE's normal unified selector rather than a manually selected tiled variant. It records the resolved unified path and concrete kernel identity along with the dense kernel identity and dense dispatch settings.

Before timing either rocKE path, the runner compares its output to one FP32 causal GQA SDPA reference built from the same Q/K/V fixture. A row passes only when `max_abs < 4e-2`; otherwise it reports `FAIL` and is excluded from ratios. The dedicated dense unsupported row is `UNSUPPORTED`, not `PASS` or `FAIL`.

Both rocKE paths use rocKE's HIP-event `time_launches` helper with the existing 10 warmups and 50 measured launches. Fixture allocation, dense-to-paged conversion, compilation, and reference computation are all outside timed regions.

## External Arms

The existing AITER and CK invocations retain their pinned selected kernels, shapes, and one-split policy. Their timed commands explicitly request GPU timing. Each row first runs the native GPU validation mode outside timing; a failed validation makes that arm `FAIL` and excludes its ratios. These tools cannot accept the in-process torch fixture without patching pinned third-party source, so the report states that exact shared fixtures apply to the two rocKE arms and that AITER/CK receive equivalent workload parameters plus their native correctness gates.

## Reporting

The result directory contains separate TSV files for `rocke_dense`, `rocke_unified`, `aiter`, and `ck`, plus raw logs for every arm. `results.csv` and `benchmark_results.md` contain:

- shape and semantic fields;
- status, reason, latency, and TFLOP/s for every arm;
- dense and unified kernel/path identities;
- rocKE max-absolute errors;
- the fastest supported correct rocKE arm;
- `unified / dense`, `AITer / best-rocKE`, and `CK / best-rocKE` latency ratios;
- an explicit unsupported/error section.

No result may call an arm `PASS` solely because it launched or because a support check succeeded.

## Tests

A CPU test covers the fixed ten-row matrix, identity-paged KV construction, output-shape metadata, and the required dense unsupported status for the final row. A gfx942 GPU-marked test exercises one BF16 GQA fixture through dense and auto-unified paths using the same tensors and asserts both FP32-reference errors meet `4e-2`. The normal benchmark run is the acceptance test for all ten rows.

## Non-goals

- No paged-KV benchmark or comparison with the workspace's paged std-QK kernel.
- No manual rocKE knob sweep, profiler integration, or kernel optimization in this change.
- No patch to AITER or CK sources to load a shared serialized fixture.
- No change to rocKE dispatch policy; the benchmark measures the two existing routes.
