# GDN/KDA Decode Instance

GDN and KDA are linear-attention operators. Where softmax attention re-reads
every past token, they carry a fixed-size recurrent state per value head — a
`head_v_dim × head_k_dim` matrix that compresses everything seen so far. Each
new token reads and updates that matrix, so per-token work does not grow with
sequence length.

They share one decode emitter. The difference is the forget gate: GDN uses one
scalar decay per head; KDA uses one decay per K channel.

This instance covers **decode** only: one token per sequence, for a batch of
sequences generated concurrently. Prefill uses the chunkwise KDA emitter.

Kernel and drivers live under `library/` (`library -> platform` one-way):

- `library/kernels/gfx950/gdn_decode.py` — shared spec, validator and emitter
- `library/builders/gfx950/gdn/gdn_decode.py` — host driver and independent fp32 reference
- `library/builders/gfx950/gdn/tune.py` — exhaustive tile sweep
- `library/benchmarks/gfx950/gdn/benchmark_gdn_decode.py` — GDN production benchmark
- `library/benchmarks/gfx950/gdn/benchmark_kda_decode.py` — KDA production/variant benchmark
- `library/dispatch/gdn/` — request, candidates and `dispatch_gdn_decode`

For equations and GPU mapping, see
[`library/builders/gfx950/gdn/ALGORITHM.md`](../../../library/builders/gfx950/gdn/ALGORITHM.md).
For commands and output interpretation, see
[`library/builders/gfx950/gdn/README.md`](../../../library/builders/gfx950/gdn/README.md).

## Contents

- [The decode step](#the-decode-step)
- [Tensor contract](#tensor-contract)
- [Spec and validation](#spec-and-validation)
- [Thread mapping](#thread-mapping)
- [Tuned tile selection](#tuned-tile-selection)
- [Dispatch](#dispatch)
- [Coverage](#coverage)
- [Failure modes](#failure-modes)

## The decode step

For each active sequence and value head, one gated-delta-rule step:

```text
q_hat = l2norm(q) * head_k_dim**-0.5
k_hat = l2norm(k)

GDN: log_decay[h]   = -exp(A_log[h]) * softplus(a[h] + dt_bias[h])
KDA: log_decay[h,d] = lower_bound * sigmoid(exp(A_log[h]) * (a[h,d] + dt_bias[h,d]))
decay               = exp(log_decay)

beta  = sigmoid(b)
S     = S * decay
v_new = (v - S @ k_hat) * beta
out   = S @ q_hat + v_new * dot(k_hat, q_hat)
S     = S + outer(v_new, k_hat)
```

The `v_new` term stores only the correction to what the state already predicts.
KDA's vector decay multiplies the state columns (`DK`); GDN's scalar decay
multiplies the whole matrix.

## Tensor contract

All tensors are contiguous row-major. `gate_kind` changes only the gate-input
contract:

| Tensor | GDN shape/type | KDA shape/type |
|---|---|---|
| `query`, `key` | `[B, 1, num_k_heads, head_k_dim]`, `dtype` | same |
| `value`, `out` | `[B, 1, num_v_heads, head_v_dim]`, `dtype` | same |
| `a` | `[B, 1, num_v_heads]`, `dtype` | `[B, 1, num_v_heads, head_k_dim]`, `dtype` |
| `b` | `[B, 1, num_v_heads]`, `dtype` | same |
| `dt_bias` | `[num_v_heads]`, `dtype` | `[num_v_heads, head_k_dim]`, `f32` |
| `A_log` | `[num_v_heads]`, `f32` | same |
| `read_indices`, `write_indices` | `[B]`, `i32` | same |
| `state` | `[pool, num_v_heads, head_v_dim, head_k_dim]`, `state_dtype` | same |

The `1` is the sequence length. State lives in a pool addressed by
`read_indices` / `write_indices`; a negative index skips the sequence. The
kernel produces `out` and updates `state` in place.

## Spec and validation

`GdnDecodeSpec.gate_kind` selects `gdn` or `kda`. `fuse_gate=True` is the
dispatched production mode; `fuse_gate=False` accepts precomputed natural-log
decay and is a benchmark-only recurrence mode. `lower_bound` controls the
fused KDA sigmoid gate.

The remaining spec fields carry head geometry, dtypes, and three tiling knobs:
`num_warps`, `warp_threads_k`, and `blocks_per_v_dim`. `simple=True` selects a
one-thread-per-state-row diagnostic body.

`is_valid_spec()` rejects unbuildable geometry before IR construction. The
host `prepare()` validates the state pool and index values, plus the
gate-kind-dependent KDA buffers:

```text
a        [B,1,HV,DK]  dtype, contiguous
dt_bias  [HV,DK]      f32, contiguous
```

Those gate buffers must share `query`'s device. The checks run before launch;
shape/dtype/device/stride checks are sync-free, while index-value validation
is default-on and can be disabled for a known-good hot prepare path.

`kernel_name()` is the compile/launcher cache key. Every field that changes
emitted code participates; default GDN fields add no suffix so existing names
remain stable.

## Thread mapping

One workgroup per `(sequence, value head, v-sub-block)`; the grid is
`batch * num_v_heads * blocks_per_v_dim`.

Within a workgroup, each warp's lanes are split `warp_threads_k` ways across the
key dimension and `wave_size / warp_threads_k` ways across value rows. A lane
holds a contiguous run of K elements, so the dot products against `k_hat` and
`q_hat` become lane-local products followed by a cross-lane sum.

That sum is an xor butterfly. Offsets inside a four-lane quad use `quad_perm` on
the VALU, avoiding the LDS crossbar and its wait; wider offsets fall back to
`ds_swizzle`. Every lane ends holding the total, so no broadcast is needed, and
only the first lane of each group stores the output scalar.

## Tuned tile selection

`blocks_per_v_dim` splits one head's value dimension across workgroups to
manufacture parallelism when the natural grid is small.

The gate kinds use separate tables because KDA's per-channel gate has a
different load/register profile:

- **GDN:** original batch-keyed table. Existing routing remains unchanged.
- **KDA:** keyed by `work = batch × num_v_heads`, so tensor-parallel head
  sharding maps to the same key as an equivalent amount of batch work.

Both tables are generated from exhaustive legal-tile sweeps with every
configuration checked against the fp32 reference before timing. Re-measure with
`library/builders/gfx950/gdn/tune.py`; exact measurements live outside the
public source tree.

## Dispatch

`dispatch_gdn_decode(GdnDecodeRequest(...))` returns the selected candidate,
spec, signature, grid, and block. Set `gate_kind=\"kda\"` for per-channel
decode. Selection is:

```text
capability → request/support checks → gate-specific tuned table → spec
```

An explicit `spec_id` forces a registered tile of the same gate kind.

## Coverage

gfx950 only. `bf16` and `f16` activation/state dtypes are supported and need not
match. Head geometry is constrained by the validator rather than a fixed list.
KDA's production fused mode and benchmark-only precomputed-log-decay mode are
both numerically covered.

## Failure modes

- **Spec rejected at dispatch.** The message names the failing geometry or
  gate-kind rule.
- **Wrong arch.** Candidates declare gfx950; another arch is rejected.
- **Malformed KDA gate buffers.** `prepare()` requires per-channel `a` and f32
  `dt_bias` with exact contiguous shapes and the same device as `query`.
- **State appears corrupted on the following step.** The kernel writes `out`
  and mutates `state`; numeric tests compare both.
- **Padding lane touched.** A negative read/write index must leave its state
  slot bit-identical.
