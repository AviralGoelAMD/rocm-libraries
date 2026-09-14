#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-side driver for the GDN single-token decode kernel.

The kernel emitter in ``kernels/gfx950/gdn_decode.py`` produces device code.
This module is everything on the host needed to actually exercise it: compile a
spec, build inputs, run it, judge the result against an independent reference,
and time it.

The reference is deliberately *not* a restatement of the kernel. It evaluates
the gated delta rule with whole-tensor fp32 arithmetic and knows nothing about
the kernel's workgroup mapping, warp tiling or cross-lane reductions, so
agreement is evidence about the algorithm rather than a tautology.

Both outputs are checked. The kernel writes ``out`` *and* mutates the recurrent
state in place; a driver that only compares ``out`` would let a corrupted state
write ship silently, because the state is not read back until the next decode
step.

Run::

    PYTHONPATH=<rocke>/library:<rocke>/platform/python \\
        python3 gdn_decode.py --batches 1,16 --variant tiled
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
import sys
from typing import Dict, Tuple

import torch

from kernels.gfx950.gdn_decode import (
    GdnDecodeSpec,
    build_gdn_decode,
    gdn_decode_grid,
    gdn_decode_signature,
    is_valid_spec,
)
from rocke.helpers.compile import compile_kernel
from rocke.runtime.launcher import KernelLauncher, LaunchConfig, no_fence

# bf16 inputs against an fp32 reference. Observed error grows with batch (more
# state rows accumulate into one output), so the bound is set well above the
# largest value seen at the batches exercised here while staying far below the
# magnitude of a real indexing or reduction bug.
TOL = 1e-2

_ARCH = "gfx950"
_LAUNCHER_CACHE: Dict[Tuple, KernelLauncher] = {}


def launcher_for(spec: GdnDecodeSpec, arch: str = _ARCH) -> KernelLauncher:
    """Compile ``spec`` and wrap it in a launcher, memoised per spec.

    Keyed on ``kernel_name()`` because that string is what the compiled code
    object is identified by; every field that changes emitted code is encoded
    in it, so two specs cannot collide on one cache entry.
    """
    key = (spec.kernel_name(), arch)
    cached = _LAUNCHER_CACHE.get(key)
    if cached is not None:
        return cached
    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid gdn_decode spec for {arch}: {why}")
    artifact = compile_kernel(build_gdn_decode(spec, arch=arch), arch=arch)
    launcher = KernelLauncher(
        hsaco=artifact.hsaco,
        kernel_name=artifact.kernel_name,
        signature=gdn_decode_signature(spec),
    )
    _LAUNCHER_CACHE[key] = launcher
    return launcher


def _gate_input(spec: GdnDecodeSpec, rnd, batch: int, hv: int, dk: int):
    """The `a` tensor, whose meaning depends on the gate mode.

    * GDN: one raw logit per head, `[B, 1, HV]`.
    * KDA, fuse_gate=True: one raw logit per K channel, `[B, 1, HV, DK]`; the
      kernel turns it into a decay.
    * KDA, fuse_gate=False: the natural-log-domain decay itself. Forced
      non-positive so `exp(a)` lands in (0, 1] -- a positive value would
      amplify the state every step rather than fade it, which the fused path
      cannot produce and so is not worth generating.

    Draws exactly one tensor in every mode, so callers can keep this at a fixed
    position in the input dict and leave the random stream undisturbed.
    """
    if spec.gate_kind != "kda":
        return rnd(batch, 1, hv)
    g = rnd(batch, 1, hv, dk, scale=0.5)
    return g if spec.fuse_gate else -g.abs()


def make_inputs(spec: GdnDecodeSpec, batch: int, seed: int = 0, device: str = "cuda"):
    """Deterministic inputs matching the kernel's packed decode contract."""
    torch_dtype = {"bf16": torch.bfloat16, "f16": torch.float16}[spec.dtype]
    state_dtype = {"bf16": torch.bfloat16, "f16": torch.float16}[spec.state_dtype]
    gen = torch.Generator(device=device).manual_seed(seed)

    def rnd(*shape, dtype=torch_dtype, scale=1.0):
        t = torch.randn(*shape, device=device, generator=gen, dtype=torch.float32)
        return (t * scale).to(dtype)

    hk, hv = spec.num_k_heads, spec.num_v_heads
    dk, dv = spec.head_k_dim, spec.head_v_dim
    return {
        "query": rnd(batch, 1, hk, dk),
        "key": rnd(batch, 1, hk, dk),
        "value": rnd(batch, 1, hv, dv),
        # `a` stays at this position in the dict so the GDN draw order -- and
        # therefore every seeded GDN input ever recorded -- is bit-for-bit
        # unchanged. See _gate_input for what it holds in each mode.
        "a": _gate_input(spec, rnd, batch, hv, dk),
        "b": rnd(batch, 1, hv),
        # f32 for KDA: the gate is evaluated in f32 and KDA prefill already
        # declares this tensor f32, so the two families share one contract.
        "dt_bias": (
            torch.randn(hv, dk, device=device, generator=gen, dtype=torch.float32) * 0.1
            if spec.gate_kind == "kda"
            else rnd(hv)
        ),
        "A_log": torch.randn(hv, device=device, generator=gen, dtype=torch.float32),
        "read_indices": torch.arange(batch, device=device, dtype=torch.int32),
        "write_indices": torch.arange(batch, device=device, dtype=torch.int32),
        # Pool is `batch` deep here; the kernel indexes it through read/write
        # indices, so a permuted or sparser pool exercises the same path.
        "state": rnd(batch, hv, dv, dk, dtype=state_dtype, scale=0.01),
    }


def ref_fp32(spec: GdnDecodeSpec, inp) -> Tuple[torch.Tensor, torch.Tensor]:
    """Whole-tensor fp32 reference for one decode step.

    Returns ``(out, state_after)``. Written directly from the gated delta rule
    with no chunking, tiling or cross-lane structure, so it shares no algebra
    with the kernel beyond the definition itself.
    """
    hv, g = spec.num_v_heads, spec.v_per_k_head
    scale = 1.0 / math.sqrt(spec.head_k_dim)
    eps = 1e-6

    # Each value head reads the key head it is grouped under.
    k_of_v = torch.arange(hv, device=inp["query"].device) // g
    q = inp["query"][:, 0].float()[:, k_of_v]  # [B, HV, DK]
    k = inp["key"][:, 0].float()[:, k_of_v]

    if spec.use_qk_l2norm:
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + eps) * scale
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + eps)
    else:
        # l2norm off: the kernel scales q by 1/sqrt(dk) and leaves k raw.
        q = q * scale

    if spec.gate_kind == "kda":
        if spec.fuse_gate:
            # Per-channel gate, bounded in (lower_bound, 0) before the exp.
            #   log_decay[d] = lower_bound * sigmoid(exp(A_log[h]) * (g[d] + dt_bias[h,d]))
            inner = torch.exp(inp["A_log"].float())[None, :, None] * (
                inp["a"][:, 0].float() + inp["dt_bias"].float()
            )
            decay = torch.exp(spec.lower_bound * torch.sigmoid(inner))  # [B, HV, DK]
        else:
            # `a` already carries natural-log-domain decay; only the exp remains.
            # A_log and dt_bias are unread here -- the caller folded them in.
            # This mirrors the competitor recurrence-only kernel, which likewise
            # applies exp(gk) in-kernel and nothing else, which is what makes the
            # two timeable at one work boundary.
            decay = torch.exp(inp["a"][:, 0].float())  # [B, HV, DK]
    else:
        # Scalar gate, unbounded below.
        #   log_decay = -exp(A_log[h]) * softplus(a + dt_bias[h])
        x = inp["a"][:, 0].float() + inp["dt_bias"].float()  # [B, HV]
        softplus = torch.where(x > 20.0, x, torch.log1p(torch.exp(x)))
        decay = torch.exp(-torch.exp(inp["A_log"].float()) * softplus)  # [B, HV]
    beta = torch.sigmoid(inp["b"][:, 0].float())

    state = inp["state"].float()[inp["read_indices"].long()]  # [B, HV, DV, DK]
    # The decay multiplies S from the right, so a per-channel gate scales
    # columns (the DK axis) and a scalar gate scales the whole matrix.
    s = state * (
        decay[..., None, :] if spec.gate_kind == "kda" else decay[..., None, None]
    )

    sk = (s @ k[..., None]).squeeze(-1)  # [B, HV, DV]
    sq = (s @ q[..., None]).squeeze(-1)
    v_new = (inp["value"][:, 0].float() - sk) * beta[..., None]
    kq = (k * q).sum(-1)  # [B, HV]

    out = sq + v_new * kq[..., None]
    s_after = s + v_new[..., None] * k[..., None, :]
    return out.unsqueeze(1), s_after


def _validate_decode_inputs(
    spec: GdnDecodeSpec, inp, batch: int, *, validate_indices: bool = True
):
    """Reject inputs that would make the decode kernel read or write out of bounds.

    The kernel indexes the state pool through ``read_indices``/``write_indices``
    with a raw pointer add and only skips the ``-1`` sentinel; it never
    hardware-bounds the index against the pool depth, so an out-of-range index
    is an out-of-bounds load/store. These host checks catch that before launch.

    Shape and dtype checks are sync-free and always run. The index *value* range
    check reads the index extrema, forcing a device->host sync, so it is gated
    by ``validate_indices`` (default on; a hot re-prepare loop whose indices are
    already known good may pass ``False``).
    """
    if spec.gate_kind == "kda":
        # The emitter indexes these buffers from compile-time B/HV/DK extents:
        # unlike the scalar GDN ABI, a legacy-sized allocation is too short and
        # the vector loads would read beyond it. Metadata checks are sync-free,
        # so they always run even when index-value validation is disabled.
        a = inp["a"]
        want_a = (batch, 1, spec.num_v_heads, spec.head_k_dim)
        if tuple(a.shape) != want_a:
            raise ValueError(f"KDA a shape {tuple(a.shape)} != {want_a}")
        want_a_dtype = {"bf16": torch.bfloat16, "f16": torch.float16}[spec.dtype]
        if a.dtype != want_a_dtype:
            raise ValueError(f"KDA a dtype {a.dtype} != {want_a_dtype}")
        if not a.is_contiguous():
            raise ValueError("KDA a must be contiguous")

        dt_bias = inp["dt_bias"]
        want_dt_bias = (spec.num_v_heads, spec.head_k_dim)
        if tuple(dt_bias.shape) != want_dt_bias:
            raise ValueError(
                f"KDA dt_bias shape {tuple(dt_bias.shape)} != {want_dt_bias}"
            )
        if dt_bias.dtype != torch.float32:
            raise ValueError(f"KDA dt_bias dtype {dt_bias.dtype} != torch.float32")
        if not dt_bias.is_contiguous():
            raise ValueError("KDA dt_bias must be contiguous")

        device = inp["query"].device
        for name, tensor in (("a", a), ("dt_bias", dt_bias)):
            if tensor.device != device:
                raise ValueError(
                    f"KDA {name} device {tensor.device} != query device {device}"
                )

    state = inp["state"]
    if state.ndim != 4:
        raise ValueError(f"state must be [pool, HV, DV, DK]; got {tuple(state.shape)}")
    pool_depth = state.shape[0]
    want = (spec.num_v_heads, spec.head_v_dim, spec.head_k_dim)
    if tuple(state.shape[1:]) != want:
        raise ValueError(f"state head dims {tuple(state.shape[1:])} != spec {want}")
    for name in ("read_indices", "write_indices"):
        idx = inp[name]
        if idx.dtype != torch.int32:
            raise ValueError(f"{name} must be int32; got {idx.dtype}")
        if tuple(idx.shape) != (batch,):
            raise ValueError(f"{name} must be [batch={batch}]; got {tuple(idx.shape)}")
    if not validate_indices:
        return
    for name in ("read_indices", "write_indices"):
        idx = inp[name]
        # -1 is the 'skip this slot' sentinel; every other value must land in
        # [0, pool_depth). min()/max() forces one device->host sync.
        lo = int(idx.min().item())
        hi = int(idx.max().item())
        if lo < -1 or hi >= pool_depth:
            raise ValueError(
                f"{name} out of range: [{lo}, {hi}] escapes -1 (skip) or "
                f"[0, {pool_depth})"
            )


def prepare(spec: GdnDecodeSpec, inp, batch: int, *, validate_indices: bool = True):
    """Allocate the kernel's outputs and freeze a launch config.

    Split out from :func:`launch` deliberately. Allocating inside a timing loop
    measures the allocator rather than the kernel, and allocating inside a HIP
    graph capture is illegal, so every caller that repeats a launch prepares
    once and then only launches.
    """
    _validate_decode_inputs(spec, inp, batch, validate_indices=validate_indices)
    torch_dtype = {"bf16": torch.bfloat16, "f16": torch.float16}[spec.dtype]
    out = torch.zeros(
        batch,
        1,
        spec.num_v_heads,
        spec.head_v_dim,
        device=inp["query"].device,
        dtype=torch_dtype,
    )
    values = dict(inp)
    values["state"] = inp["state"].clone()  # the kernel updates the state in place
    values["out"] = out
    values["batch_size"] = batch
    cfg = LaunchConfig(
        grid=gdn_decode_grid(batch, spec),
        block=(spec.block_size, 1, 1),
        stream=0,
    )
    return values, cfg


def launch(launcher: KernelLauncher, values, cfg) -> None:
    """Enqueue one kernel launch. No allocation, no synchronisation."""
    with no_fence():
        launcher(values, config=cfg)


def run(spec: GdnDecodeSpec, inp, launcher: KernelLauncher, batch: int):
    """Prepare, launch once, synchronise. Returns ``(out, state_after)``."""
    values, cfg = prepare(spec, inp, batch)
    launch(launcher, values, cfg)
    torch.cuda.synchronize()
    return values["out"], values["state"]


def check(spec: GdnDecodeSpec, batch: int, seed: int = 0) -> Tuple[float, float]:
    """Run and compare against the reference. Returns ``(out_err, state_err)``."""
    inp = make_inputs(spec, batch, seed=seed)
    ref_out, ref_state = ref_fp32(spec, inp)
    out, state = run(spec, inp, launcher_for(spec), batch)
    out_err = (out.float() - ref_out).abs().max().item()
    # Compare only the pages the kernel was told to write.
    written = inp["write_indices"].long()
    state_err = (state.float()[written] - ref_state).abs().max().item()
    return out_err, state_err


def bench(spec: GdnDecodeSpec, batch: int, reps: int = 200) -> float:
    """Median host-observed launch latency in microseconds."""
    launcher = launcher_for(spec)
    values, cfg = prepare(spec, make_inputs(spec, batch), batch)
    for _ in range(50):
        launch(launcher, values, cfg)
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start = time.perf_counter_ns()
        launch(launcher, values, cfg)
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1e3)
    return statistics.median(samples)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batches", default="1,16,64", help="comma-separated batch sizes")
    ap.add_argument(
        "--variant",
        choices=("tiled", "simple"),
        default="tiled",
        help="tiled = default warp-tiled path; simple = one-thread-per-row reference",
    )
    ap.add_argument("--bench", action="store_true", help="also report per-launch time")
    ap.add_argument("--no-check", action="store_true", help="skip the correctness gate")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no HIP device visible", file=sys.stderr)
        return 2

    spec = GdnDecodeSpec(simple=(args.variant == "simple"))
    ok, why = is_valid_spec(spec, arch=_ARCH)
    if not ok:
        print(f"spec rejected: {why}", file=sys.stderr)
        return 2
    print(f"kernel: {spec.kernel_name()}  block={spec.block_size}")

    worst = 0.0
    for batch in (int(x) for x in args.batches.split(",")):
        grid = gdn_decode_grid(batch, spec)
        line = f"B={batch:<5d} grid={grid[0]:<7d}"
        if not args.no_check:
            out_err, state_err = check(spec, batch, seed=args.seed)
            worst = max(worst, out_err, state_err)
            verdict = "OK" if max(out_err, state_err) <= TOL else "FAIL"
            line += f" out_err={out_err:.3e} state_err={state_err:.3e} {verdict}"
        if args.bench:
            line += f" {bench(spec, batch):8.2f}us"
        print(line)

    if args.no_check:
        return 0
    print(f"worst={worst:.3e} tol={TOL:.1e}")
    return 0 if worst <= TOL else 1


if __name__ == "__main__":
    raise SystemExit(main())
