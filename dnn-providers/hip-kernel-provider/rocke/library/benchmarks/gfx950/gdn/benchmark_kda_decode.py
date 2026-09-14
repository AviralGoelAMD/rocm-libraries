#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Three-arm benchmark for the gfx950 KDA single-token decode kernel.

No single comparison against the reference implementation is both fair and
production-shaped, because it has no kernel at our shipping kernel's work
boundary:

* ``fused_recurrent_gated_delta_rule`` (with ``gk``) takes a PRECOMPUTED
  per-channel decay and applies ``exp(gk)`` in-kernel -- the same boundary as
  our ``fuse_gate=False`` mode.
* ``fused_sigmoid_gating_delta_rule_update`` computes a gate in-kernel but a
  SCALAR one (the softplus/GDN formula) -- a different operator.
* ``fused_kda_decode`` has the KDA gate but also fuses conv1d and gated
  RMSNorm -- strictly more work than the recurrence.

So we measure around the gap rather than arguing past it, and label every arm
with what it does and does not compare:

``arm1``  rocKE ``fuse_gate=False`` vs the reference recurrence-only kernel.
          Both consume a precomputed per-channel decay. This is the only
          identical-boundary comparison available and is the head-to-head
          number.
``arm2``  rocKE ``fuse_gate=True`` vs rocKE ``fuse_gate=False``. Same kernel,
          same shapes, so the difference isolates the cost of computing the
          gate INSIDE the kernel. Read it carefully: the raw arm is handed its
          decay for free -- producing that decay is not counted anywhere in
          this arm -- so a ratio slightly above 1 does not mean fusing is a net
          loss. It means the in-kernel gate is cheap. The real alternative to
          fusing has to produce the decay somewhere, which costs a separate
          launch at a shape where decode is launch-bound. arm2 on the host
          clock below prices that, and it is the clock the decision rests on.
``arm3``  rocKE ``fuse_gate=True`` vs the fused conv1d+recurrence+RMSNorm
          entry point. NOT a kernel comparison; reported only with the extra
          stages named.

Known asymmetry in arm1, stated rather than hidden: our kernel also applies
``sigmoid`` to the write gate ``beta``, which the reference receives already
activated. That is one reciprocal per head against DK-wide work per channel,
so it is small -- but it is not zero, and it favours the reference.

Every arm is correctness-gated in the same run against the fp32 reference.
An arm that cannot be gated is reported as such and not timed: a timing number
from a numerically wrong kernel is worse than no number at all.

Run::

    PYTHONPATH=<rocke>/library:<rocke>/platform/python \\
        python3 benchmark_kda_decode.py --batches 1,8,32,128
"""

from __future__ import annotations

import argparse
import dataclasses as dc
import sys

import torch

from builders.gfx950.gdn.gdn_decode import (
    TOL,
    launch,
    launcher_for,
    make_inputs,
    prepare,
    ref_fp32,
)
from kernels.gfx950.gdn_decode import GdnDecodeSpec

ARCH = "gfx950"


def device_us(fn, *, warmup: int = 10, reps: int = 32) -> float | None:
    """Per-launch device time from a replayed graph, or None if capture fails.

    Decode kernels are small enough that host submission dominates a naive
    timing loop, so the work is chained inside one graph and the per-launch
    cost divided out.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(reps):
                fn()
    except Exception as exc:  # pragma: no cover - diagnostic
        print(f"    graph capture failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        torch.cuda.synchronize()
        return None
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(20):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) * 1e3 / reps)
    return best


def eager_us(fn, *, warmup: int = 20, reps: int = 100) -> float:
    """Host-observed per-call latency: one launch per synchronisation.

    This is what a decode loop dispatching from Python actually pays, and it is
    the clock arm2 has to be read on. Device time prices only the work inside
    the kernel; the alternative to fusing is a SECOND LAUNCH, whose cost is
    entirely host-side and therefore invisible to a graph-replay measurement.
    """
    import time

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start = time.perf_counter_ns()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1e3)
    samples.sort()
    return samples[len(samples) // 2]


def _log_decay_from(spec: GdnDecodeSpec, inp) -> torch.Tensor:
    """The natural-log-domain decay the fused gate produces, as [B, HV, DK]."""
    gx = inp["a"][:, 0].float() + inp["dt_bias"].float()
    inner = torch.exp(inp["A_log"].float())[None, :, None] * gx
    return spec.lower_bound * torch.sigmoid(inner)


def _rocke_arm(spec: GdnDecodeSpec, inp, batch: int):
    """(callable, out, state, err) for one rocKE configuration."""
    launcher = launcher_for(spec, arch=ARCH)
    values, cfg = prepare(spec, inp, batch)
    launch(launcher, values, cfg)
    torch.cuda.synchronize()

    ref_out, ref_state = ref_fp32(spec, inp)
    written = inp["write_indices"].long()
    err = max(
        (values["out"].float() - ref_out).abs().max().item(),
        (values["state"].float()[written] - ref_state).abs().max().item(),
    )
    return (lambda: launch(launcher, values, cfg)), values["out"], values["state"], err


def _reference_recurrent_arm(spec: GdnDecodeSpec, inp, batch: int):
    """(callable, err) for the recurrence-only kernel at our raw-gate boundary."""
    try:
        from aiter.ops.triton.gated_delta_net.gated_delta_rule import (
            fused_recurrent_gated_delta_rule,
        )
    except Exception as exc:
        print(f"    reference kernel unavailable: {exc}", file=sys.stderr)
        return None, None

    hv, g = spec.num_v_heads, spec.v_per_k_head
    # Our q/k are packed per key head; the reference expects one head axis and
    # applies GVA itself when HV > H, so they are passed as-is.
    q = inp["query"]
    k = inp["key"]
    v = inp["value"]
    gk = inp["a"].float()  # [B, 1, HV, DK], already natural-log-domain
    beta = torch.sigmoid(inp["b"].float())  # ours activates this in-kernel
    # Our state is [pool, HV, DV, DK]; the reference wants [N, HV, K, V].
    h0 = inp["state"].float()[inp["read_indices"].long()].transpose(-1, -2).contiguous()

    def run():
        return fused_recurrent_gated_delta_rule(
            q,
            k,
            v,
            gk=gk,
            beta=beta,
            initial_state=h0,
            output_final_state=True,
            use_qk_l2norm_in_kernel=bool(spec.use_qk_l2norm),
        )

    try:
        out, final = run()
    except Exception as exc:
        print(
            f"    reference kernel failed: {type(exc).__name__}: {exc}", file=sys.stderr
        )
        return None, None
    torch.cuda.synchronize()

    ref_out, ref_state = ref_fp32(spec, inp)
    err = (out.float() - ref_out).abs().max().item()
    if final is not None:
        err = max(err, (final.float().transpose(-1, -2) - ref_state).abs().max().item())
    return run, err


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batches", default="1,8,32,128")
    ap.add_argument("--num-k-heads", type=int, default=32)
    ap.add_argument("--num-v-heads", type=int, default=32)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--arms", default="arm1,arm2")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no HIP device visible", file=sys.stderr)
        return 2

    arms = set(a.strip() for a in args.arms.split(","))
    fused = dc.replace(
        GdnDecodeSpec(),
        gate_kind="kda",
        num_k_heads=args.num_k_heads,
        num_v_heads=args.num_v_heads,
        head_k_dim=args.head_dim,
        head_v_dim=args.head_dim,
    )
    raw = dc.replace(fused, fuse_gate=False)

    print(
        f"shape Hk={args.num_k_heads} Hv={args.num_v_heads} D={args.head_dim} "
        f"bf16, one token"
    )
    print(
        f"every arm correctness-gated in-run against the fp32 reference (tol {TOL})\n"
    )
    print(
        f"{'batch':>6} {'work':>7} {'rocke_fused':>12} {'rocke_raw':>10} "
        f"{'ref_recur':>10} {'arm1':>7} {'arm2':>7}"
    )

    for batch in (int(x) for x in args.batches.split(",")):
        work = batch * args.num_v_heads

        # rocKE fused: inputs are raw logits.
        inp_f = make_inputs(fused, batch)
        call_f, _, _, err_f = _rocke_arm(fused, inp_f, batch)
        if err_f > TOL:
            print(f"{batch:>6} {work:>7}   FAILED GATE rocke_fused err={err_f:.2e}")
            continue

        # rocKE raw + the reference share one input set: the log-decay the
        # fused gate would have produced, so all three do identical work on
        # identical numbers.
        inp_r = dict(inp_f)
        inp_r["a"] = _log_decay_from(fused, inp_f)[:, None].to(inp_f["a"].dtype)
        inp_r["state"] = inp_f["state"].clone()
        call_r, _, _, err_r = _rocke_arm(raw, inp_r, batch)
        if err_r > TOL:
            print(f"{batch:>6} {work:>7}   FAILED GATE rocke_raw err={err_r:.2e}")
            continue

        us_f = device_us(call_f)
        us_r = device_us(call_r)

        # arm2 on the host clock: fused is ONE launch; unfused is the gate pass
        # plus the kernel, i.e. two. That second launch is the cost fusing
        # exists to avoid and the only clock that can see it.
        host_f = eager_us(call_f)
        spec_l, inp_l = fused, inp_f

        def unfused_pair():
            gx = inp_l["a"][:, 0].float() + inp_l["dt_bias"].float()
            inner = torch.exp(inp_l["A_log"].float())[None, :, None] * gx
            _ = (spec_l.lower_bound * torch.sigmoid(inner)).to(inp_l["a"].dtype)
            call_r()

        host_pair = eager_us(unfused_pair)

        us_ref = None
        if "arm1" in arms:
            inp_ref = dict(inp_r)
            inp_ref["state"] = inp_f["state"].clone()
            call_ref, err_ref = _reference_recurrent_arm(raw, inp_ref, batch)
            if call_ref is None:
                us_ref = None
            elif err_ref is not None and err_ref > TOL:
                print(
                    f"{batch:>6} {work:>7}   reference FAILED GATE err={err_ref:.2e}"
                    " -- not timed",
                    file=sys.stderr,
                )
            else:
                us_ref = device_us(call_ref)

        a1 = f"{us_r / us_ref:.3f}" if (us_ref and us_r) else "-"
        a2h = f"{host_f / host_pair:.3f}"
        a2 = f"{us_f / us_r:.3f}" if (us_f and us_r) else "-"
        print(
            f"{batch:>6} {work:>7} {us_f:12.3f} {us_r:10.3f} "
            f"{(us_ref if us_ref else float('nan')):10.3f} {a1:>7} {a2:>7}"
            f" | host {host_f:8.2f} vs {host_pair:8.2f} = {a2h:>6}"
        )

    print(
        "\narm1 = rocke_raw / ref_recur: identical work boundary, both take a\n"
        "       precomputed per-channel decay. This is the head-to-head number.\n"
        "       Caveat: rocKE additionally sigmoids beta; the reference does not.\n"
        "arm2 = rocke_fused / rocke_raw: cost of computing the gate in-kernel,\n"
        "       with the raw arm handed its decay for free. A ratio near 1 means\n"
        "       the in-kernel gate is cheap, NOT that fusing is a net loss --\n"
        "       the unfused path still has to produce that decay somewhere.\n"
        "       The host columns price that: one launch versus two. Decode is\n"
        "       launch-bound, so the host clock is the one the default rests on.\n"
        "arm3 (not run here) would compare against the conv1d+recurrence+RMSNorm\n"
        "       entry point, which does strictly more work -- never a bare ratio."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
