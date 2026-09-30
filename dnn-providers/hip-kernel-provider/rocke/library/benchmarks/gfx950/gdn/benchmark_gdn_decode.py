#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Benchmark registered GDN single-token decode candidates.

Candidate enumeration comes only from the dispatch registry. Production
``auto`` is reported separately and remains a static policy. Kernels compile
for the visible device's arch (gfx942 or gfx950). Device time is cold-cache by
default (``--rotate-mb``); the host-observed eager latency is warm.
"""

from __future__ import annotations

import argparse
import statistics
import time
import sys

from dispatch.gdn import (
    GDN_DECODE_ARCHES,
    GdnDecodeRequest,
    dispatch_gdn_decode,
    dispatch_gdn_decode_all,
)
from kernels.common.gdn_decode import GdnDecodeSpec

DEFAULT_BATCHES = (1, 16, 64, 256)


def registered_results(req: GdnDecodeRequest):
    """Return the exact registry results consumed by candidate benchmarking."""
    return dispatch_gdn_decode_all(req)


def eager_us(spec: GdnDecodeSpec, batch: int, arch: str, reps: int = 200) -> float:
    """Median host-observed launch latency in microseconds.

    Inputs and the launch config are prepared once, outside the timed region.
    Each sample measures the CPU call plus the wait for that launch to finish,
    which is the latency a synchronous Python decode loop observes. It reuses
    one buffer set, so it is warm-cache by construction.
    """
    import torch
    from builders.gfx950.gdn.gdn_decode import (
        launch,
        launcher_for,
        make_inputs,
        prepare,
    )

    launcher = launcher_for(spec, arch=arch)
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


def device_us(spec: GdnDecodeSpec, batch: int, arch: str, rotate_bytes: int):
    """Per-launch device time from a replayed HIP graph, or None if unavailable.

    Cycles through enough input sets to touch ``rotate_bytes`` per cycle, so
    each launch reads HBM rather than a cache-resident copy (``0``: one reused
    set, warm cache). Buffers are allocated before capture because allocation
    during capture is illegal.
    """
    from builders.gfx950.gdn.gdn_decode import (
        graph_device_us,
        launcher_for,
        make_inputs,
        prepare,
        rotation_input_sets,
    )

    launcher = launcher_for(spec, arch=arch)
    inp = make_inputs(spec, batch)
    prepared = [
        prepare(spec, s, batch) for s in rotation_input_sets(inp, batch, rotate_bytes)
    ]
    micros = graph_device_us(launcher, prepared, reps=64)
    if micros is None:
        print("    graph capture unavailable", file=sys.stderr)
    return micros


def resolve_target(requested: str | None):
    """``(arch, device description)`` for the visible device, or ``None`` after
    printing why it cannot run: no device, an arch without GDN decode, or an
    ``--arch`` that names a different arch than the device."""
    import torch

    if not torch.cuda.is_available():
        print("no HIP device visible", file=sys.stderr)
        return None
    from builders.gfx950.gdn.gdn_decode import device_arch

    arch = device_arch()
    if arch not in GDN_DECODE_ARCHES:
        print(
            f"GDN decode supports {GDN_DECODE_ARCHES}; device is {arch}",
            file=sys.stderr,
        )
        return None
    if requested is not None and requested != arch:
        print(f"--arch {requested} does not match the device ({arch})", file=sys.stderr)
        return None
    return arch, f"{torch.cuda.get_device_name(0)} torch={torch.__version__}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--batches",
        default=",".join(str(b) for b in DEFAULT_BATCHES),
        help="comma-separated decode batch sizes",
    )
    ap.add_argument("--no-device", action="store_true", help="skip HIP-graph timing")
    ap.add_argument(
        "--arch",
        default=None,
        help="target arch (default: the visible device); must match the device",
    )
    ap.add_argument(
        "--rotate-mb",
        type=int,
        default=1024,
        help="cold memory (default): rotate input copies so one graph cycle "
        "touches >= N MB (4x the MI300X 256 MB Infinity Cache); 0 = warm cache, "
        "debugging only",
    )
    args = ap.parse_args()

    target = resolve_target(args.arch)
    if target is None:
        return 2
    arch, device = target

    cache = f"cold (>= {args.rotate_mb} MB per cycle)" if args.rotate_mb else "warm"
    print(f"# arch={arch} device={device} device_us={cache} eager_us=warm")
    print(
        f"{'batch':>6} {'arm':>5} {'tile':>10} {'spec_id':>22} {'grid':>8} "
        f"{'eager_us':>10} {'device_us':>10}  correctness"
    )

    failures = 0
    for batch in (int(x) for x in args.batches.split(",")):
        request = GdnDecodeRequest(batch=batch, arch=arch)
        results = registered_results(request)
        if len(results) != 54:
            print(
                f"batch {batch}: expected 54 legal registry candidates, got {len(results)}",
                file=sys.stderr,
            )
            failures += 1
            continue

        from builders.gfx950.gdn.gdn_decode import TOL, check

        auto = dispatch_gdn_decode(request)
        for result in (auto, *results):
            spec: GdnDecodeSpec = result.spec
            arm = "auto" if result is auto else "cand"
            tile = f"{spec.num_warps},{spec.warp_threads_k},{spec.blocks_per_v_dim}"
            grid = result.grid[0]
            out_err, state_err = check(spec, batch, arch=arch)
            err = max(out_err, state_err)
            if err > TOL:
                failures += 1
                print(
                    f"{batch:>6} {arm:>5} {tile:>10} "
                    f"{result.candidate.spec_id:>22} {grid:>8} "
                    f"{'-':>10} {'-':>10}  FAIL max_err={err:.3e}"
                )
                continue
            eager = eager_us(spec, batch, arch)
            device = (
                None
                if args.no_device
                else device_us(spec, batch, arch, args.rotate_mb * 2**20)
            )
            dev_s = f"{device:10.2f}" if device is not None else f"{'n/a':>10}"
            print(
                f"{batch:>6} {arm:>5} {tile:>10} "
                f"{result.candidate.spec_id:>22} {grid:>8} "
                f"{eager:10.2f} {dev_s}  max_err={err:.3e}"
            )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
