#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Re-measure the GDN decode tuning table.

The per-batch tile table in ``dispatch/gdn/gfx950.py`` is an empirical claim,
so the measurement that produced it lives here rather than outside the tree:
anyone can re-run it, challenge a band, or re-tune after a kernel change.

Device time is the metric. Host launch cost is identical across tiles and, at
small batch, larger than the kernel itself, so wall time would mask exactly the
differences the table is choosing between.

Every configuration is correctness-gated before it is timed. The reference
depends only on the batch, not the tile, so it is computed once per batch and
reused across the whole configuration space.

Run::

    PYTHONPATH=<rocke>/library:<rocke>/platform/python python3 tune.py
    PYTHONPATH=... python3 tune.py --batches 1,16 --top 5
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
from kernels.gfx950.gdn_decode import GdnDecodeSpec, is_valid_spec

ARCH = "gfx950"
DEFAULT_BATCHES = (1, 16, 64, 256)

# Search space. Anything illegal for the requested shape is pruned by the
# kernel's own validator rather than by a second copy of its rules here.
_NUM_WARPS = (1, 2, 4, 8, 16)
_WARP_THREADS_K = (1, 2, 4, 8, 16, 32)
_BLOCKS_PER_V = (1, 2, 4, 8, 16, 32)


def legal_configs(base: GdnDecodeSpec):
    out = []
    for num_warps in _NUM_WARPS:
        for warp_threads_k in _WARP_THREADS_K:
            for blocks_per_v_dim in _BLOCKS_PER_V:
                spec = dc.replace(
                    base,
                    num_warps=num_warps,
                    warp_threads_k=warp_threads_k,
                    blocks_per_v_dim=blocks_per_v_dim,
                )
                ok, _ = is_valid_spec(spec, arch=ARCH)
                if ok:
                    out.append((num_warps, warp_threads_k, blocks_per_v_dim))
    return out


def device_us(values, cfg, launcher, reps: int = 32):
    """Per-launch device time from a replayed graph, or None if capture fails."""
    for _ in range(10):
        launch(launcher, values, cfg)
    torch.cuda.synchronize()
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(reps):
                launch(launcher, values, cfg)
    except Exception:
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


def sweep_batch(base: GdnDecodeSpec, batch: int, configs):
    """Correct, timed configurations for one batch, fastest first."""
    inp = make_inputs(base, batch)
    ref_out, ref_state = ref_fp32(base, inp)
    written = inp["write_indices"].long()

    rows = []
    for tile in configs:
        spec = dc.replace(
            base,
            num_warps=tile[0],
            warp_threads_k=tile[1],
            blocks_per_v_dim=tile[2],
        )
        try:
            launcher = launcher_for(spec, arch=ARCH)
        except Exception as exc:
            print(f"  {tile} compile failed: {type(exc).__name__}", file=sys.stderr)
            continue
        values, cfg = prepare(spec, inp, batch)
        launch(launcher, values, cfg)
        torch.cuda.synchronize()
        err = max(
            (values["out"].float() - ref_out).abs().max().item(),
            (values["state"].float()[written] - ref_state).abs().max().item(),
        )
        if err > TOL:
            print(f"  {tile} INCORRECT err={err:.3e}", file=sys.stderr)
            continue
        micros = device_us(values, cfg, launcher)
        if micros is not None:
            rows.append((micros, tile, err))
    rows.sort()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--batches",
        default=",".join(str(b) for b in DEFAULT_BATCHES),
        help="comma-separated decode batch sizes to tune for",
    )
    ap.add_argument(
        "--gate-kind",
        default="gdn",
        choices=("gdn", "kda"),
        help="forget-gate granularity to tune for",
    )
    ap.add_argument(
        "--geometries",
        default="16/32",
        help=(
            "comma-separated num_k_heads/num_v_heads pairs. More than one turns "
            "the run into a test of the work-keying hypothesis: cells sharing "
            "batch * num_v_heads should agree on the best tile."
        ),
    )
    ap.add_argument("--top", type=int, default=8, help="rows to print per cell")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no HIP device visible", file=sys.stderr)
        return 2

    geometries = []
    for item in args.geometries.split(","):
        hk, hv = item.split("/")
        geometries.append((int(hk), int(hv)))
    batches = [int(x) for x in args.batches.split(",")]

    by_work = {}  # work -> [(us, tile, batch, hv), ...]

    for hk, hv in geometries:
        base = dc.replace(
            GdnDecodeSpec(),
            gate_kind=args.gate_kind,
            num_k_heads=hk,
            num_v_heads=hv,
        )
        configs = legal_configs(base)
        print(f"\n### geometry Hk={hk} Hv={hv}: {len(configs)} legal configurations")
        for batch in batches:
            rows = sweep_batch(base, batch, configs)
            if not rows:
                print(f"  batch {batch}: nothing both correct and timeable")
                continue
            work = batch * hv
            print(
                f"\n=== Hk{hk}/Hv{hv} batch {batch} (work {work}): "
                f"top {args.top} of {len(rows)} ==="
            )
            for micros, tile, err in rows[: args.top]:
                print(
                    f"  {micros:9.3f}us  num_warps={tile[0]} "
                    f"warp_threads_k={tile[1]} blocks_per_v_dim={tile[2]}  "
                    f"err={err:.2e}"
                )
            by_work.setdefault(work, []).append((rows[0][0], rows[0][1], batch, hv))

    # Does the best tile depend only on the product? Cells sharing a work value
    # but differing in (batch, heads) are the evidence either way. A real
    # disagreement here invalidates the table's KEY, not just its values.
    print("\n=== work -> best tile, across geometries ===")
    print(f"{'work':>7}  {'best tile':16} {'us':>9}  cells (batch x Hv)")
    disagreements = []
    for work in sorted(by_work):
        cells = by_work[work]
        tiles = {c[1] for c in cells}
        fastest = min(cells)
        cellstr = " ".join(f"{b}x{h}" for _, _, b, h in cells)
        flag = "" if len(tiles) == 1 else "   <-- TILES DISAGREE"
        if len(tiles) > 1:
            disagreements.append((work, sorted(tiles)))
        print(f"{work:>7}  {str(fastest[1]):16} {fastest[0]:9.3f}  {cellstr}{flag}")

    if disagreements:
        print(
            f"\nWARNING: work alone did not fix the best tile at "
            f"{len(disagreements)} work value(s). Before banding, check whether "
            "the disagreeing times sit inside run-to-run variation. If they do "
            "not, the table must not be keyed on work."
        )
    print(
        "\nBand _TUNED_TILES in dispatch/gdn/gfx950.py from the work column, "
        "record which work values were measured and which band edges are "
        "interpolated, and re-run the wiring test."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
