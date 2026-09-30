#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Measure GDN registry candidates (gfx942/gfx950) and KDA tile-table alternatives.

GDN dispatch has a static registry priority. Its sweep measures every legal
registered candidate but does not change dispatcher-default selection. KDA keeps
its separate work-keyed table, so its sweep enumerates every validator-admitted
tile to challenge the selected work band.

Every candidate is correctness-gated before device timing. Host launch cost can
hide kernel differences at small batch, so device time is the comparison metric.
Device time is cold-cache by default: each timed graph cycles through enough
input copies to touch ``--rotate-mb`` MB, so every call reads its state from
HBM as in a real model (``--rotate-mb 0``: warm, debugging only). Kernels are
compiled for the visible device's arch; the first output line records the arch,
device, torch build and cache mode.

After the per-cell tables, a GDN run prints a static-tile summary: for every
tile measured in all cells, the geometric mean of (tile time / fastest tile in
that cell). That is the statistic ``DEFAULT_TILE`` is chosen by.

Run GDN with its default batch anchors and geometries::

    PYTHONPATH=<rocke>/library:<rocke>/platform/python python3 tune.py \
        --geometries 16/32,8/16,4/8

Run the KDA work-keying study across several head geometries::

    PYTHONPATH=<rocke>/library:<rocke>/platform/python python3 tune.py --gate-kind kda \
        --geometries 16/32,8/16,4/8 \
        --batches 1,2,4,8,16,32,64,128 --top 5
"""

from __future__ import annotations

import argparse
import dataclasses as dc
import math
import sys


from dispatch.gdn import (
    GDN_DECODE_ARCHES,
    KDA_DECODE_ARCHES,
    GdnDecodeRequest,
    dispatch_gdn_decode,
    dispatch_gdn_decode_all,
)
from dispatch.gdn.common import BLOCKS_PER_V_DIM, NUM_WARPS, WARP_THREADS_K
from kernels.common.gdn_decode import GdnDecodeSpec, is_valid_spec

DEFAULT_BATCHES = (1, 16, 64, 256)

# KDA's study deliberately searches the registry's configured tile space. GDN
# dispatcher auto uses the registry instead, so its sweep only receives registry
# dispatch results.


def device_is_visible() -> bool:
    """Load the ROCm-only measurement backend only when tuning is requested."""
    global TOL, compare_to_reference, device_arch, graph_device_us, launch
    global launcher_for, make_inputs, prepare, ref_fp32, rotation_input_sets, torch
    try:
        import torch as torch_module
        from builders.gfx950.gdn.gdn_decode import (
            TOL,
            compare_to_reference,
            device_arch,
            graph_device_us,
            launch,
            launcher_for,
            make_inputs,
            prepare,
            ref_fp32,
            rotation_input_sets,
        )
    except ModuleNotFoundError:
        return False
    torch = torch_module
    return torch.cuda.is_available()


def describe_device() -> str:
    """Device name and torch build, for the provenance line."""
    return f"{torch.cuda.get_device_name(0)} torch={torch.__version__}"


def legal_configs(base: GdnDecodeSpec, arch: str):
    """Every validator-admitted tile for KDA's exhaustive tuning study."""
    out = []
    for num_warps in NUM_WARPS:
        for warp_threads_k in WARP_THREADS_K:
            for blocks_per_v_dim in BLOCKS_PER_V_DIM:
                spec = dc.replace(
                    base,
                    num_warps=num_warps,
                    warp_threads_k=warp_threads_k,
                    blocks_per_v_dim=blocks_per_v_dim,
                )
                if is_valid_spec(spec, arch=arch)[0]:
                    out.append((num_warps, warp_threads_k, blocks_per_v_dim))
    return out


def compile_result(result):
    """Compile a dispatch result for the arch its request names."""
    return launcher_for(result.spec, arch=result.request.arch)


def time_cold(spec, inp, batch: int, launcher, rotate_bytes: int):
    """Device time per launch over rotated input copies (see module doc)."""
    prepared = [
        prepare(spec, s, batch) for s in rotation_input_sets(inp, batch, rotate_bytes)
    ]
    return graph_device_us(launcher, prepared)


def sweep_registry_batch(batch: int, results, rotate_bytes: int):
    """Return correct, timed GDN registry candidates for one batch, fastest first."""
    if not results:
        return []

    base = results[0].spec
    inp = make_inputs(base, batch)
    ref_out, ref_state = ref_fp32(base, inp)

    rows = []
    for result in results:
        spec = result.spec
        tile = (spec.num_warps, spec.warp_threads_k, spec.blocks_per_v_dim)
        try:
            launcher = compile_result(result)
        except Exception as exc:
            print(
                f"  {result.candidate.spec_id} compile failed: {type(exc).__name__}",
                file=sys.stderr,
            )
            continue
        values, cfg = prepare(spec, inp, batch)
        launch(launcher, values, cfg)
        torch.cuda.synchronize()
        err = max(
            compare_to_reference(
                values["out"],
                values["state"],
                inp["state"],
                inp["write_indices"],
                ref_out,
                ref_state,
            )
        )
        if err > TOL:
            print(
                f"  {result.candidate.spec_id} INCORRECT err={err:.3e}", file=sys.stderr
            )
            continue
        micros = time_cold(spec, inp, batch, launcher, rotate_bytes)
        if micros is None:
            print(
                f"  {result.candidate.spec_id} graph capture failed; not ranked",
                file=sys.stderr,
            )
        else:
            rows.append((micros, tile, result.candidate.spec_id, err))
    rows.sort()
    return rows


def sweep_batch(base: GdnDecodeSpec, batch: int, configs, arch: str, rotate_bytes: int):
    """Return correct, timed KDA configurations for one batch, fastest first."""
    inp = make_inputs(base, batch)
    ref_out, ref_state = ref_fp32(base, inp)

    rows = []
    for tile in configs:
        spec = dc.replace(
            base,
            num_warps=tile[0],
            warp_threads_k=tile[1],
            blocks_per_v_dim=tile[2],
        )
        try:
            launcher = launcher_for(spec, arch=arch)
        except Exception as exc:
            print(f"  {tile} compile failed: {type(exc).__name__}", file=sys.stderr)
            continue
        values, cfg = prepare(spec, inp, batch)
        launch(launcher, values, cfg)
        torch.cuda.synchronize()
        err = max(
            compare_to_reference(
                values["out"],
                values["state"],
                inp["state"],
                inp["write_indices"],
                ref_out,
                ref_state,
            )
        )
        if err > TOL:
            print(f"  {tile} INCORRECT err={err:.3e}", file=sys.stderr)
            continue
        micros = time_cold(spec, inp, batch, launcher, rotate_bytes)
        if micros is None:
            print(f"  {tile} graph capture failed; not ranked", file=sys.stderr)
        else:
            rows.append((micros, tile, err))
    rows.sort()
    return rows


def report_gdn_dispatcher_default(rows, auto_id: str) -> None:
    """Print the shipped GDN default against the fastest measured candidate."""
    best_micros, best_tile, best_id, _ = rows[0]
    for rank, (micros, tile, spec_id, _) in enumerate(rows, start=1):
        if spec_id != auto_id:
            continue
        print(
            f"  dispatcher default: {micros:.3f}us  {spec_id} "
            f"tile={tile} rank={rank}/{len(rows)}"
        )
        print(
            f"  fastest legal candidate: {best_micros:.3f}us  {best_id} "
            f"tile={best_tile}"
        )
        print(f"  default / fastest = {micros / best_micros:.3f}x")
        if spec_id == best_id:
            print("  manual review: retain DEFAULT_TILE")
        else:
            print(f"  manual review: consider DEFAULT_TILE = {best_tile}")
        return
    raise RuntimeError(f"dispatcher-selected GDN spec {auto_id!r} was not measured")


def report_static_tile_summary(cells, auto_ids, top: int) -> None:
    """Rank every tile measured in all cells by geomean(tile / cell's fastest).

    ``cells`` holds one ``rows`` list per measured (geometry, batch) cell, as
    returned by :func:`sweep_registry_batch`. A tile missing from any cell is
    left out, because its geomean would be over a different cell set.
    """
    if not cells:
        return
    ratios = {}
    tiles = {}
    for rows in cells:
        fastest = rows[0][0]
        for micros, tile, spec_id, _ in rows:
            ratios.setdefault(spec_id, []).append(micros / fastest)
            tiles[spec_id] = tile
    ranked = sorted(
        (math.exp(sum(math.log(r) for r in values) / len(values)), spec_id)
        for spec_id, values in ratios.items()
        if len(values) == len(cells)
    )
    print(
        f"\n=== static-tile summary over {len(cells)} cells: "
        f"geomean of (tile time / fastest tile in cell) ==="
    )
    auto_id = auto_ids[0] if len(set(auto_ids)) == 1 else None
    for geomean, spec_id in ranked[:top]:
        mark = " <- dispatcher default" if spec_id == auto_id else ""
        print(f"  {geomean:6.3f}x  {spec_id} tile={tiles[spec_id]}{mark}")
    if auto_id is None:
        print("  dispatcher default differs between cells; no single default to rank")
        return
    for rank, (geomean, spec_id) in enumerate(ranked, start=1):
        if spec_id == auto_id:
            print(
                f"  dispatcher default {spec_id}: rank {rank}/{len(ranked)} "
                f"geomean {geomean:.3f}x"
            )
            break
    else:
        print(f"  dispatcher default {auto_id} was not measured in every cell")
        return
    best_id = ranked[0][1]
    if best_id == auto_id:
        print("  manual review: retain DEFAULT_TILE")
    else:
        print(f"  manual review: consider DEFAULT_TILE = {tiles[best_id]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batches",
        default=",".join(str(batch) for batch in DEFAULT_BATCHES),
        help="comma-separated decode batch sizes",
    )
    parser.add_argument(
        "--gate-kind",
        default="gdn",
        choices=("gdn", "kda"),
        help="forget-gate granularity to tune for",
    )
    parser.add_argument(
        "--geometries",
        default="16/32",
        help="comma-separated num_k_heads/num_v_heads pairs",
    )
    parser.add_argument("--top", type=int, default=8, help="rows to print per cell")
    parser.add_argument(
        "--arch",
        default=None,
        choices=GDN_DECODE_ARCHES,
        help="target arch (default: the visible device); must match the device",
    )
    parser.add_argument(
        "--rotate-mb",
        type=int,
        default=1024,
        help="cold memory (default): rotate input copies so one graph cycle "
        "touches >= N MB (4x the MI300X 256 MB Infinity Cache); 0 = warm cache, "
        "debugging only",
    )
    args = parser.parse_args()

    def kda_unsupported(arch: str) -> str:
        return (
            f"NOT_YET_IMPLEMENTED: KDA decode is registered on {KDA_DECODE_ARCHES} "
            f"only, not {arch}"
        )

    if args.gate_kind == "kda" and args.arch and args.arch not in KDA_DECODE_ARCHES:
        parser.error(kda_unsupported(args.arch))

    if not device_is_visible():
        print("no HIP device visible", file=sys.stderr)
        return 2
    arch = device_arch()
    if arch not in GDN_DECODE_ARCHES:
        print(
            f"GDN decode supports {GDN_DECODE_ARCHES}; device is {arch}",
            file=sys.stderr,
        )
        return 2
    if args.arch is not None and args.arch != arch:
        print(f"--arch {args.arch} does not match the device ({arch})", file=sys.stderr)
        return 2
    if args.gate_kind == "kda" and arch not in KDA_DECODE_ARCHES:
        print(kda_unsupported(arch), file=sys.stderr)
        return 2
    rotate_bytes = args.rotate_mb * 2**20
    cache = f"cold (>= {args.rotate_mb} MB per cycle)" if args.rotate_mb else "warm"
    print(
        f"# tune.py arch={arch} gate={args.gate_kind} device={describe_device()} "
        f"cache={cache}"
    )

    batches = [int(value) for value in args.batches.split(",")]
    geometries = [
        tuple(int(value) for value in item.split("/"))
        for item in args.geometries.split(",")
    ]
    failed = False
    cells, auto_ids = [], []
    for num_k_heads, num_v_heads in geometries:
        for batch in batches:
            request = GdnDecodeRequest(
                batch=batch,
                arch=arch,
                gate_kind=args.gate_kind,
                num_k_heads=num_k_heads,
                num_v_heads=num_v_heads,
            )
            if args.gate_kind == "kda":
                # KDA's measured table is work-keyed, but the study must test
                # every validator-admitted tile rather than the current band's
                # dispatcher result.
                auto = dispatch_gdn_decode(request)
                base = auto.spec
                configs = legal_configs(base, arch)
                print(f"legal KDA configurations for batch {batch}: {len(configs)}")
                rows = sweep_batch(base, batch, configs, arch, rotate_bytes)
                if not rows:
                    print(f"batch {batch}: no candidate was both correct and timeable")
                    failed = True
                    continue
                auto_tile = (
                    base.num_warps,
                    base.warp_threads_k,
                    base.blocks_per_v_dim,
                )
                print(
                    f"\n=== {arch} Hk{num_k_heads}/Hv{num_v_heads} batch {batch}: "
                    f"top {args.top} ==="
                )
                for micros, tile, err in rows[: args.top]:
                    mark = " <- dispatcher default" if tile == auto_tile else ""
                    print(f"  {micros:9.3f}us tile={tile} err={err:.2e}{mark}")
                continue

            results = dispatch_gdn_decode_all(request)
            print(f"legal registry candidates for batch {batch}: {len(results)}")
            rows = sweep_registry_batch(batch, results, rotate_bytes)
            if not rows:
                print(f"batch {batch}: no candidate was both correct and timeable")
                failed = True
                continue
            auto = dispatch_gdn_decode(request)
            auto_id = auto.candidate.spec_id
            print(
                f"\n=== {arch} Hk{num_k_heads}/Hv{num_v_heads} batch {batch}: "
                f"top {args.top} ==="
            )
            for micros, tile, spec_id, err in rows[: args.top]:
                mark = " <- dispatcher default" if spec_id == auto_id else ""
                print(f"  {micros:9.3f}us  {spec_id} tile={tile} err={err:.2e}{mark}")
            report_gdn_dispatcher_default(rows, auto_id)
            cells.append(rows)
            auto_ids.append(auto_id)

    if args.gate_kind == "gdn":
        report_static_tile_summary(cells, auto_ids, args.top)

    print(
        "\nDispatcher default is deterministic; measurements do not change selection."
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
