# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx950 candidate and tuned tile selection for GDN decode.

This module owns the two arch-specific decisions: what the chip can serve
(``Capability`` plus the residual predicate, which ends in the kernel's own
``is_valid_spec``) and how a request is turned into a concrete spec.

The tile selection is the interesting part. ``blocks_per_v_dim`` splits each
value head's V dimension across several workgroups purely to manufacture
parallelism; it costs redundant work per split. At small batch there are too
few sequences to fill the machine, so paying that cost buys occupancy. As the
batch grows the launch already has ample parallelism and the split becomes
overhead, so the tuned tile collapses to one workgroup per head and instead
widens the workgroup. That trend, not any individual measurement, is what this
table encodes.
"""

from __future__ import annotations

import dataclasses as dc
from typing import Tuple

from kernels.gfx950.gdn_decode import (
    GdnDecodeSpec,
    build_gdn_decode,
    gdn_decode_grid,
    gdn_decode_signature,
    is_valid_spec,
)
from rocke.dispatch.core import Capability, KernelCandidate, OperatorRequest

from .common import (
    FAMILY,
    GDN_ABI_VERSION,
    GdnDecodeRequest,
    normalize_dtype,
    request_errors,
    selector_matches,
)

ARCH = "gfx950"

# (max_work, (num_warps, warp_threads_k, blocks_per_v_dim), spec_id)
#
# Keyed on WORK = batch * num_v_heads, not on batch. The grid is
# batch * num_v_heads * blocks_per_v_dim and every workgroup does identical
# work, so batch and head count are interchangeable. This matters because
# tensor-parallel sharding divides num_v_heads across ranks (32 -> 16 -> 8 ->
# 4): a batch-keyed table tuned at Hv=32 picks the wrong tile on every
# multi-GPU deployment, including rungs no sweep ever visited.
#
# !!! PROVISIONAL -- DO NOT SHIP !!!
# These entries are the previous batch-keyed values re-expressed at Hv=32
# (batch b -> work 32b). They have NOT been measured against the per-channel
# KDA gate, which adds loads and registers and can move the optimum, and the
# work axis itself has not been swept. Task 8 of the implementation plan
# replaces this table with measured values and rewrites this comment with the
# usual provenance: what was measured, what is interpolated, and what the
# table does not claim.
_TUNED_TILES = (
    (128, (4, 16, 8), "w128"),
    (1024, (2, 8, 2), "w1024"),
    (4096, (1, 8, 1), "w4096"),
    (None, (8, 16, 1), "w_large"),
)

# Every tile the table can produce, for tuners and for the sweep space.
TUNED_SPEC_IDS = tuple(entry[2] for entry in _TUNED_TILES)


def work_for(batch: int, num_v_heads: int) -> int:
    """The quantity the tile table is keyed on."""
    return int(batch) * int(num_v_heads)


def tile_for_work(work: int) -> Tuple[int, int, int]:
    """Tuned ``(num_warps, warp_threads_k, blocks_per_v_dim)`` for ``work``."""
    for max_work, tile, _ in _TUNED_TILES:
        if max_work is None or work <= max_work:
            return tile
    raise AssertionError("unreachable: table has an open-ended final band")


def spec_id_for_work(work: int) -> str:
    for max_work, _, spec_id in _TUNED_TILES:
        if max_work is None or work <= max_work:
            return spec_id
    raise AssertionError("unreachable: table has an open-ended final band")


def _tile_for_spec_id(spec_id: str) -> Tuple[int, int, int]:
    for _, tile, sid in _TUNED_TILES:
        if sid == spec_id:
            return tile
    raise KeyError(spec_id)


def make_spec(req: GdnDecodeRequest, tile: Tuple[int, int, int]) -> GdnDecodeSpec:
    """Map a request plus a chosen tile onto a concrete kernel spec."""
    num_warps, warp_threads_k, blocks_per_v_dim = tile
    return dc.replace(
        GdnDecodeSpec(),
        num_k_heads=int(req.num_k_heads),
        num_v_heads=int(req.num_v_heads),
        head_k_dim=int(req.head_k_dim),
        head_v_dim=int(req.head_v_dim),
        dtype=normalize_dtype(req.dtype),
        state_dtype=normalize_dtype(req.state_dtype),
        use_qk_l2norm=bool(req.use_qk_l2norm),
        gate_kind=str(req.gate_kind),
        num_warps=num_warps,
        warp_threads_k=warp_threads_k,
        blocks_per_v_dim=blocks_per_v_dim,
    )


def _grid(spec: GdnDecodeSpec, req: OperatorRequest) -> Tuple[int, int, int]:
    assert isinstance(req, GdnDecodeRequest)
    return gdn_decode_grid(int(req.batch), spec)


def _build(spec: GdnDecodeSpec, arch: str):
    return build_gdn_decode(spec, arch=arch)


def _make_candidate(*, tile: Tuple[int, int, int], spec_id: str, priority: int):
    name = f"gdn_decode_{ARCH}_{spec_id}"

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, GdnDecodeRequest)
        if req.arch != ARCH:
            return False, f"candidate arch {ARCH} != request arch {req.arch!r}"
        ok, why = selector_matches(req, candidate)
        if not ok:
            return False, why
        # Under ``auto`` only the candidate the tuning table names may serve the
        # request, so selection is decided by measurement rather than by
        # registration order. An explicit ``spec_id`` pin bypasses this, which
        # is what makes a tuning sweep able to force a non-default tile.
        if req.spec_id.strip().lower() == "auto":
            wanted = spec_id_for_work(work_for(req.batch, req.num_v_heads))
            # Prefer the tuned tile, but only when it is valid for this geometry.
            # If it is not, fall through so any valid candidate may serve (the
            # registry picks by priority) rather than failing a kernel-supported
            # request.
            if (
                wanted != spec_id
                and is_valid_spec(
                    make_spec(req, _tile_for_spec_id(wanted)), arch=req.arch
                )[0]
            ):
                return False, (
                    f"tuned tile for work {work_for(req.batch, req.num_v_heads)} "
                    f"(batch {req.batch} x {req.num_v_heads} heads) is {wanted!r}, "
                    f"not {spec_id!r}"
                )
        # Final authority is the kernel's own validator.
        return is_valid_spec(make_spec(req, tile), arch=req.arch)

    def select(req: OperatorRequest) -> GdnDecodeSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, GdnDecodeRequest)
        return make_spec(req, tile)

    candidate = KernelCandidate(
        name=name,
        family=FAMILY,
        algorithm="warp_tiled",
        spec_id=spec_id,
        abi_version=GDN_ABI_VERSION,
        priority=priority,
        capability=Capability(arches=(ARCH,), dtypes=("bf16", "f16")),
        _supports=support,
        select_spec=select,
        signature=lambda spec: gdn_decode_signature(spec),
        grid=_grid,
        block=lambda spec: (int(spec.block_size), 1, 1),
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
        build=_build,
    )
    return candidate


def candidates() -> Tuple[KernelCandidate, ...]:
    """One candidate per tuned tile, in table order."""
    return tuple(
        _make_candidate(tile=tile, spec_id=spec_id, priority=10 + i)
        for i, (_, tile, spec_id) in enumerate(_TUNED_TILES)
    )


def register(registry) -> None:
    registry.extend(candidates())
