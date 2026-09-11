# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx942 candidate and tuned tile selection for GDN decode.

The emitter is shared with gfx950 (``kernels/common/gdn_decode.py``); it is
arch-neutral SSA and its compiled binary contains no CDNA4-only instruction.
What differs per arch is the tile table, which encodes CU count and occupancy.
"""

from __future__ import annotations

import dataclasses as dc
from typing import Tuple

from kernels.common.gdn_decode import (
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

ARCH = "gfx942"

# (max_work, (num_warps, warp_threads_k, blocks_per_v_dim), spec_id)
#
# Keyed on WORK = batch * num_v_heads, not on batch.
#
# MEASURED on gfx942 silicon: an MI300X (304 CUs, gfx942:sramecc+:xnack-) at
# D128, one token, bf16 activations and bf16 state. Every tile the emitter
# accepts was enumerated through ``is_valid_spec`` (54 of them, and the count
# is the same at every head geometry), each was gated against an independent
# fp32 reference, and the survivors were timed with a device clock (chained
# HIP-graph replay, so host submission is off the critical path).
#
# Why work and not batch. The grid is ``batch * num_v_heads * bpv`` and every
# workgroup does identical work, so batch and head count are interchangeable.
# Measured across four head geometries (Hk/Hv of 16/32, 8/16, 4/8, 2/4) and
# six batches, the best achievable time depends only on their product: at
# work=1024 the three cells that reach it differ by 0.2%, at work=256 the four
# cells differ by 2.0% -- both inside the run-to-run noise floor. A table per
# head geometry would encode the same curve four times.
#
# This matters because tensor-parallel sharding divides num_v_heads across
# ranks (32 -> 16 -> 8 -> 4), so a batch-keyed table tuned at Hv=32 picks the
# wrong tile on every multi-GPU deployment. Keying on work covers every rung,
# including ones never swept.
#
# Anchors actually measured: work in {4, 8, 16, 32, 64, 128, 256, 512, 1024,
# 2048, 4096}, from 24 (head geometry x batch) cells. Bands are the fewest
# contiguous ranges that stay close to each anchor's own best tile: four bands
# give geomean 1.018 and worst 1.069 against the per-cell optimum, while five
# bands gain only 1% -- inside the noise floor, so that would be fitting noise.
#
# What is NOT measured: work values strictly between anchors. A band edge is
# interpolation, not a measured crossover.
#
# What this claims: only that the chosen tile is close to the best LEGAL ROCKE
# tile at the measured anchors. It is a claim against our own tile space and
# nothing else. Competitive numbers live in the benchmark artifacts, never here.
#
# ``blocks_per_v_dim`` splits one head's V dimension across workgroups purely
# to manufacture parallelism when work alone cannot fill 304 CUs, which is why
# it only exceeds 1 in the two smallest bands.
_TUNED_TILES = (
    (16, (1, 16, 32), "w16"),
    (64, (4, 8, 4), "w64"),
    (512, (8, 8, 1), "w512"),
    (None, (2, 8, 1), "w_large"),
)

# Every tile the table can produce, for tuners and for the sweep space.
TUNED_SPEC_IDS = tuple(entry[2] for entry in _TUNED_TILES)


def work_for(batch: int, num_v_heads: int) -> int:
    """Workgroups before any V-dimension split -- the table's index."""
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
            work = work_for(req.batch, req.num_v_heads)
            wanted = spec_id_for_work(work)
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
                    f"tuned tile for work {work} (batch {req.batch} x "
                    f"{req.num_v_heads} v-heads) is {wanted!r}, not {spec_id!r}"
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
