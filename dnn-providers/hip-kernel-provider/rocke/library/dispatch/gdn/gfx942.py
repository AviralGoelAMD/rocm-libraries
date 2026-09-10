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

# (max_batch, (num_warps, warp_threads_k, blocks_per_v_dim), spec_id)
#
# MEASURED on gfx942 silicon: an MI300X (304 CUs, gfx942:sramecc+:xnack-) at
# the production decode shape -- 16 K heads, 32 V heads, D128, one token,
# bf16 activations and bf16 state.
#
# How these were chosen. Every tile the emitter accepts on this arch was
# enumerated through ``is_valid_spec``, each one was gated against an
# independent fp32 reference, and the survivors were timed with a device
# clock (chained HIP-graph replay, so host submission is off the critical
# path). The batch anchors actually measured are
# 1, 2, 4, 6, 8, 16, 24, 32, 48, 64, 96, 128 and 256, across three runs with
# different seeds. Bands are then the fewest contiguous ranges that keep every
# measured anchor within a few percent of that anchor's own best tile.
#
# What is *not* measured: batches strictly between two anchors. A band edge
# such as ``<= 24`` is interpolation between the 16 and 24 anchors on one side
# and 32 on the other, not a measured crossover point.
#
# What the table claims. Only that, at the anchors above, the chosen tile is
# close to the best of the legal tiles for that batch -- i.e. it is a claim
# against rocKE's own tile space, and nothing else. It is not a claim about
# any other implementation, and the campaign's competitive numbers live in the
# benchmark artifacts, never in this file.
#
# ``blocks_per_v_dim`` splits one head's V dimension across workgroups purely
# to manufacture parallelism when the batch cannot fill 304 CUs, which is why
# it only exceeds 1 in the smallest band and collapses to 1 everywhere else.
_TUNED_TILES = (
    (2, (4, 8, 4), "b2"),
    (8, (8, 8, 1), "b8"),
    (24, (4, 8, 1), "b24"),
    (None, (2, 8, 1), "b_large"),
)

# Every tile the table can produce, for tuners and for the sweep space.
TUNED_SPEC_IDS = tuple(entry[2] for entry in _TUNED_TILES)


def tile_for_batch(batch: int) -> Tuple[int, int, int]:
    """Tuned ``(num_warps, warp_threads_k, blocks_per_v_dim)`` for ``batch``."""
    for max_batch, tile, _ in _TUNED_TILES:
        if max_batch is None or batch <= max_batch:
            return tile
    raise AssertionError("unreachable: table has an open-ended final band")


def spec_id_for_batch(batch: int) -> str:
    for max_batch, _, spec_id in _TUNED_TILES:
        if max_batch is None or batch <= max_batch:
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
            wanted = spec_id_for_batch(int(req.batch))
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
                    f"tuned tile for batch {req.batch} is {wanted!r}, not {spec_id!r}"
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
