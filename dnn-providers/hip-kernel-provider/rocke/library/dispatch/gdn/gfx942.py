# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx942 candidate registration for GDN decode.

Same arch-neutral emitter as gfx950 (``kernels/gfx950/gdn_decode.py``),
validated and compiled for gfx942. The registry owns the configured tile
space; the kernel validator is the only authority on which configured tiles
are legal for a request. Production ``auto`` uses one static tile and never
consults batch or runtime measurements. gfx942 serves the scalar GDN gate
only; KDA decode is gfx950-only.
"""

from __future__ import annotations

import dataclasses as dc
from itertools import product
from typing import Tuple

from kernels.gfx950.gdn_decode import (
    GDN_DTYPES,
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

NUM_WARPS = (1, 2, 4, 8, 16)
WARP_THREADS_K = (1, 2, 4, 8, 16, 32)
BLOCKS_PER_V_DIM = (1, 2, 4, 8, 16, 32)
# Provisional: the gfx950 default, legal on gfx942 but not yet measured here.
DEFAULT_TILE = (2, 16, 8)

_LEXICOGRAPHIC_TILES = tuple(product(NUM_WARPS, WARP_THREADS_K, BLOCKS_PER_V_DIM))
CONFIGURED_TILES = (DEFAULT_TILE,) + tuple(
    tile for tile in _LEXICOGRAPHIC_TILES if tile != DEFAULT_TILE
)


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


def _make_candidate(*, tile: Tuple[int, int, int], priority: int):
    spec_id = f"nw{tile[0]}_wtk{tile[1]}_bpv{tile[2]}"
    name = f"gdn_decode_{ARCH}_{spec_id}"

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, GdnDecodeRequest)
        if req.arch != ARCH:
            return False, f"candidate arch {ARCH} != request arch {req.arch!r}"
        if req.gate_kind != "gdn":
            return False, f"gfx942 serves the 'gdn' gate only, not {req.gate_kind!r}"
        ok, why = selector_matches(req, candidate)
        if not ok:
            return False, why
        if req.spec_id.strip().lower() == "auto":
            default_is_legal = is_valid_spec(
                make_spec(req, DEFAULT_TILE), arch=req.arch
            )[0]
            if default_is_legal and tile != DEFAULT_TILE:
                return False, (
                    f"static GDN auto tile is {DEFAULT_TILE!r}, not {tile!r}"
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
        capability=Capability(arches=(ARCH,), dtypes=GDN_DTYPES),
        _supports=support,
        select_spec=select,
        signature=lambda spec: gdn_decode_signature(spec),
        grid=_grid,
        block=lambda spec: (int(spec.block_size), 1, 1),
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
        build=_build,
        # No `bind`, as on gfx950: every launch goes through the driver's
        # `prepare()` and therefore `_validate_decode_inputs`, the only guard
        # against out-of-bounds state-pool access.
    )
    return candidate


def candidates() -> Tuple[KernelCandidate, ...]:
    """Every configured GDN tile, default first."""
    return tuple(
        _make_candidate(tile=tile, priority=10 + i)
        for i, tile in enumerate(CONFIGURED_TILES)
    )


def register(registry) -> None:
    registry.extend(candidates())
