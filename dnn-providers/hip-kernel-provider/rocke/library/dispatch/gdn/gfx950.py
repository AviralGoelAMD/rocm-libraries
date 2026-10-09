# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx950 candidate registration for GDN/KDA decode.

The registry owns the configured tile space, registered once per gate kind.
The kernel validator remains the only authority that decides which configured
tiles are legal for a request. Production ``auto`` uses one static tile per
gate kind (and, for KDA, per state width) and never consults batch, head count
or runtime measurements.
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

ARCH = "gfx950"

NUM_WARPS = (1, 2, 4, 8, 16)
WARP_THREADS_K = (1, 2, 4, 8, 16, 32)
BLOCKS_PER_V_DIM = (1, 2, 4, 8, 16, 32)
# GDN ``auto`` is one static tile on purpose; batch only changes the grid.
# This replaced a batch-keyed table that picked (4,16,8) / (2,8,2) / (1,8,1) /
# (8,16,1) for batch <=4 / <=32 / <=128 / larger. No single legal tile matches
# all four of those winners. (2,16,8) was measured on gfx950 against that table
# at batch 1/16/64/256: batch 256 was neutral, batch 64 is slower, and batch 1
# and 16 read slower but were too noisy to call. The batch-64 cost was accepted
# in exchange for one deterministic default. Across every legal tile at those
# batches, (2,16,8) has the lowest geomean and worst-case slowdown against each
# batch's fastest tile, though it is not the fastest at any single batch. To
# revisit, run ``tune.py --gate-kind gdn``; it reports this default's rank and
# its ratio to the fastest legal tile per batch.
DEFAULT_TILE = (2, 16, 8)

_LEXICOGRAPHIC_TILES = tuple(product(NUM_WARPS, WARP_THREADS_K, BLOCKS_PER_V_DIM))
# Known limitation: when the static default is illegal (e.g. head_k_dim
# 64/192), the fallback is the first legal tile in this order -- DEFAULT_TILE,
# then product order starting at (1,1,1), which is register-heavy. Needs a
# footprint-based fallback order. KDA registers the same order.
CONFIGURED_TILES = (DEFAULT_TILE,) + tuple(
    tile for tile in _LEXICOGRAPHIC_TILES if tile != DEFAULT_TILE
)

# KDA registers the same configured tile space as GDN and, like GDN, ``auto``
# is one static tile per state width; batch and heads only change the grid.
#
# 2-byte state (bf16/f16). Chosen on gfx950 by a cold-memory sweep over
# Hk=Hv in {4,8,12,16,24,32,48,96} x batch {1,8,16,32,64,128,256} with the
# load-first + streaming-state kernel: among the shortlisted single tiles,
# (4,16,4) had the lowest geomean and worst-case slowdown against each shape's
# fastest tile, at a small average cost against the work-keyed table it
# replaced (largest at batch 128-256).
KDA_DEFAULT_TILE = (4, 16, 4)
# f32 state. Same shapes and method with the f32-state kernel: (8,16,4) had
# the lowest geomean slowdown against each shape's fastest tile and spills
# nothing; against the f32 work-keyed table it replaced ((4,16,8) for work
# <= 128, else (8,16,4)) it is slightly slower, most at batch 1.
KDA_DEFAULT_TILE_F32 = (8, 16, 4)
# To revisit either, run ``tune.py --gate-kind kda [--state-dtype f32]``.
assert KDA_DEFAULT_TILE in CONFIGURED_TILES, "KDA_DEFAULT_TILE is not configured"
assert (
    KDA_DEFAULT_TILE_F32 in CONFIGURED_TILES
), "KDA_DEFAULT_TILE_F32 is not configured"

# Fused (fuse_conv / fuse_out_norm) requests need one workgroup per head, so
# their static defaults are BPV=1 tiles, one per gate kind and state width.
# Chosen on gfx950 with both flags on (sweep of all 19 legal BPV=1 tiles over
# Hk=Hv in {4,8,12,16,24,32,48,96} x batch {1,8,16,32,64,128,256}): (4,16,1)
# had the lowest geomean slowdown against each shape's fastest tile for every
# (gate, state) pair, with a clear margin over the runners-up; the KDA f32
# pick held under a cold, interleaved re-time. Re-measure with
# ``tune.py --fuse-conv --fuse-out-norm``.
FUSED_DEFAULT_TILES = {
    ("gdn", "bf16"): (4, 16, 1),
    ("gdn", "f32"): (4, 16, 1),
    ("kda", "bf16"): (4, 16, 1),
    ("kda", "f32"): (4, 16, 1),
}
for _fused_tile in FUSED_DEFAULT_TILES.values():
    assert _fused_tile in CONFIGURED_TILES and _fused_tile[2] == 1, _fused_tile

# Spec ids carry the gate kind: GDN's are ``nw{}_wtk{}_bpv{}``, KDA's add a
# ``kda_`` prefix, so the two can never collide in the shared registry.
_KDA_SPEC_ID_PREFIX = "kda_"


def spec_id_for(tile: Tuple[int, int, int], gate_kind: str) -> str:
    """Registry spec id of the ``gate_kind`` candidate that carries ``tile``."""
    num_warps, warp_threads_k, blocks_per_v_dim = tile
    base = f"nw{num_warps}_wtk{warp_threads_k}_bpv{blocks_per_v_dim}"
    return _KDA_SPEC_ID_PREFIX + base if gate_kind == "kda" else base


def auto_tile(
    gate_kind: str, state_dtype: str = "bf16", fused: bool = False
) -> Tuple[int, int, int]:
    """The static ``auto`` tile for a gate kind, state width and fusion mode."""
    width = "f32" if normalize_dtype(state_dtype) == "f32" else "bf16"
    if fused:
        return FUSED_DEFAULT_TILES[(gate_kind, width)]
    if gate_kind != "kda":
        return DEFAULT_TILE
    return KDA_DEFAULT_TILE_F32 if width == "f32" else KDA_DEFAULT_TILE


# ``auto`` cache policy of the state (load, store). Streaming (nontemporal) is
# the default: the state is read and written once per call, and on the unfused
# tiles nontemporal accesses were fastest on gfx950. The fused f32 path is the
# exception and uses the default policy for both. Its BPV=1 tiles give each
# lane several state rows, each stored as two 16 B halves of a 32 B f32 vector;
# with nontemporal stores, write counters showed each 64 B line reaching HBM
# more than once, and default stores wrote only the state's own bytes. A cold
# sweep of every BPV=1 tile with all four (load, store) pairs over
# Hk=Hv in {4,8,12,16,24,32,48,96} x batch {1,8,16,32,64,128,256} found default
# loads and stores on (4,16,1) the best static choice. A 2-byte state stores
# one 16 B vector per row, so it keeps streaming. GDN shares the state access
# code; its fused f32 default follows by construction, not by a separate
# measurement. To revisit, sweep the hints per tile (tune.py --state-load-hint /
# --state-store-hint).
def auto_state_hints(state_dtype: str = "bf16", fused: bool = False) -> Tuple[str, str]:
    """The ``auto`` (state_load_hint, state_store_hint) for a state width and mode."""
    if fused and normalize_dtype(state_dtype) == "f32":
        return "default", "default"
    return "streaming", "streaming"


def make_spec(req: GdnDecodeRequest, tile: Tuple[int, int, int]) -> GdnDecodeSpec:
    """Map a request plus a chosen tile onto a concrete kernel spec."""
    num_warps, warp_threads_k, blocks_per_v_dim = tile
    fused = bool(req.fuse_conv or req.fuse_out_norm)
    auto_load, auto_store = auto_state_hints(req.state_dtype, fused)
    load_hint = auto_load if req.state_load_hint == "auto" else req.state_load_hint
    store_hint = auto_store if req.state_store_hint == "auto" else req.state_store_hint
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
        fuse_conv=bool(req.fuse_conv),
        fuse_out_norm=bool(req.fuse_out_norm),
        state_load_hint=load_hint,
        state_store_hint=store_hint,
    )


def _grid(spec: GdnDecodeSpec, req: OperatorRequest) -> Tuple[int, int, int]:
    assert isinstance(req, GdnDecodeRequest)
    return gdn_decode_grid(int(req.batch), spec)


def _build(spec: GdnDecodeSpec, arch: str):
    return build_gdn_decode(spec, arch=arch)


def _make_candidate(*, tile: Tuple[int, int, int], priority: int, gate_kind: str):
    spec_id = spec_id_for(tile, gate_kind)
    name = f"gdn_decode_{ARCH}_{spec_id}"

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, GdnDecodeRequest)
        if req.arch != ARCH:
            return False, f"candidate arch {ARCH} != request arch {req.arch!r}"
        # A candidate serves exactly one gate kind: the two gates emit
        # different kernels, so a GDN candidate must never answer a KDA
        # request or the reverse.
        if req.gate_kind != gate_kind:
            return False, (
                f"candidate {spec_id!r} serves the {gate_kind!r} gate, request "
                f"asks for {req.gate_kind!r}"
            )
        ok, why = selector_matches(req, candidate)
        if not ok:
            return False, why
        if req.spec_id.strip().lower() == "auto":
            default = auto_tile(
                gate_kind,
                req.state_dtype,
                fused=bool(req.fuse_conv or req.fuse_out_norm),
            )
            default_is_legal = is_valid_spec(make_spec(req, default), arch=req.arch)[0]
            if default_is_legal and tile != default:
                return False, (
                    f"static {gate_kind.upper()} auto tile is {default!r}, "
                    f"not {tile!r}"
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
        # No `bind`: this family is selectable but not launchable through the
        # generic runner, so today EVERY launch goes through the driver's
        # `prepare()` and therefore through `_validate_decode_inputs`. That is
        # the only thing standing between a mis-shaped tensor and an
        # out-of-bounds access -- the kernel emits no buffer descriptor, so
        # there is no `num_records` to clamp one. Whoever adds `bind` must
        # route it through that validator, or the checks stop covering the
        # path callers actually use.
    )
    return candidate


def candidates() -> Tuple[KernelCandidate, ...]:
    """GDN then KDA candidates, one per configured tile for each gate kind."""
    out = []
    for gate_kind in ("gdn", "kda"):
        for tile in CONFIGURED_TILES:
            out.append(
                _make_candidate(tile=tile, priority=10 + len(out), gate_kind=gate_kind)
            )
    return tuple(out)


def register(registry) -> None:
    registry.extend(candidates())
