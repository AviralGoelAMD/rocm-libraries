# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx950 candidate registration for GDN decode.

The registry owns the configured tile space, built by the shared candidate
factory in ``common.py``. The kernel validator remains the only authority that
decides which configured tiles are legal for a request. Production GDN ``auto``
uses one static priority order and never consults batch or runtime
measurements; KDA ``auto`` keeps its measured work-keyed table below.
"""

from __future__ import annotations

from typing import Optional, Tuple

from kernels.common.gdn_decode import is_valid_spec
from rocke.dispatch.core import KernelCandidate

from .common import (
    AutoReject,
    GdnDecodeRequest,
    configured_tiles,
    gdn_tile_candidates,
    make_candidate,
    make_spec,
)

ARCH = "gfx950"
GATE_KINDS = ("gdn", "kda")

# KDA keeps its measured work-keyed table. GDN candidate registration below
# owns the full configured tile space and has no GDN batch-winner table.
#
# (max_work, (num_warps, warp_threads_k, blocks_per_v_dim), spec_id)
_TUNED_TILES_KDA = (
    (128, (4, 16, 4), "kda_w128"),
    (512, (1, 16, 4), "kda_w512"),
    (None, (2, 16, 1), "kda_w_large"),
)

# Static GDN ``auto`` tile: the emitter's own default tile (``GdnDecodeSpec()``),
# kept when the registry moved to one static default. It was not re-chosen by
# a registry-wide sweep on gfx950; ``tune.py`` on a gfx950 device reports how far it
# sits from the fastest legal tile in each cell.
DEFAULT_TILE = (2, 16, 8)

CONFIGURED_TILES = configured_tiles(DEFAULT_TILE)

TUNED_SPEC_IDS = tuple(entry[2] for entry in _TUNED_TILES_KDA)


def work_for(batch: int, num_v_heads: int) -> int:
    """KDA's measured selection quantity."""
    return int(batch) * int(num_v_heads)


def tile_for_work(work: int, gate_kind: str = "kda") -> Tuple[int, int, int]:
    if gate_kind != "kda":
        raise ValueError(f"no work-keyed table for gate kind {gate_kind!r}")
    for max_work, tile, _ in _TUNED_TILES_KDA:
        if max_work is None or work <= max_work:
            return tile
    raise AssertionError("unreachable: KDA table has an open-ended final band")


def spec_id_for_work(work: int) -> str:
    for max_work, _, spec_id in _TUNED_TILES_KDA:
        if max_work is None or work <= max_work:
            return spec_id
    raise AssertionError("unreachable: KDA table has an open-ended final band")


def _tile_for_kda_spec_id(spec_id: str) -> Tuple[int, int, int]:
    for _, tile, candidate_spec_id in _TUNED_TILES_KDA:
        if candidate_spec_id == spec_id:
            return tile
    raise KeyError(spec_id)


def _kda_auto(spec_id: str) -> AutoReject:
    """KDA ``auto``: the work band's tile whenever the validator admits it."""

    def reject(req: GdnDecodeRequest) -> Optional[str]:
        work = work_for(req.batch, req.num_v_heads)
        wanted = spec_id_for_work(work)
        if (
            wanted != spec_id
            and is_valid_spec(
                make_spec(req, _tile_for_kda_spec_id(wanted)), arch=req.arch
            )[0]
        ):
            return f"tuned KDA tile for work {work} is {wanted!r}, not {spec_id!r}"
        return None

    return reject


def candidates() -> Tuple[KernelCandidate, ...]:
    """GDN configured candidates followed by KDA measured candidates."""
    gdn = gdn_tile_candidates(
        arch=ARCH, served_gate_kinds=GATE_KINDS, default_tile=DEFAULT_TILE
    )
    kda = tuple(
        make_candidate(
            arch=ARCH,
            served_gate_kinds=GATE_KINDS,
            tile=tile,
            priority=10 + len(gdn) + i,
            auto_reject=_kda_auto(spec_id),
            gate_kind="kda",
            spec_id=spec_id,
        )
        for i, (_, tile, spec_id) in enumerate(_TUNED_TILES_KDA)
    )
    return gdn + kda


def register(registry) -> None:
    registry.extend(candidates())
