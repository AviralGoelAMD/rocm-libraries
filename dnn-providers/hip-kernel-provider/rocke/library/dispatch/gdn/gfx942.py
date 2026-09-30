# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx942 candidate registration for GDN decode.

Same arch-neutral emitter and candidate factory as gfx950 (``common.py``),
validated and compiled for gfx942. The registry owns the configured tile
space; the kernel validator is the only authority on which configured tiles
are legal for a request. Production ``auto`` uses one static tile and never
consults batch or runtime measurements.

gfx942 serves the scalar GDN gate only. KDA decode is not yet validated here:
the emitter and validator accept it, but it has no on-device coverage and no
gfx942 work-band table, so dispatch refuses it as ``NOT_YET_IMPLEMENTED``.
"""

from __future__ import annotations

from typing import Tuple

from rocke.dispatch.core import KernelCandidate

from .common import configured_tiles, gdn_tile_candidates

ARCH = "gfx942"
GATE_KINDS = ("gdn",)

# Static ``auto`` tile. Chosen with ``tune.py`` on MI300X (gfx942) as the
# tile with the best geometric mean of (tile time / fastest legal tile time)
# over (Hk, Hv) in {(16, 32), (8, 16), (4, 8)} x batch {1, 16, 64, 256}, at the
# default request: D128, bf16 activations and state, qk-l2norm on. Timed
# cold-cache (``--rotate-mb 1024``: every call reads its state from HBM). The
# per-cell winner moves with batch, so no single tile is fastest in every cell;
# the tuner prints each cell and the summary. Re-measure after any emitter,
# compiler or tile-space change.
DEFAULT_TILE = (2, 16, 8)

CONFIGURED_TILES = configured_tiles(DEFAULT_TILE)


def candidates() -> Tuple[KernelCandidate, ...]:
    """Every configured GDN tile, default first."""
    return gdn_tile_candidates(
        arch=ARCH, served_gate_kinds=GATE_KINDS, default_tile=DEFAULT_TILE
    )


def register(registry) -> None:
    registry.extend(candidates())
