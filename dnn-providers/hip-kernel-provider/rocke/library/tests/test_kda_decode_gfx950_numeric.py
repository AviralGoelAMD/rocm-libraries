# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-device numeric checks for the KDA gate kind of the decode kernel.

Scored against the same whole-tensor fp32 reference the GDN mode uses, with the
per-channel gate selected. The reference shares no algebra with the kernel --
no tiling, no warp structure, no cross-lane reductions -- so agreement is
evidence about the algorithm rather than a restatement of the implementation.
Its per-channel path is separately audited on CPU by
``test_kda_decode_reference.py``, including which axis the decay scales.

Both results are compared. The kernel writes its output *and* mutates the
recurrent state in place; checking only the output would let a corrupted state
write ship silently, because nothing reads the state back until the next decode
step.

These lanes need a real gfx950 and ROCm torch, so they are marked ``gpu`` and
skipped elsewhere. The spec rules and the reference are covered by CPU-only
tests so this family still contributes coverage without a device.
"""

from __future__ import annotations

import dataclasses as dc

import pytest

ARCH = "gfx950"

torch = pytest.importorskip("torch", reason="ROCm torch required")

pytestmark = pytest.mark.gpu


def _device_is_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return ARCH in torch.cuda.get_device_properties(0).gcnArchName
    except Exception:
        return False


requires_gfx950 = pytest.mark.skipif(
    not _device_is_gfx950(), reason=f"needs a {ARCH} device"
)


@pytest.fixture(scope="module")
def harness():
    from builders.gfx950.gdn.gdn_decode import TOL, check

    return {"TOL": TOL, "check": check}


def _kda(**kw):
    from kernels.gfx950.gdn_decode import GdnDecodeSpec

    return dc.replace(GdnDecodeSpec(), gate_kind="kda", **kw)


@requires_gfx950
@pytest.mark.parametrize("batch", [1, 3, 16, 64])
def test_kda_simple_path_matches_reference(harness, batch):
    """The one-thread-per-row reference emitter, with a per-channel gate.

    This path owns a whole state row per thread, so the gate vector maps onto
    it directly with no warp-tiled indexing in the way. Proving the numerics
    here first means a later warp-tiled failure is an indexing bug and not a
    gate-formula bug.
    """
    out_err, state_err = harness["check"](_kda(simple=True), batch)

    assert out_err <= harness["TOL"], f"KDA simple output error {out_err:.3e}"
    assert state_err <= harness["TOL"], f"KDA simple state error {state_err:.3e}"


@requires_gfx950
def test_gdn_simple_path_still_matches_reference(harness):
    """The GDN gate must be untouched by the KDA branch living beside it."""
    from kernels.gfx950.gdn_decode import GdnDecodeSpec

    out_err, state_err = harness["check"](GdnDecodeSpec(simple=True), 4)

    assert max(out_err, state_err) <= harness["TOL"]


@requires_gfx950
@pytest.mark.parametrize("batch", [1, 3, 16, 64])
def test_kda_warp_tiled_matches_reference(harness, batch):
    """The production warp-tiled emitter, with a per-channel gate.

    The simple path above already proves the gate formula, so a failure here
    is an indexing bug: this emitter splits the DK axis across lanes, and the
    decay vector has to line up with the state slice each lane owns.
    """
    out_err, state_err = harness["check"](_kda(), batch)

    assert out_err <= harness["TOL"], f"KDA warp-tiled output error {out_err:.3e}"
    assert state_err <= harness["TOL"], f"KDA warp-tiled state error {state_err:.3e}"


@requires_gfx950
@pytest.mark.parametrize("tile", [(4, 16, 8), (2, 8, 2), (1, 8, 1), (8, 16, 1)])
def test_kda_matches_reference_across_tiles(harness, tile):
    """Every shipped tile must be correct with the vector gate.

    The tiles differ in how the DK axis is split across lanes (warp_threads_k)
    and how many state rows each lane carries, so one passing tile says little
    about the others. A tuning table that can route to a wrong kernel is worse
    than no tuning at all.
    """
    num_warps, warp_threads_k, blocks_per_v_dim = tile
    spec = _kda(
        num_warps=num_warps,
        warp_threads_k=warp_threads_k,
        blocks_per_v_dim=blocks_per_v_dim,
    )
    out_err, state_err = harness["check"](spec, batch=16)

    assert out_err <= harness["TOL"], f"KDA tile {tile} output error {out_err:.3e}"
    assert state_err <= harness["TOL"], f"KDA tile {tile} state error {state_err:.3e}"


@requires_gfx950
def test_kda_mha_shape_matches_reference(harness):
    """The shipping MHA shape: Hk == Hv == 32, kv_group 1.

    The GDN shapes are all GQA (Hv > Hk), so this is the first time the gather
    that maps a value head to its key head runs with no grouping at all.
    """
    out_err, state_err = harness["check"](_kda(num_k_heads=32, num_v_heads=32), 8)

    assert out_err <= harness["TOL"], f"KDA MHA output error {out_err:.3e}"
    assert state_err <= harness["TOL"], f"KDA MHA state error {state_err:.3e}"
