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
