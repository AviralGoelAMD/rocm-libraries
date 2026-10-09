# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-device numerics for the fused conv1d / gated-RMSNorm decode modes.

Scored against the whole-tensor fp32 reference with the fused steps applied
(``ref_fp32`` runs the conv1d + SiLU first and the gated RMSNorm last). ``check``
compares the output, the written recurrent state, the written conv state, and
every untouched page of both pools, so a misplaced or racing write fails.

Needs a real gfx950 and ROCm torch; marked ``gpu`` and skipped elsewhere. The
spec rules, ABI, reference and IR are covered on CPU by
``test_gdn_decode_fused.py``.
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

FLAGS = [(True, False), (False, True), (True, True)]


def _request(**kw):
    from dispatch.gdn import GdnDecodeRequest

    base = dict(batch=3, arch=ARCH, num_k_heads=16, num_v_heads=16)
    base.update(kw)
    return GdnDecodeRequest(**base)


@requires_gfx950
@pytest.mark.parametrize("gate", ["gdn", "kda"])
@pytest.mark.parametrize("state_dtype", ["bf16", "f32"])
@pytest.mark.parametrize("flags", FLAGS, ids=["cv", "rn", "cv_rn"])
@pytest.mark.parametrize("batch", [1, 3, 16])
def test_auto(gate, state_dtype, flags, batch):
    from builders.gfx950.gdn.gdn_decode import TOL, check
    from dispatch.gdn import dispatch_gdn_decode

    conv, norm = flags
    spec = dispatch_gdn_decode(
        _request(
            batch=batch,
            gate_kind=gate,
            state_dtype=state_dtype,
            fuse_conv=conv,
            fuse_out_norm=norm,
        )
    ).spec
    out_err, state_err = check(spec, batch)
    assert out_err < TOL and state_err < TOL, (spec.kernel_name(), out_err, state_err)


@requires_gfx950
@pytest.mark.parametrize("gate", ["gdn", "kda"])
@pytest.mark.parametrize("state_dtype", ["bf16", "f32"])
def test_every_legal_fused_tile(gate, state_dtype):
    """Every BPV=1 registry tile the validator admits is exact, both flags on."""
    from builders.gfx950.gdn.gdn_decode import TOL, check
    from dispatch.gdn import dispatch_gdn_decode_all

    results = dispatch_gdn_decode_all(
        _request(
            gate_kind=gate, state_dtype=state_dtype, fuse_conv=True, fuse_out_norm=True
        )
    )
    assert results, "no legal fused candidate"
    for result in results:
        spec = result.spec
        assert spec.blocks_per_v_dim == 1, spec.kernel_name()
        out_err, state_err = check(spec, 3)
        assert out_err < TOL and state_err < TOL, (
            spec.kernel_name(),
            out_err,
            state_err,
        )


@requires_gfx950
@pytest.mark.parametrize("gate", ["gdn", "kda"])
@pytest.mark.parametrize("state_dtype", ["bf16", "f32"])
@pytest.mark.parametrize(
    "norm, gate_once",
    [(False, True), (True, True), (True, False)],
    ids=["cv", "cv_rn", "cv_rn_nglane"],
)
@pytest.mark.parametrize("batch", [1, 3, 16])
def test_conv_once_every_legal_fused_tile(gate, state_dtype, norm, gate_once, batch):
    """conv_once on every BPV=1 registry tile the validator admits: output,
    written state and conv taps, and untouched pages, against the oracle."""
    from builders.gfx950.gdn.gdn_decode import TOL, check
    from dispatch.gdn import dispatch_gdn_decode_all

    results = dispatch_gdn_decode_all(
        _request(
            batch=batch,
            gate_kind=gate,
            state_dtype=state_dtype,
            fuse_conv=True,
            fuse_out_norm=norm,
            conv_once=True,
            norm_gate_once=gate_once,
        )
    )
    assert results, "no legal conv_once candidate"
    for result in results:
        spec = result.spec
        assert spec.conv_once and spec.blocks_per_v_dim == 1, spec.kernel_name()
        out_err, state_err = check(spec, batch)
        assert out_err < TOL and state_err < TOL, (
            spec.kernel_name(),
            out_err,
            state_err,
        )


@requires_gfx950
@pytest.mark.parametrize("gate", ["gdn", "kda"])
@pytest.mark.parametrize("state_dtype", ["bf16", "f32"])
@pytest.mark.parametrize("norm", [False, True], ids=["cv", "cv_rn"])
@pytest.mark.parametrize("batch", [1, 3])
def test_conv_once_dpp_reduce_every_legal_fused_tile(gate, state_dtype, norm, batch):
    """dpp_reduce (wsum's xor-4 / xor-8 stages on DPP row mirrors) on every
    BPV=1 conv_once tile, against the oracle."""
    from builders.gfx950.gdn.gdn_decode import TOL, check
    from dispatch.gdn import dispatch_gdn_decode_all

    results = dispatch_gdn_decode_all(
        _request(
            batch=batch,
            gate_kind=gate,
            state_dtype=state_dtype,
            fuse_conv=True,
            fuse_out_norm=norm,
            conv_once=True,
            dpp_reduce=True,
        )
    )
    assert results, "no legal dpp_reduce conv_once candidate"
    for result in results:
        spec = result.spec
        assert spec.dpp_reduce, spec.kernel_name()
        out_err, state_err = check(spec, batch)
        assert out_err < TOL and state_err < TOL, (
            spec.kernel_name(),
            out_err,
            state_err,
        )


@requires_gfx950
@pytest.mark.parametrize("gate", ["gdn", "kda"])
@pytest.mark.parametrize("state_dtype", ["bf16", "f32"])
@pytest.mark.parametrize("batch", [1, 16])
def test_dpp_reduce_every_legal_plain_tile(gate, state_dtype, batch):
    """dpp_reduce on every unfused registry tile (default GQA heads, every
    BPV), against the oracle."""
    from builders.gfx950.gdn.gdn_decode import TOL, check
    from dispatch.gdn import GdnDecodeRequest, dispatch_gdn_decode_all

    results = dispatch_gdn_decode_all(
        GdnDecodeRequest(
            batch=batch,
            arch=ARCH,
            gate_kind=gate,
            state_dtype=state_dtype,
            dpp_reduce=True,
        )
    )
    assert results, "no legal dpp_reduce candidate"
    for result in results:
        spec = result.spec
        assert spec.dpp_reduce, spec.kernel_name()
        out_err, state_err = check(spec, batch)
        assert out_err < TOL and state_err < TOL, (
            spec.kernel_name(),
            out_err,
            state_err,
        )


@requires_gfx950
@pytest.mark.parametrize("gate", ["gdn", "kda"])
@pytest.mark.parametrize("state_dtype", ["bf16", "f32"])
def test_conv_once_with_waves_per_eu(gate, state_dtype):
    """waves_per_eu=4 changes register allocation only, never the result."""
    from builders.gfx950.gdn.gdn_decode import TOL, check
    from dispatch.gdn import dispatch_gdn_decode

    spec = dispatch_gdn_decode(
        _request(
            batch=16,
            gate_kind=gate,
            state_dtype=state_dtype,
            fuse_conv=True,
            fuse_out_norm=True,
            conv_once=True,
            waves_per_eu=4,
        )
    ).spec
    assert spec.waves_per_eu == 4, spec.kernel_name()
    out_err, state_err = check(spec, 16)
    assert out_err < TOL and state_err < TOL, (spec.kernel_name(), out_err, state_err)


@requires_gfx950
@pytest.mark.parametrize("state_dtype", ["bf16", "f32"])
def test_norm_with_gqa_heads(state_dtype):
    """fuse_out_norm alone works for Hv > Hk (the Qwen3-Next GDN shape)."""
    from builders.gfx950.gdn.gdn_decode import TOL, check
    from dispatch.gdn import dispatch_gdn_decode

    spec = dispatch_gdn_decode(
        _request(
            batch=8,
            num_k_heads=16,
            num_v_heads=32,
            state_dtype=state_dtype,
            fuse_out_norm=True,
        )
    ).spec
    out_err, state_err = check(spec, 8)
    assert out_err < TOL and state_err < TOL, (out_err, state_err)


@requires_gfx950
@pytest.mark.parametrize("gate", ["gdn", "kda"])
@pytest.mark.parametrize("num_warps", [1, 2, 4])
@pytest.mark.parametrize("conv_once", [False, True], ids=["lane", "once"])
def test_conv_in_place_slots(gate, num_warps, conv_once):
    """read slot == write slot: the in-place conv-state shift must stay exact
    with one wave (no barrier) and with several (barrier before the writes)."""
    from builders.gfx950.gdn.gdn_decode import (
        TOL,
        drain,
        launch,
        launcher_for,
        make_inputs,
        out_error,
        prepare,
        ref_conv_state_after,
        ref_fp32,
    )
    from kernels.gfx950.gdn_decode import GdnDecodeSpec

    spec = dc.replace(
        GdnDecodeSpec(),
        num_k_heads=16,
        num_v_heads=16,
        gate_kind=gate,
        num_warps=num_warps,
        warp_threads_k=16,
        blocks_per_v_dim=1,
        fuse_conv=True,
        fuse_out_norm=True,
        conv_once=conv_once,
    )
    batch = 8
    inp = make_inputs(spec, batch, pool_depth=batch, disjoint_writes=False)
    ref_out, ref_state = ref_fp32(spec, inp)
    ref_conv = ref_conv_state_after(spec, inp)
    values, cfg = prepare(spec, inp, batch)
    launch(launcher_for(spec), values, cfg)
    drain()
    slots = inp["write_indices"].long()
    assert out_error(spec, values["out"], ref_out) < TOL
    assert (values["state"].float()[slots] - ref_state).abs().max().item() < TOL
    assert (values["conv_state"].float()[slots] - ref_conv).abs().max().item() < TOL
