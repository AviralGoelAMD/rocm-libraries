# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-device numerics of every performance knob (ALGORITHM.md §4.10).

Each knob in ``KNOBS`` is switched on alone, on every tile of every class it
applies to: gate kind (gdn / kda) x plain / fused x state dtype (bf16 / f32),
at the default heads (DK = DV = 128). A plain class runs every registered tile
(``CONFIGURED_TILES``) and the reference path; a fused class runs every BPV=1
tile of the same warp and K-lane counts. A knob is skipped on a tile where its
``*_applies`` predicate is False or the validator rejects the spec: there it
emits the knob-off kernel, which the host test ``test_gdn_decode_knobs.py``
pins byte for byte. Every knob in ``KNOBS`` also runs at DK = DV = 64 and 256,
and ``out_lds`` at DV = 256 and 512 and with two store rounds. ``row_loads``,
which only applies to single-wave ``fuse_conv`` tiles, has its own tests: each
forced order on every such BPV=1 tile at DK = DV = 128, and on a set of them at
DK = DV = 64 and 256 and with an f16 state. All twelve knobs also run at once.

``check`` scores the output, the written recurrent state, the written conv
state, and every untouched pool page against the fp32 reference, so a knob
that misplaces, drops or races a write fails. The knobs documented as running
the same floating-point operations are also compared bit for bit with the
knob-off kernel on identical inputs.

Needs a real gfx950 and ROCm torch; marked ``gpu`` and skipped elsewhere.
"""

from __future__ import annotations

import dataclasses as dc
from itertools import product

import pytest

ARCH = "gfx950"

torch = pytest.importorskip("torch", reason="ROCm torch required")

pytestmark = pytest.mark.gpu


def _device_is_gfx950() -> bool:
    # rocke's own query, as the sibling GDN suites do: get_device_arch goes
    # through hipDeviceGetAttribute and strips the feature flags.
    if not torch.cuda.is_available():
        return False
    try:
        from rocke.runtime.hip_module import get_device_arch

        return get_device_arch() == ARCH
    except Exception:
        return False


requires_gfx950 = pytest.mark.skipif(
    not _device_is_gfx950(), reason=f"needs a {ARCH} device"
)

# knob -> the spec overrides that switch it on (norm_gate_once needs conv_once)
KNOBS = {
    "state_load_hint": dict(state_load_hint="streaming"),
    "state_store_hint": dict(state_store_hint="streaming"),
    "dpp_reduce": dict(dpp_reduce=True),
    "xcd_remap": dict(xcd_remap=True),
    "stream_rows": dict(stream_rows=True),
    "interleave_cols": dict(interleave_cols=True),
    "conv_once": dict(conv_once=True),
    "norm_gate_once": dict(conv_once=True, norm_gate_once=True),
    "out_lds": dict(out_lds=True),
    "waves_per_eu": dict(waves_per_eu=4),
    "kernarg_preload": dict(kernarg_preload=True),
}
_FUSED_HEADS = dict(num_k_heads=16, num_v_heads=16)


def _plain_tiles():
    """Every registered tile, then the reference path."""
    from dispatch.gdn.gfx950 import CONFIGURED_TILES

    tiles = [
        dict(num_warps=nw, warp_threads_k=wtk, blocks_per_v_dim=bpv)
        for nw, wtk, bpv in CONFIGURED_TILES
    ]
    return tiles + [dict(simple=True)]


def _fused_tiles():
    """Every BPV=1 tile of the registered warp and K-lane counts."""
    from dispatch.gdn.gfx950 import NUM_WARPS, WARP_THREADS_K

    return [
        dict(num_warps=nw, warp_threads_k=wtk, blocks_per_v_dim=1)
        for nw, wtk in product(NUM_WARPS, WARP_THREADS_K)
    ]


def _applies(spec, knob) -> bool:
    from kernels.gfx950 import gdn_decode as g

    pred = {
        "dpp_reduce": g.dpp_reduce_applies,
        "xcd_remap": g.xcd_remap_applies,
        "stream_rows": g.stream_rows_applies,
        "interleave_cols": g.interleave_cols_applies,
    }.get(knob)
    return pred is None or pred(spec)


def _knob_specs(knob, gate, fused, state_dtype):
    """Every legal knob-on spec of one class where the knob changes code."""
    from kernels.gfx950.gdn_decode import GdnDecodeSpec, is_valid_spec

    base = dict(gate_kind=gate, state_dtype=state_dtype, **KNOBS[knob])
    if fused:
        heads = _FUSED_HEADS
        tiles = _fused_tiles()
        # conv only, norm only, both: each fused knob runs on every mode it
        # accepts.
        modes = [(True, False), (False, True), (True, True)]
    else:
        heads = {}
        tiles = _plain_tiles()
        modes = [(False, False)]
    specs = []
    for tile, (conv, norm) in product(tiles, modes):
        spec = dc.replace(
            GdnDecodeSpec(),
            **heads,
            **base,
            **tile,
            fuse_conv=conv,
            fuse_out_norm=norm,
        )
        if is_valid_spec(spec, arch=ARCH)[0] and _applies(spec, knob):
            specs.append(spec)
    return specs


def _assert_exact(spec, batch):
    from builders.gfx950.gdn.gdn_decode import check

    verdict = check(spec, batch)
    assert verdict.passed, (spec.kernel_name(), batch, verdict)


@requires_gfx950
@pytest.mark.parametrize("knob", sorted(KNOBS))
@pytest.mark.parametrize("gate", ["gdn", "kda"])
@pytest.mark.parametrize("fused", [False, True], ids=["plain", "fused"])
@pytest.mark.parametrize("state_dtype", ["bf16", "f32"])
def test_knob_on_every_tile_it_applies_to(knob, gate, fused, state_dtype):
    specs = _knob_specs(knob, gate, fused, state_dtype)
    if not specs:
        # Only the fused-only knobs on a plain class and interleave_cols on a
        # 2-byte state have no tile to change.
        assert (not fused and knob in ("conv_once", "norm_gate_once", "out_lds")) or (
            knob == "interleave_cols" and state_dtype != "f32"
        ), (knob, gate, fused, state_dtype)
        pytest.skip(f"{knob} changes no code in this class")
    for spec in specs:
        _assert_exact(spec, 3)


@requires_gfx950
@pytest.mark.parametrize("fused", [False, True], ids=["plain", "fused"])
@pytest.mark.parametrize("batch", [1, 3, 16])
def test_xcd_remap_on_a_grid_that_is_not_a_multiple_of_8(fused, batch):
    """Five heads at BPV=1: grids of 5, 15 and 80 workgroups. The remap must
    be a bijection, or a head is computed twice and another never."""
    from kernels.gfx950.gdn_decode import GdnDecodeSpec, is_valid_spec

    spec = dc.replace(
        GdnDecodeSpec(),
        num_k_heads=5,
        num_v_heads=5,
        num_warps=2,
        warp_threads_k=16,
        blocks_per_v_dim=1,
        fuse_conv=fused,
        fuse_out_norm=fused,
        xcd_remap=True,
    )
    assert is_valid_spec(spec, arch=ARCH)[0], is_valid_spec(spec, arch=ARCH)[1]
    _assert_exact(spec, batch)


@requires_gfx950
@pytest.mark.parametrize("gate", ["gdn", "kda"])
@pytest.mark.parametrize("state_dtype", ["bf16", "f32"])
def test_every_knob_together(gate, state_dtype):
    """All twelve knobs on at once, at the fused (4,16,1) and (1,16,1) tiles
    (``row_loads`` forces ``"first"``, the order auto does not choose there;
    it changes code only on the one-wave tile)."""
    from kernels.gfx950.gdn_decode import GdnDecodeSpec

    for nw in (1, 4):
        knobs = {"row_loads": "first"}
        for kw in KNOBS.values():
            knobs.update(kw)
        spec = dc.replace(
            GdnDecodeSpec(),
            num_k_heads=16,
            num_v_heads=16,
            gate_kind=gate,
            state_dtype=state_dtype,
            num_warps=nw,
            warp_threads_k=16,
            blocks_per_v_dim=1,
            fuse_conv=True,
            fuse_out_norm=True,
            **knobs,
        )
        for batch in (1, 16):
            _assert_exact(spec, batch)


# The knobs ALGORITHM.md §4.10 says run the same floating-point operations on
# the same values. The fp32 tolerance above cannot tell a reordered sum from
# the original, so these are compared to the knob-off kernel bit for bit.
_BITWISE_KNOBS = [
    "state_load_hint",
    "state_store_hint",
    "dpp_reduce",
    "xcd_remap",
    "stream_rows",
    "conv_once",
    "out_lds",
    "waves_per_eu",
    "kernarg_preload",
]


def _assert_bitwise(spec, knob):
    """``spec`` and its knob-off twin give identical output, state and conv
    state on identical inputs."""
    default = spec.__class__()
    _assert_same_bits(
        dc.replace(spec, **{k: getattr(default, k) for k in KNOBS[knob]}), spec
    )


def _assert_same_bits(off, spec, batch=3, state_steps=0):
    """``off`` and ``spec`` give identical output, state and conv state on
    identical inputs; with ``state_steps``, the state may differ by up to that
    many 16-bit rounding steps per element."""
    from builders.gfx950.gdn.gdn_decode import (
        drain,
        launch,
        launcher_for,
        make_inputs,
        prepare,
    )

    inp = make_inputs(off, batch)
    arms = []
    for s in (off, spec):
        values, cfg = prepare(s, inp, batch)
        launch(launcher_for(s), values, cfg)
        drain()
        arms.append(values)
    for key in ("out", "state") + (("conv_state",) if spec.fuse_conv else ()):
        if key == "state" and state_steps:
            steps = (_ordered(arms[0][key]) - _ordered(arms[1][key])).abs().max()
            assert int(steps) <= state_steps, (spec.kernel_name(), key, int(steps))
            continue
        assert torch.equal(arms[0][key], arms[1][key]), (spec.kernel_name(), key)


def _ordered(t):
    """A 16-bit float tensor's bit patterns as integers in value order, so
    adjacent representable values differ by 1 (and +0 equals -0)."""
    bits = t.view(torch.int16).to(torch.int32)
    return torch.where(bits < 0, -(bits & 0x7FFF), bits)


@requires_gfx950
@pytest.mark.parametrize("knob", _BITWISE_KNOBS)
@pytest.mark.parametrize("gate", ["gdn", "kda"])
@pytest.mark.parametrize("state_dtype", ["bf16", "f32"])
def test_knob_output_is_bitwise_the_knob_off_output(knob, gate, state_dtype):
    specs = [
        s
        for fused in (False, True)
        for s in _knob_specs(knob, gate, fused, state_dtype)
    ]
    assert specs, (knob, gate, state_dtype)
    for spec in specs:
        _assert_bitwise(spec, knob)


@requires_gfx950
@pytest.mark.parametrize("gate", ["gdn", "kda"])
@pytest.mark.parametrize("num_warps", [1, 2, 4])
@pytest.mark.parametrize("fuse_out_norm", [False, True], ids=["cv", "cv_rn"])
def test_conv_once_in_place_slots(gate, num_warps, fuse_out_norm):
    """read slot == write slot: conv_once shifts each channel's taps from the
    thread that read them, with no barrier, so the in-place shift must stay
    exact with any number of waves."""
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
        fuse_out_norm=fuse_out_norm,
        conv_once=True,
        norm_gate_once=fuse_out_norm,
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


# (plain tile, fused tile) per head dim DK = DV: both have more than one state
# row per lane and WTK >= 8, so every tile-dependent knob changes code.
_HEAD_DIM_TILES = {
    64: ((2, 8, 2), (2, 8, 1)),
    256: ((2, 16, 8), (8, 16, 1)),
}


@requires_gfx950
@pytest.mark.parametrize("knob", sorted(KNOBS))
@pytest.mark.parametrize("head_dim", sorted(_HEAD_DIM_TILES))
@pytest.mark.parametrize("gate", ["gdn", "kda"])
@pytest.mark.parametrize("state_dtype", ["bf16", "f32"])
def test_knob_at_other_head_dims(knob, head_dim, gate, state_dtype):
    """Every knob at DK = DV = 64 and 256, warp-tiled and in every fused
    mode it accepts; the bit-for-bit knobs also against the knob-off kernel."""
    from kernels.gfx950.gdn_decode import GdnDecodeSpec, is_valid_spec

    (pw, pk, pb), (fw, fk, fb) = _HEAD_DIM_TILES[head_dim]
    base = dc.replace(
        GdnDecodeSpec(),
        head_k_dim=head_dim,
        head_v_dim=head_dim,
        gate_kind=gate,
        state_dtype=state_dtype,
        **KNOBS[knob],
    )
    cands = [dc.replace(base, num_warps=pw, warp_threads_k=pk, blocks_per_v_dim=pb)] + [
        dc.replace(
            base,
            **_FUSED_HEADS,
            num_warps=fw,
            warp_threads_k=fk,
            blocks_per_v_dim=fb,
            fuse_conv=conv,
            fuse_out_norm=norm,
        )
        for conv, norm in ((True, False), (False, True), (True, True))
    ]
    specs = [s for s in cands if is_valid_spec(s, arch=ARCH)[0] and _applies(s, knob)]
    assert specs or (knob == "interleave_cols" and state_dtype != "f32")
    for spec in specs:
        _assert_exact(spec, 3)
        if knob in _BITWISE_KNOBS:
            _assert_bitwise(spec, knob)


# out_lds row groups wider than the DK = DV = 128 tiles' 2 rows per storing
# lane: 4 rows (DV = 256), 8 rows (DV = 512), and DV / 8 = 72 storing lanes on
# a 64-lane workgroup, so the second store round is partly guarded off.
_WIDE_OUT_LDS = {
    "dv256": dict(num_k_heads=16, num_v_heads=16, head_v_dim=256, num_warps=8),
    "dv512": dict(num_k_heads=16, num_v_heads=16, head_v_dim=512, num_warps=16),
    "2rounds": dict(
        num_k_heads=4,
        num_v_heads=4,
        head_k_dim=8,
        head_v_dim=576,
        num_warps=1,
        warp_threads_k=1,
    ),
}


@requires_gfx950
@pytest.mark.parametrize("case", sorted(_WIDE_OUT_LDS))
@pytest.mark.parametrize(
    "mode",
    [
        dict(fuse_out_norm=True),
        dict(fuse_conv=True, fuse_out_norm=True),
        dict(fuse_conv=True, fuse_out_norm=True, conv_once=True, norm_gate_once=True),
    ],
    ids=["rn", "cv_rn", "cv_rn_ngo"],
)
@pytest.mark.parametrize("gate", ["gdn", "kda"])
@pytest.mark.parametrize("state_dtype", ["bf16", "f32"])
def test_out_lds_wide_row_groups(case, mode, gate, state_dtype):
    from kernels.gfx950.gdn_decode import GdnDecodeSpec, is_valid_spec

    spec = dc.replace(
        GdnDecodeSpec(),
        gate_kind=gate,
        state_dtype=state_dtype,
        blocks_per_v_dim=1,
        out_lds=True,
        **{"warp_threads_k": 16, **_WIDE_OUT_LDS[case]},
        **mode,
    )
    assert is_valid_spec(spec, arch=ARCH)[0], is_valid_spec(spec, arch=ARCH)[1]
    for batch in (1, 3):
        _assert_exact(spec, batch)
    _assert_bitwise(spec, "out_lds")


def _row_loads_pairs(gate, state_dtype, head_dim=128, dtype="bf16", wtks=None):
    """(auto spec, the same spec forced to the order auto does not choose) on
    every single-wave BPV=1 tile (or those with ``wtks`` K lanes), in each
    fused mode with fuse_conv: without conv_once, with it (auto loads first at
    DK = DV = 128, WTK 4 / 8), with it under a waves_per_eu floor, and with it
    and the register-heavy knobs."""
    from kernels.gfx950.gdn_decode import (
        GdnDecodeSpec,
        auto_late_row_inputs,
        is_valid_spec,
        row_loads_overrides,
    )

    heavy = dict(
        conv_once=True,
        stream_rows=True,
        dpp_reduce=True,
        interleave_cols=state_dtype == "f32",
        state_load_hint="streaming",
        state_store_hint="streaming",
    )
    variants = ({}, dict(conv_once=True), dict(conv_once=True, waves_per_eu=4), heavy)
    tiles = (
        [t for t in _fused_tiles() if t["num_warps"] == 1]
        if wtks is None
        else [dict(num_warps=1, warp_threads_k=w, blocks_per_v_dim=1) for w in wtks]
    )
    for tile in tiles:
        for norm, knobs in product((False, True), variants):
            auto = dc.replace(
                GdnDecodeSpec(),
                **_FUSED_HEADS,
                **tile,
                **knobs,
                head_k_dim=head_dim,
                head_v_dim=head_dim,
                dtype=dtype,
                gate_kind=gate,
                state_dtype=state_dtype,
                fuse_conv=True,
                fuse_out_norm=norm,
            )
            if not is_valid_spec(auto, arch=ARCH)[0]:
                continue
            order = "first" if auto_late_row_inputs(auto) else "late"
            forced = dc.replace(auto, row_loads=order)
            assert row_loads_overrides(forced), forced.kernel_name()
            yield auto, forced


@requires_gfx950
@pytest.mark.parametrize("gate", ["gdn", "kda"])
@pytest.mark.parametrize("state_dtype", ["bf16", "f32"])
def test_row_loads_on_every_single_wave_fused_tile(gate, state_dtype):
    """Each forced row_loads order against the fp32 reference, and bit for
    bit against the auto order it replaces: it moves loads, not math."""
    pairs = list(_row_loads_pairs(gate, state_dtype))
    # WTK 1, 2, 4, 8 and 16 (32 needs DK >= 256), two fused modes, four variants
    assert len(pairs) == 5 * 2 * 4, len(pairs)
    for auto, forced in pairs:
        _assert_exact(forced, 3)
        _assert_same_bits(auto, forced)


# case -> (DK = DV, I/O dtype, state dtype, K lanes of the single-wave tiles
# run, the forced orders that change code there). Auto loads late everywhere
# but at DK = DV = 128 with conv_once at WTK 4 / 8, so only the f16 case, at
# DK = DV = 128, forces "late" too.
_ROW_LOADS_OTHER = {
    "dk64": (64, "bf16", "bf16", (1, 8), {"first"}),
    "dk64_stf32": (64, "bf16", "f32", (1, 8), {"first"}),
    "dk256": (256, "bf16", "bf16", (1, 16, 32), {"first"}),
    "dk256_stf32": (256, "bf16", "f32", (1, 16, 32), {"first"}),
    "f16": (128, "f16", "f16", (4, 16), {"first", "late"}),
}


@requires_gfx950
@pytest.mark.parametrize("case", sorted(_ROW_LOADS_OTHER))
@pytest.mark.parametrize("gate", ["gdn", "kda"])
def test_row_loads_at_other_head_dims_and_an_f16_state(case, gate):
    """The forced orders the walk lists at DK = DV = 64 and 256 (WTK 32 only
    exists there) and with an f16 state, on a set of single-wave fused tiles:
    against the fp32 reference and bit for bit against the auto order. With
    an f16 state the output and conv state are bit for bit, but the compiler
    folds the final f32 -> f16 rounding of a state element into its FMA
    (v_fma_mix) for a different set of elements in each order, so the state
    may differ by one f16 rounding step."""
    head_dim, dtype, state_dtype, wtks, orders = _ROW_LOADS_OTHER[case]
    pairs = list(_row_loads_pairs(gate, state_dtype, head_dim, dtype, wtks))
    assert len(pairs) == len(wtks) * 2 * 4, len(pairs)
    assert {forced.row_loads for _auto, forced in pairs} == orders
    steps = 1 if state_dtype == "f16" else 0
    for auto, forced in pairs:
        _assert_exact(forced, 3)
        _assert_same_bits(auto, forced, state_steps=steps)


@requires_gfx950
def test_a_mistyped_row_loads_is_refused_after_auto_launched():
    """The launcher cache is keyed on the kernel name and read before the
    validator: a malformed ``row_loads`` must not find auto's launcher."""
    from builders.gfx950.gdn.gdn_decode import launcher_for

    auto, _forced = next(_row_loads_pairs("kda", "f32", wtks=(4,)))
    launcher_for(auto)
    for value in ("LATE", "late ", None, 1):
        with pytest.raises(ValueError, match="row_loads must be one of"):
            launcher_for(dc.replace(auto, row_loads=value))
