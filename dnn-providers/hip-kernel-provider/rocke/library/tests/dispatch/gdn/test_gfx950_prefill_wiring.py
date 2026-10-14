# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Dispatch-to-launch wiring for the gfx950 GDN prefill split path.

GDN prefill is the chunkwise KDA kernel in GDN mode: two split halves
(``chunk_prep`` + ``chunk_scan``), selected by an explicit algorithm pin, with
the scan's ``value_splits`` chosen from a batch-heads-banded tuning table. No
GPU here -- this asserts the request maps onto the right spec, tile and grid.
"""

from __future__ import annotations

import pytest

from dispatch.gdn import (
    GDN_PREFILL_REGISTRY,
    GdnPrefillRequest,
    dispatch_gdn_prefill,
    gdn_prefill_candidates,
)
from dispatch.gdn.prefill_gfx950 import value_splits_for

ARCH = "gfx950"


def _req(**kw) -> GdnPrefillRequest:
    base = dict(
        batch=8,
        seqlen=1024,
        arch=ARCH,
        num_k_heads=8,
        num_v_heads=8,  # BH = 8*8 = 64 by default
        head_k_dim=128,
        head_v_dim=128,
    )
    base.update(kw)
    return GdnPrefillRequest(**base)


def test_registry_has_exactly_the_two_split_halves():
    names = {c.name for c in gdn_prefill_candidates()}
    assert names == {
        "gdn_prefill_gfx950_chunk_prep",
        "gdn_prefill_gfx950_chunk_scan",
    }


def test_prep_carries_the_raw_gdn_gate():
    result = dispatch_gdn_prefill(_req(algorithm="chunk_prep"))
    assert result.candidate.name == "gdn_prefill_gfx950_chunk_prep"
    spec = result.spec
    assert spec.gate_kind == "gdn"
    assert spec.kv_group == 1  # num_v_heads == num_k_heads -> MHA
    assert spec.raw_inputs
    assert spec.fuse_gate
    assert spec.fuse_qk_l2norm
    assert spec.fuse_beta_sigmoid
    assert spec.has_dt_bias
    # Raw prep keeps the 256-thread builder regardless of the scan's split.
    assert spec.tile.block_size == 256


def test_prep_kv_group_tracks_gqa_ratio():
    result = dispatch_gdn_prefill(
        _req(num_k_heads=8, num_v_heads=32, algorithm="chunk_prep")
    )
    assert result.spec.kv_group == 4


def test_dispatched_prep_matches_the_builder_prep_spec():
    """The dispatcher hand-builds the raw GDN prep spec; pin it to the builder's
    canonical ``prep_spec_of(scan, raw=True)`` plus the GDN gate flags, so the
    two copies of that derivation cannot drift apart unnoticed (the dispatch
    layer cannot import the builder helper, hence a test rather than delegation).
    """
    import dataclasses

    pytest.importorskip("torch", reason="builder module imports torch")
    from builders.gfx950.kda.kda_chunk_split import prep_spec_of
    from dispatch.gdn.prefill_gfx950 import _prep_spec, _scan_spec

    for num_v_heads in (8, 16, 32):  # kv_group 1, 2, 4
        req = _req(num_v_heads=num_v_heads, algorithm="chunk_prep")
        scan = _scan_spec(req)
        expected = dataclasses.replace(
            prep_spec_of(scan, raw=True),
            gate_kind="gdn",
            kv_group=int(req.kv_group),
        )
        assert _prep_spec(req) == expected, f"prep drift at kv_group={req.kv_group}"


@pytest.mark.parametrize(
    "num_v_heads,batch_heads",
    [
        (2, 16),  # BH=16  -> vs=8 band
        (8, 64),  # BH=64  -> vs=4 band
        (16, 128),  # BH=128 -> vs=2 band
        (32, 256),  # BH=256 -> past the bands (default)
    ],
)
def test_scan_value_splits_table(num_v_heads, batch_heads):
    """Each probe batch lands in a different band; the expected split and its
    block size are read from the tuning table, not re-typed here, so a retune of
    the table cannot silently drift this test."""
    from dispatch.gdn.prefill_gfx950 import _SPLIT_TILE, value_splits_for

    expected_vs = value_splits_for(batch_heads)
    expected_block = _SPLIT_TILE[expected_vs]["block_size"]
    result = dispatch_gdn_prefill(
        _req(num_k_heads=2, num_v_heads=num_v_heads, algorithm="chunk_scan")
    )
    assert result.candidate.name == "gdn_prefill_gfx950_chunk_scan"
    assert result.request.batch_heads == batch_heads
    spec = result.spec
    assert spec.value_splits == expected_vs
    assert spec.tile.block_size == expected_block
    assert spec.token_major_io


def test_split_bands_use_m16_atom():
    for num_v_heads, vs in ((4, 8), (8, 4), (16, 2)):
        spec = dispatch_gdn_prefill(
            _req(num_k_heads=4, num_v_heads=num_v_heads, algorithm="chunk_scan")
        ).spec
        assert spec.value_splits == vs
        assert spec.tile.scan_atom_m == 16


def test_value_splits_bands():
    """Independent oracle for the tuned bands: deliberately hardcoded, *not*
    derived from ``_VALUE_SPLIT_BANDS``, so a wrong edit to that table is caught
    here instead of the test silently agreeing with it."""
    assert value_splits_for(1) == 8
    assert value_splits_for(32) == 8  # band edge
    assert value_splits_for(33) == 4  # just past -> next band
    assert value_splits_for(64) == 4
    assert value_splits_for(65) == 2
    assert value_splits_for(128) == 2
    assert value_splits_for(129) == 1  # past the last band -> default
    assert value_splits_for(4096) == 1


@pytest.mark.parametrize(
    "num_v_heads,batch_heads,value_splits,wave_split,xcd",
    [
        (1, 8, 8, 2, 8),  # TP=4 production shard
        (4, 32, 8, 2, 8),
        (8, 64, 4, 2, 8),
        (16, 128, 2, 2, 8),
        (32, 256, 1, 1, 0),
    ],
)
def test_scan_latency_hiding_knobs_per_band(
    num_v_heads, batch_heads, value_splits, wave_split, xcd
):
    """Every band stages tiles and V one chunk ahead; the split bands also split
    each state band across two waves and keep a head's splits on one XCD."""
    spec = dispatch_gdn_prefill(
        _req(num_k_heads=1, num_v_heads=num_v_heads, algorithm="chunk_scan")
    ).spec
    assert spec.value_splits == value_splits
    assert spec.prefetch_tiles and spec.v_prefetch
    assert spec.min_occupancy == 1
    assert spec.wave_split == wave_split
    assert spec.xcd_remap == xcd
    # Every knob must reach the cache key, or one build could be served from
    # another's entry.
    name = spec.kernel_name()
    assert "_pf" in name and "_vp" in name
    assert ("_ws2" in name) is (wave_split == 2)
    assert ("_x8" in name) is (xcd == 8)
    # The block covers (head_v / value_splits / atom_m) row bands, wave_split
    # waves each.
    bands = spec.head_v // value_splits // spec.scan_atom.m
    assert spec.tile.block_size == 64 * bands * wave_split


def test_dispatched_scan_is_always_lds_legal():
    """The prefetch gate must never hand the builder a spec it will reject."""
    from kernels.gfx950.kda_chunkwise import is_valid_scan_spec

    for num_v_heads in (1, 2, 4, 8, 16, 32, 128):
        for has_initial_state in (False, True):
            spec = dispatch_gdn_prefill(
                _req(
                    num_k_heads=1,
                    num_v_heads=num_v_heads,
                    algorithm="chunk_scan",
                    has_initial_state=has_initial_state,
                )
            ).spec
            ok, why = is_valid_scan_spec(spec, arch=ARCH)
            assert ok, f"BH={8 * num_v_heads} vs={spec.value_splits}: {why}"


def test_auto_has_no_fused_default():
    with pytest.raises(ValueError, match="no fused default"):
        dispatch_gdn_prefill(_req())  # algorithm/spec_id both "auto"


def test_spec_id_pin_bypasses_auto_guard_and_selects_scan():
    result = dispatch_gdn_prefill(_req(spec_id="gfx950_gdn_chunk_scan"))
    assert result.candidate.name == "gdn_prefill_gfx950_chunk_scan"


def test_seqlen_must_tile_chunk():
    with pytest.raises(ValueError):
        dispatch_gdn_prefill(_req(seqlen=1000, algorithm="chunk_scan"))


def test_gqa_ratio_must_divide():
    with pytest.raises(ValueError):
        dispatch_gdn_prefill(
            _req(num_k_heads=8, num_v_heads=12, algorithm="chunk_prep")
        )


def test_scan_is_refused_when_its_prep_cannot_build():
    """Half a split path is not a selectable answer.

    The scan reads tiles the prep writes, and the two halves validate on
    different axes: ``head_k=32`` satisfies the scan's state-partition rules but
    fails the prep, whose fused L2 normalization reduces a fixed 128-element
    row. Admitting the scan alone would defer that failure to launch time, past
    the gate whose job is to name the reason -- so the refusal must happen here,
    and must say which half is at fault.
    """
    with pytest.raises(ValueError, match="tile builder for this scan is unbuildable"):
        dispatch_gdn_prefill(
            _req(head_k_dim=32, head_v_dim=128, algorithm="chunk_scan")
        )


def test_bf16_only():
    with pytest.raises(ValueError):
        dispatch_gdn_prefill(_req(dtype="f16", algorithm="chunk_scan"))


def test_wrong_arch_rejected():
    with pytest.raises(ValueError):
        dispatch_gdn_prefill(_req(arch="gfx942", algorithm="chunk_scan"))


def test_launch_geometry_is_present_and_consistent():
    result = dispatch_gdn_prefill(_req(algorithm="chunk_scan"))
    assert len(result.grid) == 3 and all(x > 0 for x in result.grid)
    assert result.block[0] == result.spec.tile.block_size
    assert len(result.signature) > 0


def test_prep_grid_scales_with_chunks():
    result = dispatch_gdn_prefill(_req(seqlen=1024, algorithm="chunk_prep"))
    # BH * num_chunks workgroups; num_chunks = 1024 / 32 = 32, BH = 64.
    assert result.request.num_chunks == 32
    assert len(result.grid) == 3 and all(x > 0 for x in result.grid)


def test_float_chunk_size_is_rejected():
    """int(32.7) used to dispatch as chunk 32 -- a chunk the caller never asked
    for, delivered successfully. Reject at the boundary, never normalize."""
    with pytest.raises(ValueError, match="chunk_size must be an int"):
        dispatch_gdn_prefill(_req(chunk_size=32.7, algorithm="chunk_prep"))


def test_float_batch_is_rejected():
    with pytest.raises(ValueError, match="batch must be an int"):
        dispatch_gdn_prefill(_req(batch=3.9, algorithm="chunk_prep"))


def test_retuning_onto_an_unlisted_split_fails_by_name():
    """A band selecting a split with no tile entry must say what to do, rather
    than raise a bare ``KeyError`` that reads as a dispatcher bug."""
    from dispatch.gdn import prefill_gfx950 as pg

    with pytest.raises(ValueError, match="no _SPLIT_TILE entry"):
        pg._scan_tile(_req(algorithm="chunk_scan"), 16)
