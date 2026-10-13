# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU behavior checks for the rocKE-only KDA decode benchmark."""

from __future__ import annotations

import pytest

from benchmarks.gfx950.gdn import benchmark_kda_decode as bench


def test_parse_variants_accepts_known_names_and_deduplicates():
    assert bench.parse_variants("fused,precomputed,fused,simple") == (
        "fused",
        "precomputed",
        "simple",
    )


def test_parse_variants_rejects_unknown_and_empty():
    with pytest.raises(ValueError, match="unknown variant"):
        bench.parse_variants("fused,aiter")
    with pytest.raises(ValueError, match="at least one"):
        bench.parse_variants("")


def test_production_variants_derive_from_the_dispatched_tile():
    variants, result = bench.production_variants(
        batch=8,
        num_k_heads=32,
        num_v_heads=32,
        head_dim=128,
        names=("fused", "precomputed", "simple"),
    )

    assert result.candidate.spec_id == "kda_nw1_wtk16_bpv8"
    fused = variants["fused"]
    raw = variants["precomputed"]
    simple = variants["simple"]
    tile = (fused.num_warps, fused.warp_threads_k, fused.blocks_per_v_dim)
    assert tile == (1, 16, 8)
    assert (raw.num_warps, raw.warp_threads_k, raw.blocks_per_v_dim) == tile
    assert raw.fuse_gate is False
    assert simple.simple is True
    assert simple.gate_kind == "kda"


def test_default_run_fails_and_never_times_a_nan_state(monkeypatch):
    """A variant whose output is exact but whose written state holds a NaN
    fails the shared fp32 gate and is never timed."""
    torch = pytest.importorskip("torch")
    from builders.gfx950.gdn import gdn_decode as g

    real = g.make_inputs

    def nan_state(spec, inp, launcher, batch):
        ref = g.reference(spec, inp)
        written = inp["write_indices"].long()
        state = inp["state"].clone()
        state[written] = ref[1].to(state.dtype)
        state[written[0]].view(-1)[0] = float("nan")
        return {"out": ref[0].to(torch.bfloat16), "state": state}

    def timed(*args, **kw):
        raise AssertionError("a variant that failed the gate was timed")

    monkeypatch.setattr(
        bench, "make_inputs", lambda spec, batch: real(spec, batch, device="cpu")
    )
    monkeypatch.setattr(bench, "launcher_for", lambda spec, arch=None: object())
    monkeypatch.setattr(g, "run_values", nan_state)
    monkeypatch.setattr(bench, "eager_us", timed)
    failures = bench.run_default(
        batches=(1,),
        names=("fused", "precomputed"),
        num_k_heads=8,
        num_v_heads=8,
        head_dim=128,
        no_device=True,
    )
    assert len(failures.messages) == 2, failures.messages
    assert all("fp32 gate failed (worst nan" in m for m in failures.messages)


def _swept(tid, us, tile, status="ok", index=0):
    return {
        "tuning_id": tid,
        "us": us,
        "tile": list(tile),
        "status": status,
        "index": index,
        "reason": "",
        "out_err": 1e-3,
        "state_err": 1e-3,
        "conv_err": 0.0,
    }


def test_compare_selected_to_sweep_reports_best_and_ratio():
    rows = [
        _swept("auto", 5.0, (1, 16, 4), index=1),
        _swept("fast", 4.0, (2, 16, 1), index=0),
        _swept("bad", 1.0, (4, 16, 4), status="fail", index=2),
        _swept("slow", 6.0, (4, 16, 4), index=3),
    ]
    result = bench.compare_selected_to_sweep("auto", rows)

    assert result.selected_us == 5.0
    assert result.best_id == "fast"
    assert result.best_tile == (2, 16, 1)
    assert result.best_us == 4.0
    assert result.selected_over_best == 1.25


def test_compare_selected_to_sweep_fails_when_selected_missing_or_rows_empty():
    with pytest.raises(ValueError, match="no correct, timeable"):
        bench.compare_selected_to_sweep("auto", [])
    with pytest.raises(ValueError, match="selected configuration"):
        bench.compare_selected_to_sweep("auto", [_swept("fast", 4.0, (2, 16, 1))])
    # A selected row that failed the gate is not a timed row.
    with pytest.raises(ValueError, match="selected configuration"):
        bench.compare_selected_to_sweep(
            "auto",
            [_swept("fast", 4.0, (2, 16, 1)), _swept("auto", 1.0, (1, 16, 8), "fail")],
        )


def test_tile_sweep_runs_the_table_sweep_stage_one_and_fails_closed(
    monkeypatch, capsys
):
    """``--sweep-tiles`` times every tile's base configuration plus ``auto``
    through ``decode_table_sweep`` and fails on any configuration that did not
    pass and time."""
    calls = []

    def fake_stage1(shape, auto, configs, indices, opts, meta):
        calls.append((shape, list(indices), opts.bases_only, opts.rotate_mb))
        rows = [
            _swept(r.spec.tuning_id, 10.0 + i, (1, 1, 1), index=i)
            for i, r in enumerate(configs)
        ]
        rows[-1]["us"] = 9.0 if shape.batch == 1 else 30.0
        rows[0]["status"] = "fail" if shape.batch == 8 else "ok"
        return rows

    monkeypatch.setattr(bench.table_sweep, "stage1_shape", fake_stage1)
    failures = bench.run_sweep(
        batches=(1, 8),
        num_k_heads=32,
        num_v_heads=32,
        head_dim=128,
        top=2,
        device="gfx950",
        rotate_mb=0,
    )
    (s1, idx1, bases1, warm1), (s8, idx8, _, _) = calls
    assert (s1.gate_kind, s1.mode, s1.state_dtype, s1.batch, s8.batch) == (
        "kda",
        "plain",
        "bf16",
        1,
        8,
    )
    assert bases1 is True and warm1 == 0
    # Every tile's base configuration (54 at D128) plus the auto configuration.
    assert len(idx1) == 55
    out = capsys.readouterr().out
    auto_id = bench.dispatch_gdn_decode(s1.request()).spec.tuning_id
    assert f"auto={auto_id} 9.000us" in out and "auto/fastest=1.000" in out
    assert len(failures.messages) == 1 and "batch 8" in failures.messages[0]


def test_failure_accumulator_is_fail_closed():
    failures = bench.Failures()
    assert failures.exit_code == 0
    failures.add("device timing failed")
    assert failures.exit_code == 1
    assert failures.messages == ["device timing failed"]
