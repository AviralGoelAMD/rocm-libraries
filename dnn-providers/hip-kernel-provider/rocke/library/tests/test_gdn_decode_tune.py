# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU control-flow tests for the GDN/KDA decode tuner."""

from __future__ import annotations

from types import SimpleNamespace

from builders.gfx950.gdn import tune


def test_sweep_registry_batch_returns_empty_without_registry_results():
    assert tune.sweep_registry_batch(1, ()) == []


def test_main_fails_when_any_requested_registry_cell_is_missing(monkeypatch, capsys):
    monkeypatch.setattr(tune, "device_is_visible", lambda: True)

    def fake_results(request):
        return () if request.num_k_heads == 16 and request.batch == 2 else (object(),)

    monkeypatch.setattr(tune, "dispatch_gdn_decode_all", fake_results)
    monkeypatch.setattr(
        tune,
        "sweep_registry_batch",
        lambda batch, results: [] if not results else [(1.0, (1, 8, 1), "test", 0.0)],
    )
    monkeypatch.setattr(
        tune,
        "dispatch_gdn_decode",
        lambda request: SimpleNamespace(candidate=SimpleNamespace(spec_id="test")),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "tune.py",
            "--geometries",
            "16/32,8/16",
            "--batches",
            "1,2",
            "--top",
            "1",
        ],
    )

    assert tune.main() == 1
    assert (
        "batch 2: no candidate was both correct and timeable" in capsys.readouterr().out
    )


def test_main_reports_dispatcher_default_outside_top_rows(monkeypatch, capsys):
    monkeypatch.setattr(tune, "device_is_visible", lambda: True)
    monkeypatch.setattr(
        tune,
        "dispatch_gdn_decode_all",
        lambda request: (object(), object(), object()),
    )
    monkeypatch.setattr(
        tune,
        "sweep_registry_batch",
        lambda batch, results: [
            (5.0, (4, 16, 8), "fast", 0.0),
            (6.1, (2, 16, 8), "default", 0.0),
            (6.4, (1, 8, 1), "other", 0.0),
        ],
    )
    monkeypatch.setattr(
        tune,
        "dispatch_gdn_decode",
        lambda request: SimpleNamespace(candidate=SimpleNamespace(spec_id="default")),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["tune.py", "--batches", "1", "--geometries", "16/32", "--top", "1"],
    )

    assert tune.main() == 0
    output = capsys.readouterr().out
    assert "dispatcher default: 6.100us  default tile=(2, 16, 8) rank=2/3" in output
    assert "fastest legal candidate: 5.000us  fast tile=(4, 16, 8)" in output
    assert "default / fastest = 1.220x" in output
    assert "consider DEFAULT_TILE = (4, 16, 8)" in output


def test_main_reports_missing_gdn_default_and_continues(monkeypatch, capsys):
    monkeypatch.setattr(tune, "device_is_visible", lambda: True)
    monkeypatch.setattr(
        tune, "dispatch_gdn_decode_all", lambda request: (object(), object())
    )
    # Batch 1: the default failed correctness, so it never reached the rows.
    rows = {
        1: [(5.0, (4, 16, 8), "fast", 0.0)],
        2: [(5.0, (2, 16, 8), "default", 0.0)],
    }
    monkeypatch.setattr(
        tune, "sweep_registry_batch", lambda batch, results: rows[batch]
    )
    monkeypatch.setattr(
        tune,
        "dispatch_gdn_decode",
        lambda request: SimpleNamespace(candidate=SimpleNamespace(spec_id="default")),
    )
    monkeypatch.setattr(
        "sys.argv", ["tune.py", "--batches", "1,2", "--geometries", "16/32"]
    )

    assert tune.main() == 1
    output = capsys.readouterr().out
    assert (
        "dispatcher default 'default' is NOT in the correct-and-timeable set" in output
    )
    assert "=== Hk16/Hv32 batch 2" in output
    assert "manual review: retain DEFAULT_TILE" in output


def test_report_dispatcher_default_keeps_fastest_default(capsys):
    tune.report_dispatcher_default(
        [(5.0, (2, 16, 8), "default", 0.0)], "default", "DEFAULT_TILE"
    )

    assert capsys.readouterr().out.splitlines() == [
        "  dispatcher default: 5.000us  default tile=(2, 16, 8) rank=1/1",
        "  fastest legal candidate: 5.000us  default tile=(2, 16, 8)",
        "  default / fastest = 1.000x",
        "  manual review: retain DEFAULT_TILE",
    ]


def test_main_sweeps_kda_registry_with_the_requested_state_dtype(monkeypatch, capsys):
    """KDA cells sweep the registry for the requested gate kind and state width,
    and name the default constant that state width ships."""
    seen = []
    monkeypatch.setattr(tune, "device_is_visible", lambda: True)

    def fake_results(request):
        seen.append((request.gate_kind, request.state_dtype))
        return (object(), object())

    monkeypatch.setattr(tune, "dispatch_gdn_decode_all", fake_results)
    monkeypatch.setattr(
        tune,
        "sweep_registry_batch",
        lambda batch, results: [
            (4.0, (4, 16, 8), "kda_nw4_wtk16_bpv8", 0.0),
            (5.0, (8, 16, 4), "kda_nw8_wtk16_bpv4", 0.0),
        ],
    )
    monkeypatch.setattr(
        tune,
        "dispatch_gdn_decode",
        lambda request: SimpleNamespace(
            candidate=SimpleNamespace(spec_id="kda_nw8_wtk16_bpv4")
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "tune.py",
            "--gate-kind",
            "kda",
            "--state-dtype",
            "f32",
            "--batches",
            "1,8",
        ],
    )

    assert tune.main() == 0
    assert seen == [("kda", "f32"), ("kda", "f32")]
    output = capsys.readouterr().out
    assert output.count("consider KDA_DEFAULT_TILE_F32 = (4, 16, 8)") == 2


def test_main_sweeps_fused_registry_and_names_the_fused_default(monkeypatch, capsys):
    """--fuse-conv / --fuse-out-norm reach every swept request, and the report
    names the fused default entry for that gate kind and state width."""
    seen = []
    monkeypatch.setattr(tune, "device_is_visible", lambda: True)

    def fake_results(request):
        seen.append((request.gate_kind, request.fuse_conv, request.fuse_out_norm))
        return (object(),)

    monkeypatch.setattr(tune, "dispatch_gdn_decode_all", fake_results)
    monkeypatch.setattr(
        tune,
        "sweep_registry_batch",
        lambda batch, results: [
            (4.0, (2, 16, 1), "kda_nw2_wtk16_bpv1", 0.0),
            (5.0, (4, 16, 1), "kda_nw4_wtk16_bpv1", 0.0),
        ],
    )
    monkeypatch.setattr(
        tune,
        "dispatch_gdn_decode",
        lambda request: SimpleNamespace(
            candidate=SimpleNamespace(spec_id="kda_nw4_wtk16_bpv1")
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "tune.py",
            "--gate-kind",
            "kda",
            "--geometries",
            "16/16",
            "--batches",
            "1",
            "--fuse-conv",
            "--fuse-out-norm",
        ],
    )

    assert tune.main() == 0
    assert seen == [("kda", True, True)]
    output = capsys.readouterr().out
    assert "consider FUSED_DEFAULT_TILES[('kda', 'bf16')] = (2, 16, 1)" in output
