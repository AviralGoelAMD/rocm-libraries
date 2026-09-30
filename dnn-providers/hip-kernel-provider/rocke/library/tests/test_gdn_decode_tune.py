# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU control-flow tests for the GDN/KDA decode tuner."""

from __future__ import annotations

import dataclasses as dc
from itertools import product
from types import SimpleNamespace

from builders.gfx950.gdn import tune
import pytest

from dispatch.gdn.common import BLOCKS_PER_V_DIM, NUM_WARPS, WARP_THREADS_K
from kernels.common.gdn_decode import GdnDecodeSpec, is_valid_spec


@pytest.fixture
def device(monkeypatch):
    """A visible device of a chosen arch, without touching HIP."""

    def use(arch: str = "gfx950"):
        monkeypatch.setattr(tune, "device_is_visible", lambda: True)
        monkeypatch.setattr(tune, "device_arch", lambda: arch, raising=False)
        monkeypatch.setattr(tune, "describe_device", lambda: "test-device")

    return use


def test_legal_configs_reuses_registry_tile_space():
    assert tune.NUM_WARPS is NUM_WARPS
    assert tune.WARP_THREADS_K is WARP_THREADS_K
    assert tune.BLOCKS_PER_V_DIM is BLOCKS_PER_V_DIM

    base = dc.replace(GdnDecodeSpec(), gate_kind="kda", num_k_heads=16, num_v_heads=32)
    expected = [
        tile
        for tile in product(NUM_WARPS, WARP_THREADS_K, BLOCKS_PER_V_DIM)
        if is_valid_spec(
            dc.replace(
                base,
                num_warps=tile[0],
                warp_threads_k=tile[1],
                blocks_per_v_dim=tile[2],
            ),
            arch="gfx950",
        )[0]
    ]

    assert tune.legal_configs(base, "gfx950") == expected


def test_sweep_registry_batch_returns_empty_without_registry_results():
    assert tune.sweep_registry_batch(1, (), rotate_bytes=0) == []


def test_main_fails_when_any_requested_registry_cell_is_missing(
    monkeypatch, capsys, device
):
    device()

    def fake_results(request):
        return () if request.num_k_heads == 16 and request.batch == 2 else (object(),)

    monkeypatch.setattr(tune, "dispatch_gdn_decode_all", fake_results)
    monkeypatch.setattr(
        tune,
        "sweep_registry_batch",
        lambda batch, results, rotate_bytes: (
            [] if not results else [(1.0, (1, 8, 1), "test", 0.0)]
        ),
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


def test_main_reports_dispatcher_default_outside_top_rows(monkeypatch, capsys, device):
    device()
    monkeypatch.setattr(
        tune,
        "dispatch_gdn_decode_all",
        lambda request: (object(), object(), object()),
    )
    monkeypatch.setattr(
        tune,
        "sweep_registry_batch",
        lambda batch, results, rotate_bytes: [
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


def test_main_targets_the_device_arch_and_labels_it(monkeypatch, capsys, device):
    device("gfx942")
    requested = []

    def fake_results(request):
        requested.append(request.arch)
        return (object(),)

    monkeypatch.setattr(tune, "dispatch_gdn_decode_all", fake_results)
    monkeypatch.setattr(
        tune,
        "sweep_registry_batch",
        lambda batch, results, rotate_bytes: [(5.0, (2, 16, 8), "default", 0.0)],
    )
    monkeypatch.setattr(
        tune,
        "dispatch_gdn_decode",
        lambda request: SimpleNamespace(candidate=SimpleNamespace(spec_id="default")),
    )
    monkeypatch.setattr("sys.argv", ["tune.py", "--batches", "1"])

    assert tune.main() == 0
    assert requested == ["gfx942"]
    out = capsys.readouterr().out
    assert out.splitlines()[0] == (
        "# tune.py arch=gfx942 gate=gdn device=test-device "
        "cache=cold (>= 1024 MB per cycle)"
    )
    assert "=== gfx942 Hk16/Hv32 batch 1" in out


def test_main_refuses_an_arch_other_than_the_device(monkeypatch, capsys, device):
    device("gfx950")
    monkeypatch.setattr(
        tune, "dispatch_gdn_decode_all", lambda request: pytest.fail("must not sweep")
    )
    monkeypatch.setattr("sys.argv", ["tune.py", "--arch", "gfx942"])

    assert tune.main() == 2
    assert "does not match the device (gfx950)" in capsys.readouterr().err


def test_compile_result_targets_the_request_arch(monkeypatch):
    compiled = []
    monkeypatch.setattr(
        tune,
        "launcher_for",
        lambda spec, arch: compiled.append((spec, arch)),
        raising=False,
    )
    result = SimpleNamespace(spec="spec", request=SimpleNamespace(arch="gfx942"))

    tune.compile_result(result)

    assert compiled == [("spec", "gfx942")]


def test_static_tile_summary_ranks_by_geomean_over_common_cells(capsys):
    """Geomean of per-cell (tile / fastest); a tile absent from a cell is out."""
    cells = [
        [
            (1.0, (4, 16, 8), "a", 0.0),
            (1.1, (2, 16, 8), "d", 0.0),
            (2.0, (1, 1, 1), "x", 0.0),
        ],
        [(1.0, (2, 16, 8), "d", 0.0), (1.2, (4, 16, 8), "a", 0.0)],
    ]

    tune.report_static_tile_summary(cells, ["d", "d"], top=5)

    lines = capsys.readouterr().out.splitlines()
    # d: sqrt(1.1 * 1.0) = 1.049 beats a: sqrt(1.0 * 1.2) = 1.095; x is in one cell only.
    assert lines[2] == "   1.049x  d tile=(2, 16, 8) <- dispatcher default"
    assert lines[3] == "   1.095x  a tile=(4, 16, 8)"
    assert "x tile" not in "\n".join(lines)
    assert lines[-2] == "  dispatcher default d: rank 1/2 geomean 1.049x"
    assert lines[-1] == "  manual review: retain DEFAULT_TILE"


def test_main_rejects_kda_off_gfx950(monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["tune.py", "--gate-kind", "kda", "--arch", "gfx942"]
    )
    with pytest.raises(SystemExit) as exc:
        tune.main()
    assert exc.value.code == 2


def test_report_gdn_dispatcher_default_keeps_fastest_default(capsys):
    tune.report_gdn_dispatcher_default([(5.0, (2, 16, 8), "default", 0.0)], "default")

    assert capsys.readouterr().out.splitlines() == [
        "  dispatcher default: 5.000us  default tile=(2, 16, 8) rank=1/1",
        "  fastest legal candidate: 5.000us  default tile=(2, 16, 8)",
        "  default / fastest = 1.000x",
        "  manual review: retain DEFAULT_TILE",
    ]
