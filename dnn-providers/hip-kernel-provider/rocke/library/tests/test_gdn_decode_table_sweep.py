# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host tests of the gfx950 GDN/KDA decode table sweep (no GPU).

The GPU stages run against a fake driver whose timers return chosen samples,
so selection, ordering and aggregation are checked exactly.
"""

from __future__ import annotations

import itertools
import json
import os
from types import SimpleNamespace

import pytest

from benchmarks.gfx950.gdn import decode_table_sweep as sweep
from dispatch.gdn import dispatch_gdn_decode

SHAPE = sweep.Shape("gdn", "plain", "bf16", 16, 32, 128, 1)


def _parse(*argv):
    ap = sweep.build_parser()
    args = ap.parse_args(list(argv))
    sweep.check_args(ap, args)
    return args


# ---------------------------------------------------------------- CLI
def test_cli_defaults_and_value_parsers():
    args = _parse()
    assert (args.rounds, args.s2_rounds, args.runs, args.top_k) == (5, 30, 3, 8)
    assert args.rotate_mb is None  # resolved to the driver's ROTATE_MB at run time
    assert args.sweep_level == "production" and args.shard == (0, 1)
    args = _parse(
        "--heads",
        "16/32,4/8",
        "--batches",
        "1,8",
        "--shard",
        "2/4",
        "--stage1-only",
        "--tuning-ids",
        "a,b",
    )
    assert args.heads == ((16, 32), (4, 8))
    assert args.batches == (1, 8)
    assert args.shard == (2, 4)
    assert args.tuning_ids == ("a", "b")
    for bad in (("--shard", "4/4"), ("--heads", "16"), ("--batches", ",")):
        with pytest.raises(SystemExit):
            _parse(*bad)


def test_cli_refuses_a_partial_stage_two():
    """A window sees part of each shape, so ranking it for stage 2 is refused."""
    for window in (("--shard", "1/2"), ("--offset", "3"), ("--limit", "5")):
        with pytest.raises(SystemExit):
            _parse(*window)
        _parse(*window, "--stage1-only")
        _parse(*window, "--list-only")
    with pytest.raises(SystemExit):
        _parse("--stage1-json", "x.json", "--stage1-only")
    for walk in (("--bases-only",), ("--tuning-ids", "a"), ("--sweep-level", "full")):
        with pytest.raises(SystemExit):
            _parse("--stage1-json", "x.json", *walk)
    with pytest.raises(SystemExit):
        _parse("--runs", "0")


def test_cli_grid_and_shape_file_filters(tmp_path):
    args = _parse(
        "--gate-kind",
        "kda",
        "--mode",
        "fused",
        "--state-dtype",
        "f32",
        "--heads",
        "8/8,32/32",
        "--batches",
        "1,16",
    )
    shapes = sweep.select_shapes(args)
    assert [(s.num_k_heads, s.batch) for s in shapes] == [
        (8, 1),
        (8, 16),
        (32, 1),
        (32, 16),
    ]
    assert {s.case for s in shapes} == {"kda_fused_f32"}

    path = tmp_path / "shapes.json"
    path.write_text(
        json.dumps(
            [
                {
                    "gate_kind": "gdn",
                    "mode": "plain",
                    "state_dtype": "f16",
                    "heads": ["4/8", "16/32"],
                    "batches": [1, 2],
                },
                {
                    "gate_kind": "kda",
                    "mode": "plain",
                    "state_dtype": "bf16",
                    "num_k_heads": 8,
                    "num_v_heads": 8,
                    "batch": 4,
                    "head_dim": 64,
                },
            ]
        )
    )
    every = sweep.select_shapes(_parse("--shapes", str(path)))
    assert len(every) == 5
    # f16 state reads the bf16 AUTO_DEFAULTS row.
    assert every[0].case == "gdn_plain_bf16" and every[0].state_dtype == "f16"
    narrowed = sweep.select_shapes(
        _parse("--shapes", str(path), "--heads", "16/32", "--batches", "2")
    )
    assert narrowed == [sweep.Shape("gdn", "plain", "f16", 16, 32, 128, 2)]
    assert sweep.select_shapes(_parse("--shapes", str(path), "--head-dim", "64")) == [
        sweep.Shape("kda", "plain", "bf16", 8, 8, 64, 4)
    ]
    path.write_text(
        json.dumps(
            [
                {
                    "gate_kind": "gdn",
                    "mode": "plain",
                    "state_dtype": "bf16",
                    "batch": 1,
                    "heds": ["1/1"],
                }
            ]
        )
    )
    with pytest.raises(ValueError, match="unknown --shapes keys"):
        sweep.select_shapes(_parse("--shapes", str(path)))
    with pytest.raises(ValueError, match="mode"):
        sweep.Shape("gdn", "conv", "bf16", 16, 32, 128, 1)


def test_shipped_shape_grid_covers_every_case_and_dispatches():
    path = os.path.join(os.path.dirname(sweep.__file__), "decode_shapes.json")
    shapes = sweep.load_shapes(path)
    by_case: dict = {}
    for shape in shapes:
        by_case.setdefault(shape.case, []).append(shape)
    assert sorted(by_case) == sorted(
        f"{g}_{m}_{s}"
        for g in ("gdn", "kda")
        for m in ("plain", "fused")
        for s in ("bf16", "f32")
    )
    assert all(len(v) == 21 for v in by_case.values())
    for shape in shapes:
        assert dispatch_gdn_decode(shape.request()).spec.tuning_id


# ---------------------------------------------------------------- walk
def test_window_shards_are_disjoint_and_cover_the_window():
    for count, offset, limit, n in itertools.product(
        (0, 1, 7, 54), (0, 3), (0, 5), (1, 2, 3, 8)
    ):
        expected = list(
            range(offset, count if not limit else min(count, offset + limit))
        )
        shards = [sweep.window(count, (i, n), offset, limit) for i in range(n)]
        flat = [i for s in shards for i in s]
        assert sorted(flat) == expected
        assert len(flat) == len(set(flat))


def test_shape_configs_always_carry_auto_once():
    args = sweep.options(bases_only=True)
    auto, configs = sweep.shape_configs(SHAPE, args)
    ids = [r.spec.tuning_id for r in configs]
    assert all(not r.spec.knobs for r in configs[:-1])
    # auto runs a knob stack, so --bases-only appends it, once, at the end.
    assert ids[-1] == auto.spec.tuning_id and ids.count(auto.spec.tuning_id) == 1
    assert len(configs) == 54 + 1
    full_auto, full = sweep.shape_configs(SHAPE, sweep.options())
    assert [r.spec.tuning_id for r in full].count(full_auto.spec.tuning_id) == 1
    picked = sweep.shape_configs(SHAPE, sweep.options(tuning_ids=(ids[3], ids[5])))[1]
    assert [r.spec.tuning_id for r in picked] == [ids[3], ids[5], auto.spec.tuning_id]
    with pytest.raises(TypeError, match="unknown sweep options"):
        sweep.options(rotate=0)


def test_pinned_result_replays_a_recorded_row_and_refuses_another():
    auto, configs = sweep.shape_configs(SHAPE, sweep.options())
    result = configs[7]
    row = sweep.config_row(SHAPE, result, 7, auto.spec.tuning_id, {})
    row = json.loads(json.dumps(row))  # as read back from a file
    assert sweep.pinned_result(SHAPE, row).spec == result.spec
    other = sweep.Shape("gdn", "plain", "bf16", 16, 32, 64, 1)
    with pytest.raises(Exception):
        sweep.pinned_result(other, row)


# ---------------------------------------------------------------- stage 2
def test_interleaved_order_alternates_within_and_across_runs():
    seen = []
    samplers = [lambda i=i: seen.append(i) or float(i) for i in range(4)]
    samples = sweep.interleaved_samples(samplers, rounds=3, run=0)
    assert seen == [0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3]
    assert samples == [[0.0] * 3, [1.0] * 3, [2.0] * 3, [3.0] * 3]
    assert sweep.run_order(4, 1) == [3, 2, 1, 0]
    assert sweep.run_order(5, 2) == [2, 3, 4, 0, 1]
    assert sweep.run_order(4, 3) == [0, 1, 2, 3]
    seen.clear()
    sweep.interleaved_samples(samplers, rounds=2, run=2)
    assert seen == [2, 3, 0, 1, 1, 0, 3, 2]


def _row(index, us, status="ok", auto=False):
    return {
        "index": index,
        "us": us,
        "status": status,
        "auto": auto,
        "tuning_id": f"c{index}",
    }


def test_stage2_picks_top_k_then_auto_whatever_its_status():
    rows = [
        _row(0, 5.0),
        _row(1, 3.0),
        _row(2, 3.0),
        _row(3, 1.0, "fail"),
        _row(4, 9.0, auto=True),
        _row(5, 4.0),
    ]
    assert [r["index"] for r in sweep.stage2_picks(rows, 3)] == [1, 2, 5, 4]
    assert [r["index"] for r in sweep.stage2_picks(rows, 6)] == [1, 2, 5, 0, 4]
    rows[4]["status"] = "fail"
    assert [r["index"] for r in sweep.stage2_picks(rows, 1)] == [1, 4]


class _FakeCold:
    def __init__(self, samples, copies):
        self._samples = samples
        self.copies = copies
        self.closed = False

    def replay_us(self):
        return next(self._samples)

    def close(self):
        self.closed = True


class _CaptureUnavailable(RuntimeError):
    pass


class _FakeDriver:
    """The builder driver's sweep surface, with chosen timer samples. It
    checks that the gate sees disjoint read / write slots and timing the
    in-place ones."""

    CaptureUnavailable = _CaptureUnavailable

    def __init__(self, samples, *, failing=(), raising=(), no_graph=()):
        # tuning_id -> samples, consumed in order across every capture
        self.samples = {tid: iter(values) for tid, values in samples.items()}
        self.failing, self.raising, self.no_graph = failing, raising, no_graph
        self.ids = {}
        self.colds = []
        self.value_sets_made = 0

    def name(self, spec):
        return self.ids[spec]

    def make_inputs(self, spec, batch, seed, disjoint_writes):
        return {"disjoint": disjoint_writes}

    def reference(self, spec, inp):
        assert inp["disjoint"], "the reference must score the gate's inputs"
        return None

    def touched_bytes(self, spec, inp, batch):
        assert not inp["disjoint"], "the rotation is sized on the timed inputs"
        return 2**20

    def launcher_for(self, spec, arch):
        if self.name(spec) in self.raising:
            raise RuntimeError("compile failed")
        return spec

    def run_values(self, spec, inp, launcher, batch):
        assert inp["disjoint"], "the gate must run on disjoint read/write slots"
        return {}

    def gate(self, spec, inp, values, ref):
        assert inp["disjoint"]
        ok = self.name(spec) not in self.failing
        return SimpleNamespace(
            out_err=0.0 if ok else 1.0,
            state_err=0.0,
            conv_err=0.0,
            spill=False,
            passed=ok,
            worst=0.0 if ok else 1.0,
        )

    def cold_value_sets(self, spec, inp, batch, rotate_mb):
        assert not inp["disjoint"], "timing runs on the in-place slots"
        self.value_sets_made += 1
        return [object()] * 4

    def capture_cold(self, spec, batch, sets, launcher):
        tid = self.name(spec)
        if tid in self.no_graph:
            raise _CaptureUnavailable("RuntimeError: operation not permitted")
        cold = _FakeCold(self.samples[tid], len(sets))
        self.colds.append(cold)
        return cold

    def drain(self):
        pass


def _fake(monkeypatch, configs, samples, **kw):
    driver = _FakeDriver(samples, **kw)
    driver.ids = {r.spec.kernel_spec: r.spec.tuning_id for r in configs}
    monkeypatch.setattr(sweep, "_driver", lambda: driver)
    return driver


def test_stage1_records_the_median_and_every_failure(monkeypatch):
    args = sweep.options(bases_only=True, rotate_mb=1024, rounds=5)
    auto, configs = sweep.shape_configs(SHAPE, args)
    ids = [r.spec.tuning_id for r in configs]
    driver = _fake(
        monkeypatch,
        configs,
        {ids[0]: [5.0, 1.0, 100.0, 3.0, 4.0], ids[-1]: [2.0] * 5},
        failing={ids[1]},
        raising={ids[2]},
        no_graph={ids[3]},
    )
    rows = sweep.stage1_shape(
        SHAPE, auto, configs, [0, 1, 2, 3, len(ids) - 1], args, {"rotate_mb": 1024}
    )
    status = {r["tuning_id"]: r["status"] for r in rows}
    assert status == {
        ids[0]: "ok",
        ids[1]: "fail",
        ids[2]: "error",
        ids[3]: "no_graph",
        ids[-1]: "ok",
    }
    first = rows[0]
    assert first["us"] == 4.0 and first["runs"] == [4.0] and first["copies"] == 4
    kernel = configs[0].spec.kernel_spec
    assert first["stage"] == 1
    assert first["tile"] == [
        kernel.num_warps,
        kernel.warp_threads_k,
        kernel.blocks_per_v_dim,
    ]
    assert rows[-1]["auto"] and not first["auto"]
    assert "compile failed" in rows[2]["reason"]
    assert "operation not permitted" in rows[3]["reason"]  # the capture's cause
    assert rows[1]["us"] is None  # a gate failure is never timed
    assert all(c.closed for c in driver.colds)
    assert driver.value_sets_made == 1  # one set of buffers per shape


def test_stage2_retimes_interleaved_and_reports_the_median_of_run_medians(monkeypatch):
    args = sweep.options(rotate_mb=1024, s2_rounds=3, runs=3)
    auto, configs = sweep.shape_configs(SHAPE, args)
    a, b = configs[0], next(
        r for r in configs if r.spec.tuning_id == auto.spec.tuning_id
    )
    # Per run, 3 rounds each; run medians a: 2, 7, 4 -> 4; auto: 6, 1, 3 -> 3.
    samples = {
        a.spec.tuning_id: [1, 2, 9, 7, 7, 0, 4, 4, 5],
        b.spec.tuning_id: [6, 6, 8, 1, 2, 1, 3, 0, 3],
    }
    driver = _fake(monkeypatch, configs, samples)
    rows = sweep.stage2_shape(SHAPE, auto, [(0, a), (5, b)], args, {"rotate_mb": 1024})
    assert [r["runs"] for r in rows] == [[2, 7, 4], [6, 1, 3]]
    assert [r["us"] for r in rows] == [4, 3]
    assert [r["stage"] for r in rows] == [2, 2] and rows[1]["auto"]
    assert driver.value_sets_made == 3  # fresh buffers every run
    assert len(driver.colds) == 6 and all(c.closed for c in driver.colds)


# ---------------------------------------------------------------- files
def _rows():
    auto, configs = sweep.shape_configs(SHAPE, sweep.options())
    meta = {"rotate_mb": 1024, "sha": "abc", "device": "gfx950", "arch": "gfx950"}
    out = []
    for stage, index, us in ((1, 0, 3.0), (1, 1, 2.0), (2, 0, 2.5), (2, 1, 2.6)):
        row = sweep.config_row(
            SHAPE,
            configs[index],
            index,
            configs[0].spec.tuning_id,
            dict(meta, stage=stage),
        )
        row.update(status="ok", us=us, runs=[us])
        out.append(row)
    return out


def test_rows_round_trip_through_json_and_refuse_a_foreign_file(tmp_path):
    rows = _rows()
    path = tmp_path / "rows.json"
    sweep.write_rows(str(path), rows)
    assert sweep.load_rows([str(tmp_path / "*.json")]) == rows
    assert {sweep.shape_of(r) for r in rows} == {SHAPE}
    for key in (
        "tuning_id",
        "tile",
        "knobs",
        "us",
        "runs",
        "status",
        "sha",
        "arch",
        "out_err",
        "spill",
        *sweep.SHAPE_FIELDS,
    ):
        assert key in rows[0]
    (tmp_path / "other.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="not a"):
        sweep.load_rows([str(tmp_path / "other.json")])


def test_summarize_prefers_stage_two_and_fails_closed(capsys):
    rows = _rows()
    assert sweep.summarize(rows) == 0
    out = capsys.readouterr().out
    # Stage 2 decides: config 0 (auto here) is fastest there, unlike stage 1.
    line = next(l for l in out.splitlines() if l.startswith(SHAPE.label))
    assert rows[2]["tuning_id"] in line and "1.000" in line and " 2 " in line
    rows[3]["status"] = "fail"
    rows[3]["reason"] = "fp32 gate"
    assert sweep.summarize(rows) == 1
    for row in rows:
        row["auto"] = False
    capsys.readouterr()
    assert sweep.summarize(rows[:2]) == 1
    assert "auto configuration not timed" in capsys.readouterr().out
    # A windowed stage-1 run times auto in one window only.
    assert sweep.summarize(rows[:2], require_auto=False) == 0
    rows[0]["rotate_mb"] = 0
    sweep.summarize(rows)
    assert "WARNING" in capsys.readouterr().out


def test_stage2_from_files_splits_shapes_across_shards(monkeypatch, tmp_path):
    shapes = [sweep.Shape("gdn", "plain", "bf16", 16, 32, 128, b) for b in (1, 2, 3)]
    rows = []
    for shape in shapes:
        auto, configs = sweep.shape_configs(shape, sweep.options())
        for i in (0, 1):
            row = sweep.config_row(
                shape,
                configs[i],
                i,
                auto.spec.tuning_id,
                {"stage": 1, "rotate_mb": 1024},
            )
            row.update(status="ok", us=float(i + 1))
            rows.append(row)
    sweep.write_rows(str(tmp_path / "s1.json"), rows)
    seen = {}

    def fake_stage2(shape, auto, picks, args, meta):
        seen[shape] = [r.spec.tuning_id for _, r in picks]
        return []

    monkeypatch.setattr(sweep, "stage2_shape", fake_stage2)
    covered = []
    for i in range(2):
        seen.clear()
        args = sweep.options(
            stage1_json=[str(tmp_path / "s1.json")], shard=(i, 2), top_k=1
        )
        sweep.retime(args, {})
        covered.extend(seen)
        for shape, ids in seen.items():
            auto = dispatch_gdn_decode(shape.request()).spec.tuning_id
            assert ids[0] == rows[0]["tuning_id"] and ids[-1] == auto
    assert sorted(covered) == shapes


# ---------------------------------------------------------------- fail closed
def test_cli_refuses_flags_a_mode_would_accept_and_ignore():
    # --summarize reads saved rows only: no filter, output or walk/timing flag.
    for extra in (
        ("--gate-kind", "kda"),
        ("--batches", "16"),
        ("--output-json", "x.json"),
        ("--bases-only",),
        ("--top-k", "2"),
        ("--rotate-mb", "0"),
    ):
        with pytest.raises(SystemExit):
            _parse("--summarize", "x.json", *extra)
    _parse("--summarize", "x.json", "y.json")
    # --stage1-only runs no stage 2.
    for extra in (("--top-k", "2"), ("--s2-rounds", "4"), ("--runs", "1")):
        with pytest.raises(SystemExit):
            _parse("--stage1-only", *extra)
    _parse("--stage1-only", "--rounds", "3")
    with pytest.raises(SystemExit):
        _parse("--stage1-json", "x.json", "--list-only")
    # Out-of-domain values are refused, never replaced by a default.
    for bad in (("--head-dim", "0"), ("--head-dim", "-8"), ("--rotate-mb", "-1024")):
        with pytest.raises(SystemExit):
            _parse(*bad)


def test_shape_entries_refuse_conflicting_or_empty_keys():
    base = {"gate_kind": "gdn", "mode": "plain", "state_dtype": "bf16"}
    for bad, match in (
        (
            dict(heads=["16/32"], num_k_heads=4, num_v_heads=8, batches=[1]),
            "both heads",
        ),
        (dict(heads=["16/32"], batches=[1], batch=64), "both batches"),
        (dict(heads=["16/32"], batches=[]), "empty batches"),
        (dict(heads=[], batches=[1]), "empty heads"),
    ):
        with pytest.raises(ValueError, match=match):
            sweep.expand_entry(dict(base, **bad))


def test_tuning_ids_the_walk_does_not_list_are_an_error(capsys):
    production = sweep.shape_configs(SHAPE, sweep.options())[1]
    listed = {r.spec.tuning_id for r in production}
    real = production[3].spec.tuning_id
    with pytest.raises(ValueError, match="typo_id") as err:
        sweep.shape_configs(SHAPE, sweep.options(tuning_ids=(real, "typo_id")))
    assert "--sweep-level production" in str(err.value) and real not in str(err.value)
    # A full-level id is not in the production walk: refused there, found at full.
    full = sweep.shape_configs(SHAPE, sweep.options(sweep_level="full"))[1]
    extra = next(r.spec.tuning_id for r in full if r.spec.tuning_id not in listed)
    with pytest.raises(ValueError, match="production"):
        sweep.shape_configs(SHAPE, sweep.options(tuning_ids=(extra,)))
    found = sweep.shape_configs(
        SHAPE, sweep.options(sweep_level="full", tuning_ids=(extra,))
    )[1]
    assert found[0].spec.tuning_id == extra
    # The command line exits non-zero and names the id.
    assert sweep.main(["--batches", "1", "--list-only", "--tuning-ids", "typo_id"]) == 1
    assert "typo_id" in capsys.readouterr().out


def test_list_only_lists_both_row_load_orders_on_single_wave_fused_tiles(capsys):
    """``row_loads``: a single-wave fuse_conv tile lists each knob set at each
    waves_per_eu value in auto's order and in the other one ("late" with
    conv_once at WTK 4 / 8 and no waves_per_eu floor, where auto loads first;
    "first" everywhere else). Every other tile, and the plain mode, list one."""

    def listed(*argv):
        assert (
            sweep.main(["--batches", "1", "--heads", "8/8", *argv, "--list-only"]) == 0
        )
        by_set: dict = {}
        for line in capsys.readouterr().out.splitlines():
            fields = line.split(maxsplit=3)
            if len(fields) < 4 or not fields[0].isdigit():
                continue
            knobs = json.loads(fields[3].split("  <- auto")[0])
            order = knobs.pop("row_loads", "auto")
            key = (fields[1], tuple(sorted(knobs.items())))
            by_set.setdefault(key, set()).add(order)
        return by_set

    fused = listed("--gate-kind", "kda", "--mode", "fused", "--state-dtype", "f32")
    orders: dict = {}
    for (spec_id, knobs), got in fused.items():
        assert "auto" in got, (spec_id, knobs)
        orders.setdefault(spec_id, set()).update(got)
        if "_nw1_" not in f"_{spec_id}_":
            assert got == {"auto"}, (spec_id, knobs)
            continue
        k = dict(knobs)
        hoists = (
            k.get("conv_once")
            and k.get("waves_per_eu", 0) == 0
            and spec_id in ("kda_nw1_wtk4_bpv1", "kda_nw1_wtk8_bpv1")
        )
        assert got == {"auto", "late" if hoists else "first"}, (spec_id, knobs)
    assert orders["kda_nw1_wtk8_bpv1"] == {"auto", "first", "late"}
    assert orders["kda_nw1_wtk16_bpv1"] == {"auto", "first"}
    assert orders["kda_nw4_wtk8_bpv1"] == {"auto"}
    plain = listed("--gate-kind", "kda", "--mode", "plain", "--state-dtype", "f32")
    assert all(got == {"auto"} for got in plain.values())


def test_a_stage_one_failure_fails_the_run_after_stage_two(monkeypatch, tmp_path):
    """A configuration that fails the gate in stage 1 is never re-timed, so the
    stage-2 rows alone are clean: the summary must still fail, in one process
    and from saved files (``--summarize s1 s2``)."""
    ids = [r.spec.tuning_id for r in sweep.shape_configs(SHAPE, sweep.options())[1]]
    args = sweep.options(
        tuning_ids=(ids[0], ids[1]), rotate_mb=1024, rounds=1, top_k=1, runs=1
    )
    auto, configs = sweep.shape_configs(SHAPE, args)
    _fake(
        monkeypatch,
        configs,
        {r.spec.tuning_id: [1.0] * 64 for r in configs},
        failing={ids[1]},
    )
    rows = sweep.sweep(args, [SHAPE], {"rotate_mb": 1024})
    stage1 = [r for r in rows if r["stage"] == 1]
    stage2 = [r for r in rows if r["stage"] == 2]
    assert {r["status"] for r in stage2} == {"ok"} and ids[1] not in {
        r["tuning_id"] for r in stage2
    }
    assert sweep.summarize(stage2) == 0
    assert sweep.summarize(rows) == 1
    s1, s2 = str(tmp_path / "s1.json"), str(tmp_path / "s2.json")
    sweep.write_rows(s1, stage1)
    sweep.write_rows(s2, stage2)
    assert sweep.main(["--summarize", s1, s2]) == 1


def test_stage2_from_files_fails_on_a_shape_with_no_timed_row(monkeypatch, tmp_path):
    auto, configs = sweep.shape_configs(SHAPE, sweep.options())
    rows = []
    for i in (0, 1):
        row = sweep.config_row(
            SHAPE, configs[i], i, auto.spec.tuning_id, {"stage": 1, "rotate_mb": 1024}
        )
        row.update(status="fail", reason="fp32 gate")
        rows.append(row)
    path = str(tmp_path / "s1.json")
    sweep.write_rows(path, rows)

    def no_stage2(*args):
        raise AssertionError("a shape with no timed row was re-timed")

    monkeypatch.setattr(sweep, "stage2_shape", no_stage2)
    out = sweep.retime(sweep.options(stage1_json=[path]), {"rotate_mb": 1024})
    assert [(r["stage"], r["status"]) for r in out] == [(2, "error")]
    assert "no timed stage-1 row" in out[0]["reason"]
    assert sweep.summarize(out) == 1
    # Filters that leave no recorded shape: an error, not an empty success.
    with pytest.raises(ValueError, match="no recorded stage-1 shape"):
        sweep.retime(sweep.options(stage1_json=[path], gate_kind="kda"), {})
    with pytest.raises(SystemExit):
        sweep.main(["--stage1-json", path, "--gate-kind", "kda"])


def test_the_gate_rejects_a_kernel_that_reads_its_write_slot(monkeypatch):
    """The sweep gates on disjoint read / write slots. A kernel that reads its
    state from the write slot computes the right answer when the slots are the
    same (the in-place inputs timing uses), so only disjoint slots catch it."""
    torch = pytest.importorskip("torch")
    from builders.gfx950.gdn import gdn_decode as g

    def reads_write_slot(spec, inp, launcher, batch):
        out, state, _ = g.reference(spec, dict(inp, read_indices=inp["write_indices"]))
        values = {"out": out.to(torch.bfloat16), "state": inp["state"].clone()}
        values["state"][inp["write_indices"].long()] = state.to(values["state"].dtype)
        return values

    def not_timed(*args):
        raise AssertionError("a configuration that failed the gate was timed")

    driver = SimpleNamespace(
        make_inputs=lambda spec, batch, seed, disjoint_writes: g.make_inputs(
            spec, batch, seed=seed, device="cpu", disjoint_writes=disjoint_writes
        ),
        reference=g.reference,
        gate=g.gate,
        touched_bytes=g.touched_bytes,
        launcher_for=lambda spec, arch: object(),
        run_values=reads_write_slot,
        cold_value_sets=not_timed,
        capture_cold=not_timed,
        CaptureUnavailable=g.CaptureUnavailable,
        drain=lambda: None,
    )
    monkeypatch.setattr(sweep, "_driver", lambda: driver)
    auto = dispatch_gdn_decode(SHAPE.request())
    (row,) = sweep.stage1_shape(SHAPE, auto, [auto], [0], sweep.options(), {})
    assert row["status"] == "fail", row
    # The blind spot this closes: on in-place slots the same kernel passes.
    spec = auto.spec.kernel_spec
    inp = g.make_inputs(spec, 1, device="cpu", disjoint_writes=False)
    values = reads_write_slot(spec, inp, None, 1)
    assert g.gate(spec, inp, values, g.reference(spec, inp)).passed
