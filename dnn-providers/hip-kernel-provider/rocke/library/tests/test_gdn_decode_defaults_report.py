# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host tests of the gfx950 GDN/KDA decode defaults report: the selection
rule's every step, scoring over shapes, and the printed ``AUTO_DEFAULTS`` row."""

from __future__ import annotations

import argparse
import os
from types import MappingProxyType

import pytest

from benchmarks.gfx950.gdn import decode_defaults_report as report
from benchmarks.gfx950.gdn import decode_table_sweep as sweep
from dispatch.gdn import dispatch_gdn_decode
from dispatch.gdn.axes import AUTO_DEFAULTS, AutoDefault, _auto

SHAPES = [
    sweep.Shape("kda", "plain", "bf16", h, h, 128, b)
    for h, b in ((8, 1), (8, 64), (32, 16))
]


def _cand(name, score, worst, nknobs):
    return report.Candidate(
        tuning_id=name,
        tile=(1, 16, 8),
        knobs={f"k{i}": True for i in range(nknobs)},
        score=score,
        worst=worst,
        worst_shape=SHAPES[0],
        ratios={},
    )


@pytest.mark.parametrize(
    "candidates,pick,step",
    [
        # Clear score winner.
        ([_cand("a", 1.000, 1.30, 5), _cand("b", 1.020, 1.00, 0)], "a", "score"),
        # Scores tied within 1%: the better worst shape wins.
        ([_cand("a", 1.000, 1.20, 0), _cand("b", 1.005, 1.10, 5)], "b", "worst shape"),
        # Scores and worst shapes tied: fewer non-default knobs.
        (
            [_cand("a", 1.000, 1.100, 3), _cand("b", 1.004, 1.105, 1)],
            "b",
            "fewer knobs",
        ),
        # Everything tied: the lower score.
        (
            [_cand("b", 1.004, 1.10, 2), _cand("a", 1.000, 1.10, 2)],
            "a",
            "score (last step)",
        ),
        # A worst shape just outside the band breaks the tie although the
        # other has fewer knobs.
        (
            [_cand("a", 1.000, 1.100, 4), _cand("b", 1.000, 1.112, 0)],
            "a",
            "worst shape",
        ),
    ],
)
def test_rule_each_step_decides_where_the_previous_ties(candidates, pick, step):
    chosen, decided_by = report.apply_rule(candidates, tie=0.01)
    assert (chosen.tuning_id, decided_by) == (pick, step)


def test_rule_needs_a_candidate():
    with pytest.raises(ValueError, match="every shape"):
        report.apply_rule([])


def _row(shape, tid, us, *, stage=1, auto=False, status="ok", knobs=None, seed=0):
    return dict(
        shape.as_dict(),
        stage=stage,
        status=status,
        us=us,
        tuning_id=tid,
        tile=[1, 16, 8],
        knobs=knobs or {},
        auto=auto,
        rotate_mb=1024,
        sha="abc",
        device="gfx950",
        index=0,
        reason="" if status == "ok" else status,
        sweep_level="full",
        tuning_sample=40,
        seed=seed,
    )


def _robust_case():
    """x and y are each fastest somewhere and bad elsewhere; z is never fastest
    but close everywhere; w misses a shape; auto is a middling config."""
    times = {
        "x": (1.00, 1.50, 1.50),
        "y": (1.50, 1.00, 1.50),
        "z": (1.05, 1.05, 1.05),
        "auto": (1.20, 1.20, 1.00),
    }
    rows = [
        _row(shape, tid, us[i], auto=tid == "auto")
        for tid, us in times.items()
        for i, shape in enumerate(SHAPES)
    ]
    rows.append(_row(SHAPES[0], "w", 0.50, knobs={"dpp_reduce": True}))
    rows.append(_row(SHAPES[1], "w", 0.90, status="fail"))
    return rows


def test_scores_use_complete_configs_and_the_rule_can_differ_from_every_shape_best():
    cases = report.collect(_robust_case(), stage=1)
    (case,) = cases
    assert case == ("kda", False, "bf16")
    scored = {c.tuning_id: c for c in report.score_candidates(cases[case])}
    # w is fastest at one shape but untimed at the others: not a candidate.
    assert set(scored) == {"x", "y", "z", "auto"}
    # Ratios are against each shape's fastest, w included where it ran.
    assert scored["x"].ratios[SHAPES[0]] == pytest.approx(2.0)
    assert (scored["z"].worst, scored["z"].worst_shape) == (
        pytest.approx(2.1),
        SHAPES[0],
    )
    pick, _ = report.apply_rule(list(scored.values()))
    fastest = {
        min(("x", "y", "z", "auto"), key=lambda t: cases[case]["times"][s][t])
        for s in SHAPES
    }
    assert pick.tuning_id == "z" and "z" not in fastest


def test_repeated_rows_merge_by_median_and_stages_stay_apart():
    rows = [_row(SHAPES[0], "a", us) for us in (1.0, 9.0, 2.0)]
    rows += [_row(SHAPES[0], "a", 5.0, stage=2)]
    assert report.collect(rows, 1)[("kda", False, "bf16")]["times"][SHAPES[0]] == {
        "a": 2.0
    }
    assert report.collect(rows, 2)[("kda", False, "bf16")]["times"][SHAPES[0]] == {
        "a": 5.0
    }


def test_report_prints_the_proposal_and_fails_without_auto(capsys):
    args = argparse.Namespace(stage=1, tie=0.01, top=3)
    assert report.report(_robust_case(), args) == 0
    out = capsys.readouterr().out
    assert out.startswith("ADVISORY")
    assert "proposal z (decided by score)" in out
    assert (
        '("kda", False, "bf16"): AutoDefault((1, 16, 8), MappingProxyType({})),' in out
    )
    # The shortlist: the top 3 by score, the pick, and every shape's fastest
    # (w, y, auto), with the stage-1 walk flags the ids came from.
    assert (
        "--sweep-level full --tuning-sample 40 --seed 0 "
        "--tuning-ids z,auto,x,w,y --top-k 5" in out
    )
    rows = [r for r in _robust_case() if not (r["auto"] and r["batch"] == 64)]
    assert report.report(rows, args) == 1
    assert "auto: not timed at 1 shapes" in capsys.readouterr().out
    # Rows from two different walks cannot form one re-time command.
    mixed = _robust_case() + [_row(SHAPES[0], "v", 9.0, seed=1)]
    assert report.report(mixed, args) == 1
    assert "come from 2 walks" in capsys.readouterr().out


def test_a_shape_without_a_timed_row_or_with_auto_failing_is_never_dropped(capsys):
    """B16: every configuration failed (no timed row); B64: auto failed the
    gate. Neither may silently shrink the case the score covers."""
    args = argparse.Namespace(stage=1, tie=0.01, top=3)
    rows = [_row(SHAPES[0], t, us, auto=t == "a") for t, us in (("a", 1.0), ("b", 1.1))]
    rows += [
        _row(SHAPES[2], "a", None, auto=True, status="fail"),
        _row(SHAPES[2], "b", None, status="error"),
    ]
    assert report.report(rows, args) == 1
    out = capsys.readouterr().out
    assert "2 shapes, 1 with a timed row" in out
    assert f"UNSCORED {SHAPES[2].label}: FAIL (auto) a: fail, ERROR b: error" in out
    rows = [
        _row(s, t, us, auto=t == "a")
        for s in SHAPES[:2]
        for t, us in (("a", 1.0), ("b", 1.1))
    ]
    rows = [r for r in rows if not (r["auto"] and r["batch"] == 64)]
    rows.append(_row(SHAPES[1], "a", None, auto=True, status="fail"))
    assert report.report(rows, args) == 1
    assert f"UNSCORED {SHAPES[1].label}: FAIL (auto) a: fail" in capsys.readouterr().out


def test_cli_refuses_a_tie_band_or_top_that_cannot_work():
    for bad in (("--tie", "-0.01"), ("--tie", "nan"), ("--top", "0")):
        with pytest.raises(SystemExit):
            report.main(["rows.json", *bad])


def test_printed_rows_rebuild_every_shipped_default():
    """The printed row of each case's current ``auto`` configuration, as the
    sweep records it, evaluates to that case's ``AUTO_DEFAULTS`` entry."""
    path = os.path.join(os.path.dirname(sweep.__file__), "decode_shapes.json")
    seen = set()
    for shape in sweep.load_shapes(path):
        case = report.case_key(shape)
        if case in seen:
            continue
        seen.add(case)
        auto = dispatch_gdn_decode(shape.request())
        kernel = auto.spec.kernel_spec
        cand = report.Candidate(
            tuning_id=auto.spec.tuning_id,
            tile=(kernel.num_warps, kernel.warp_threads_k, kernel.blocks_per_v_dim),
            knobs=dict(auto.spec.knobs),
            score=1.0,
            worst=1.0,
            worst_shape=shape,
            ratios={},
        )
        text = report.auto_defaults_row(case, cand)
        key, value = text.rstrip(",").split(": ", 1)
        namespace = {
            "_auto": _auto,
            "AutoDefault": AutoDefault,
            "MappingProxyType": MappingProxyType,
        }
        assert eval(key) == case
        assert eval(value, namespace) == AUTO_DEFAULTS[case], text
    assert seen == set(AUTO_DEFAULTS)
