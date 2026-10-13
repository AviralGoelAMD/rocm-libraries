#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Sweep every gfx950 GDN/KDA decode configuration the dispatcher registry offers.

A *shape* is one decode problem: gate kind, mode (``plain``, or ``fused`` =
``fuse_conv`` + ``fuse_out_norm``), state dtype, head geometry ``Hk/Hv``, head
dim and batch. Per shape this walks ``dispatch_gdn_decode_all`` (every
registered tile x knob configuration at ``--sweep-level``; ``--bases-only``
keeps each tile's base configuration), checks every configuration once against
the fp32 reference on disjoint read / write slots (output, written state and
conv-state slots, untouched slots bit-exact) and times the ones that pass in
cold memory: each timing is a HIP graph cycling through buffer sets, one cycle
touching at least ``--rotate-mb`` MiB (``builders/gfx950/gdn/gdn_decode.py``,
``cold_copies``), so every launch reads its state from HBM as in a model's
decode step.

* Stage 1 times every configuration on its own: median of ``--rounds`` replays.
* Stage 2 re-times the shape's ``--top-k`` fastest stage-1 configurations and
  the ``auto`` configuration together: one graph each, replays interleaved, the
  order reversed every round and the base order forward / reversed / rotated by
  half from run to run; per run the median of ``--s2-rounds`` replays, and the
  reported time is the median of ``--runs`` run medians.

The ``auto`` configuration (``dispatch_gdn_decode``) is always timed: it is
appended to the walk when the walk does not list it. Measurements never change
what ``auto`` selects; ``decode_defaults_report.py`` turns the rows into an
advisory proposal for ``AUTO_DEFAULTS``.

    rocke-gdn-decode-table-sweep --gate-kind kda --mode fused --state-dtype f32 \\
        --heads 32/32 --batches 16 --output-json kda.json
    rocke-gdn-decode-table-sweep --shapes decode_shapes.json --list-only
    # one case on N GPUs: stage 1 split by configuration, then stage 2 by shape
    rocke-gdn-decode-table-sweep --shapes decode_shapes.json --gate-kind kda \\
        --mode plain --state-dtype bf16 --stage1-only --shard I/N --output-json s1_I.json
    rocke-gdn-decode-table-sweep --stage1-json 's1_*.json' --shard I/N --output-json s2_I.json
    rocke-gdn-decode-table-sweep --summarize 's1_*.json' 's2_*.json'
"""

from __future__ import annotations

import argparse
import dataclasses as dc
import glob
import json
import os
import statistics
import subprocess
import sys
from typing import Callable, Iterable, Sequence

from dispatch.gdn import GdnDecodeRequest, dispatch_gdn_decode, dispatch_gdn_decode_all
from dispatch.gdn.axes import AUTO_STATE_ROW

ARCH = "gfx950"
SCHEMA = "rocke.gdn_decode_table_sweep/1"
MODES = {"plain": {}, "fused": {"fuse_conv": True, "fuse_out_norm": True}}
STATE_DTYPES = ("bf16", "f16", "f32")
GATE_KINDS = ("gdn", "kda")
DEFAULT_HEADS = "16/32"
DEFAULT_BATCHES = (1, 8, 16, 32, 64, 128, 256)
DEFAULT_HEAD_DIM = 128
SHAPE_FIELDS = (
    "gate_kind",
    "mode",
    "state_dtype",
    "num_k_heads",
    "num_v_heads",
    "head_dim",
    "batch",
)
# A row's status: timed, failed the fp32 gate (never timed), raised while
# compiling or launching, or graph capture was unavailable.
OK, FAIL, ERROR, NO_GRAPH = "ok", "fail", "error", "no_graph"


@dc.dataclass(frozen=True, order=True)
class Shape:
    """One decode problem; ``case`` is its ``AUTO_DEFAULTS`` row."""

    gate_kind: str
    mode: str
    state_dtype: str
    num_k_heads: int
    num_v_heads: int
    head_dim: int
    batch: int

    def __post_init__(self):
        if self.gate_kind not in GATE_KINDS:
            raise ValueError(
                f"gate_kind must be one of {GATE_KINDS}: {self.gate_kind!r}"
            )
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {tuple(MODES)}: {self.mode!r}")
        if self.state_dtype not in STATE_DTYPES:
            raise ValueError(
                f"state_dtype must be one of {STATE_DTYPES}: {self.state_dtype!r}"
            )
        for name in ("num_k_heads", "num_v_heads", "head_dim", "batch"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive int: {value!r}")

    def request(self, **pin) -> GdnDecodeRequest:
        return GdnDecodeRequest(
            batch=self.batch,
            arch=ARCH,
            gate_kind=self.gate_kind,
            num_k_heads=self.num_k_heads,
            num_v_heads=self.num_v_heads,
            head_k_dim=self.head_dim,
            head_v_dim=self.head_dim,
            state_dtype=self.state_dtype,
            **MODES[self.mode],
            **pin,
        )

    @property
    def case(self) -> str:
        return f"{self.gate_kind}_{self.mode}_{AUTO_STATE_ROW[self.state_dtype]}"

    @property
    def label(self) -> str:
        return (
            f"{self.gate_kind} {self.mode} {self.state_dtype} "
            f"{self.num_k_heads}/{self.num_v_heads} D{self.head_dim} B{self.batch}"
        )

    def as_dict(self) -> dict:
        return dc.asdict(self)


def shape_of(row: dict) -> Shape:
    return Shape(**{name: row[name] for name in SHAPE_FIELDS})


# ---------------------------------------------------------------- CLI parsing
def parse_ints(text: str) -> tuple:
    values = tuple(int(x) for x in str(text).split(",") if x.strip())
    if not values:
        raise argparse.ArgumentTypeError(f"expected comma-separated ints, got {text!r}")
    return values


def parse_heads(text: str) -> tuple:
    """``"16/32,4/8"`` -> ``((16, 32), (4, 8))``: Hk/Hv pairs."""
    pairs = []
    for item in str(text).split(","):
        if not item.strip():
            continue
        parts = item.split("/")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(f"--heads wants Hk/Hv pairs, got {item!r}")
        pairs.append((int(parts[0]), int(parts[1])))
    if not pairs:
        raise argparse.ArgumentTypeError(f"--heads wants Hk/Hv pairs, got {text!r}")
    return tuple(pairs)


def parse_shard(text: str) -> tuple:
    index, count = (int(x) for x in str(text).split("/"))
    if count < 1 or not 0 <= index < count:
        raise argparse.ArgumentTypeError(
            f"--shard wants I/N with 0 <= I < N, got {text!r}"
        )
    return index, count


_ENTRY_KEYS = {
    "gate_kind",
    "mode",
    "state_dtype",
    "heads",
    "num_k_heads",
    "num_v_heads",
    "head_dim",
    "batches",
    "batch",
}


def expand_entry(entry: dict) -> list:
    """Shapes of one ``--shapes`` entry: ``gate_kind``, ``mode``,
    ``state_dtype``, ``heads`` (``["Hk/Hv", ...]``) or ``num_k_heads`` +
    ``num_v_heads``, ``batches`` (list) or ``batch``, optional ``head_dim``
    (default 128). An unknown key, both forms of the heads or the batches, or
    an empty list is an error, never resolved silently."""
    unknown = set(entry) - _ENTRY_KEYS
    if unknown:
        raise ValueError(f"unknown --shapes keys {sorted(unknown)} in {entry}")
    explicit = {"num_k_heads", "num_v_heads"} & set(entry)
    if "heads" in entry and explicit:
        raise ValueError(
            f"--shapes entry has both heads and {sorted(explicit)}: {entry}"
        )
    if "batches" in entry and "batch" in entry:
        raise ValueError(f"--shapes entry has both batches and batch: {entry}")
    for key in ("heads", "batches"):
        if key in entry and not entry[key]:
            raise ValueError(f"--shapes entry has an empty {key} list: {entry}")
    if "heads" in entry:
        heads = parse_heads(",".join(entry["heads"]))
    else:
        heads = ((entry["num_k_heads"], entry["num_v_heads"]),)
    batches = entry["batches"] if "batches" in entry else [entry["batch"]]
    return [
        Shape(
            gate_kind=entry["gate_kind"],
            mode=entry["mode"],
            state_dtype=entry["state_dtype"],
            num_k_heads=hk,
            num_v_heads=hv,
            head_dim=entry.get("head_dim", DEFAULT_HEAD_DIM),
            batch=batch,
        )
        for hk, hv in heads
        for batch in batches
    ]


def load_shapes(path: str) -> list:
    with open(path, encoding="utf-8") as fh:
        entries = json.load(fh)
    if not isinstance(entries, list):
        raise ValueError(f"{path}: expected a JSON list of shape entries")
    return [shape for entry in entries for shape in expand_entry(entry)]


def matches(shape: Shape, args) -> bool:
    """``shape`` passes every shape filter the command line set."""
    return (
        (args.gate_kind is None or shape.gate_kind == args.gate_kind)
        and (args.mode is None or shape.mode == args.mode)
        and (args.state_dtype is None or shape.state_dtype == args.state_dtype)
        and (
            args.heads is None
            or (shape.num_k_heads, shape.num_v_heads) in set(args.heads)
        )
        and (args.batches is None or shape.batch in set(args.batches))
        and (args.head_dim is None or shape.head_dim == args.head_dim)
    )


def select_shapes(args) -> list:
    """The shapes to sweep, deduplicated in first-seen order.

    With ``--shapes FILE`` the file's shapes, narrowed by any of
    ``--gate-kind`` / ``--mode`` / ``--state-dtype`` / ``--heads`` /
    ``--batches`` / ``--head-dim`` the command line gives. Without it, the
    grid those flags define (defaults: gdn, plain, bf16, 16/32, batches
    1..256, head dim 128).
    """
    if args.shapes:
        shapes = [s for s in load_shapes(args.shapes) if matches(s, args)]
    else:
        shapes = [
            Shape(
                gate_kind=args.gate_kind or "gdn",
                mode=args.mode or "plain",
                state_dtype=args.state_dtype or "bf16",
                num_k_heads=hk,
                num_v_heads=hv,
                head_dim=DEFAULT_HEAD_DIM if args.head_dim is None else args.head_dim,
                batch=batch,
            )
            for hk, hv in (args.heads or parse_heads(DEFAULT_HEADS))
            for batch in (args.batches or DEFAULT_BATCHES)
        ]
    return list(dict.fromkeys(shapes))


# ---------------------------------------------------------------- enumeration
def shape_configs(shape: Shape, args):
    """``(auto, configs)``: the ``auto`` result and the shape's configuration
    stream -- ``dispatch_gdn_decode_all`` at ``--sweep-level`` (base
    configurations only with ``--bases-only``; only ``--tuning-ids`` when
    given), with ``auto`` appended when the stream does not list it. The
    order is the registry's, so an index names the same configuration in
    every process that walks the same tree with the same arguments.

    A ``--tuning-ids`` entry this walk does not list is an error naming it and
    the walk, never dropped: a typo, or a shortlist from a walk at another
    ``--sweep-level`` / ``--tuning-sample`` / ``--seed``, would otherwise
    narrow the run to ``auto`` alone."""
    req = shape.request()
    auto = dispatch_gdn_decode(req)
    configs = list(
        dispatch_gdn_decode_all(
            req,
            sweep_level=args.sweep_level,
            tuning_sample=args.tuning_sample,
            seed=args.seed,
        )
    )
    if args.bases_only:
        configs = [r for r in configs if not r.spec.knobs]
    if args.tuning_ids:
        wanted = set(args.tuning_ids)
        configs = [r for r in configs if r.spec.tuning_id in wanted]
        found = {r.spec.tuning_id for r in configs} | {auto.spec.tuning_id}
        missing = [t for t in dict.fromkeys(args.tuning_ids) if t not in found]
        if missing:
            raise ValueError(
                f"--tuning-ids not in the walk of {shape.label} at --sweep-level "
                f"{args.sweep_level} --tuning-sample {args.tuning_sample} --seed "
                f"{args.seed}{' --bases-only' if args.bases_only else ''}: "
                f"{', '.join(missing)} (pass the walk flags the ids came from)"
            )
    if auto.spec.tuning_id not in {r.spec.tuning_id for r in configs}:
        configs.append(auto)
    return auto, configs


def window(count: int, shard=(0, 1), offset: int = 0, limit: int = 0) -> list:
    """Indices of a shape's ``count`` configurations this process runs:
    ``offset <= i < offset + limit`` (``limit`` 0 = to the end), and of those
    every ``N``-th from ``I`` (``shard = (I, N)``). Indices stay absolute, so
    a resumed or sharded run keeps the numbering, and the ``N`` shards of one
    window are disjoint and together cover it."""
    index, total = shard
    end = count if not limit else min(count, offset + limit)
    return [i for i in range(offset, end) if i % total == index]


def _unmeasured() -> dict:
    """A row's result fields before anything ran."""
    return dict(
        status="",
        us=None,
        runs=[],
        out_err=None,
        state_err=None,
        conv_err=None,
        spill=None,
        copies=None,
        reason="",
    )


def config_row(shape: Shape, result, index: int, auto_id: str, meta: dict) -> dict:
    """A row's identity: shape, configuration, run settings; no result yet."""
    kernel = result.spec.kernel_spec
    return dict(
        meta,
        **shape.as_dict(),
        case=shape.case,
        index=index,
        spec_id=result.candidate.spec_id,
        tile=[kernel.num_warps, kernel.warp_threads_k, kernel.blocks_per_v_dim],
        tuning_id=result.spec.tuning_id,
        knobs=dict(result.spec.knobs),
        auto=result.spec.tuning_id == auto_id,
        **_unmeasured(),
    )


def shape_error_row(shape: Shape, exc: Exception, meta: dict, stage: int = 1) -> dict:
    """The row of a shape with nothing to time: the dispatcher refuses it, or
    (stage 2 from files) it has no timed stage-1 row."""
    row = dict(
        meta,
        stage=stage,
        **shape.as_dict(),
        case=shape.case,
        index=-1,
        spec_id="",
        tile=[],
        tuning_id="",
        knobs={},
        auto=False,
        **_unmeasured(),
    )
    _error(row, exc)
    return row


# ---------------------------------------------------------------- stage 2
def run_order(count: int, run: int) -> list:
    """Run ``run``'s base order of ``count`` arms: forward, reversed, then
    rotated by half, repeating every three runs."""
    order = list(range(count))
    if run % 3 == 1:
        order.reverse()
    elif run % 3 == 2:
        order = order[count // 2 :] + order[: count // 2]
    return order


def interleaved_samples(
    samplers: Sequence[Callable[[], float]], rounds: int, run: int
) -> list:
    """``rounds`` samples per sampler, interleaved: each round takes one from
    every sampler, in :func:`run_order` on even rounds and reversed on odd
    ones, so slow drift and position in the round cancel across arms: exactly
    for linear drift when ``rounds`` is even, approximately when it is odd."""
    order = run_order(len(samplers), run)
    samples = [[] for _ in samplers]
    for r in range(rounds):
        for i in order if r % 2 == 0 else reversed(order):
            samples[i].append(samplers[i]())
    return samples


def stage2_picks(rows: Iterable[dict], top_k: int) -> list:
    """The stage-1 rows stage 2 re-times for one shape: the ``top_k`` fastest
    timed rows (ties by index), then the ``auto`` row if it is not among them
    (whatever its status: a failing ``auto`` is re-gated and reported, never
    dropped)."""
    rows = list(rows)
    timed = sorted(
        (r for r in rows if r["status"] == OK), key=lambda r: (r["us"], r["index"])
    )
    picks = timed[:top_k]
    for row in rows:
        if row["auto"] and row not in picks:
            picks.append(row)
    return picks


# ---------------------------------------------------------------- JSON rows
def write_rows(path: str, rows: list) -> None:
    """Rewrite ``path`` atomically; a crash keeps the previous complete file."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"schema": SCHEMA, "rows": rows}, fh, indent=1)
    os.replace(tmp, path)


def load_rows(patterns: Iterable[str]) -> list:
    """Every row of every file matching ``patterns`` (globs allowed). A file
    without this tool's schema is an error."""
    rows = []
    for pattern in patterns:
        paths = sorted(glob.glob(pattern)) or [pattern]
        for path in paths:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict) or data.get("schema") != SCHEMA:
                raise ValueError(f"{path}: not a {SCHEMA} file")
            rows.extend(data["rows"])
    return rows


def rows_by_shape(rows: Iterable[dict]) -> dict:
    out: dict = {}
    for row in rows:
        out.setdefault(shape_of(row), []).append(row)
    return out


# ---------------------------------------------------------------- summary
def shape_summary(rows: list) -> dict:
    """One shape's answer from its rows. ``best`` / ``auto`` come from the
    stage-2 rows if it has any, else stage 1: ``best`` is the fastest timed
    row, ``auto`` the ``auto`` row (``None`` when not timed), ``ratio`` auto /
    best. ``bad`` is every row of EVERY stage that failed the gate, raised or
    could not be timed: a stage-1 failure is never re-timed, so stage 2 alone
    would hide it."""
    stage = 2 if any(r["stage"] == 2 for r in rows) else 1
    staged = [r for r in rows if r["stage"] == stage]
    timed = [r for r in staged if r["status"] == OK]
    best = min(timed, key=lambda r: (r["us"], r["index"])) if timed else None
    auto = next((r for r in timed if r["auto"]), None)
    ratio = auto["us"] / best["us"] if auto and best else None
    return dict(
        stage=stage,
        best=best,
        auto=auto,
        ratio=ratio,
        timed=len(timed),
        bad=[r for r in rows if r["status"] != OK],
    )


def summarize(rows: list, out=None, require_auto: bool = True) -> int:
    """Print the best configuration per shape against ``auto``. Returns 1 if
    any row failed the gate, raised or could not be timed, or (with
    ``require_auto``; a windowed stage-1 run times ``auto`` in one window
    only) a shape has no timed ``auto`` row; else 0."""
    out = out or sys.stdout
    exit_code = 0
    if any(r["rotate_mb"] <= 0 for r in rows):
        print(
            "WARNING: some rows were timed WARM (--rotate-mb 0): debugging only",
            file=out,
        )
    print(
        f"{'shape':<36} {'stage':>5} {'timed':>5} {'best_us':>9} {'auto_us':>9} "
        f"{'auto/best':>9}  best tuning_id  knobs",
        file=out,
    )
    for shape, shape_rows in sorted(rows_by_shape(rows).items()):
        s = shape_summary(shape_rows)
        best, auto = s["best"], s["auto"]
        if best is None:
            print(
                f"{shape.label:<36} {s['stage']:>5} {0:>5}  no timed configuration",
                file=out,
            )
            exit_code = 1
        else:
            auto_us = f"{auto['us']:9.3f}" if auto else f"{'-':>9}"
            ratio = f"{s['ratio']:9.3f}" if s["ratio"] else f"{'-':>9}"
            print(
                f"{shape.label:<36} {s['stage']:>5} {s['timed']:>5} {best['us']:9.3f} "
                f"{auto_us} {ratio}  {best['tuning_id']}  "
                f"{json.dumps(best['knobs'], sort_keys=True)}",
                file=out,
            )
            if auto is None:
                print(f"{'':<36} auto configuration not timed", file=out)
                exit_code = 1 if require_auto else exit_code
        for row in s["bad"]:
            print(
                f"{'':<36} S{row['stage']} {row['status'].upper()} "
                f"{row['tuning_id']}: {row['reason'] or row['status']}",
                file=out,
            )
            exit_code = 1
    return exit_code


# ---------------------------------------------------------------- list-only
def list_configs(args, shapes: list) -> int:
    print(
        f"sweep_level={args.sweep_level} tuning_sample={args.tuning_sample} "
        f"seed={args.seed} bases_only={args.bases_only} shard={args.shard[0]}/"
        f"{args.shard[1]} offset={args.offset} limit={args.limit} (CPU list-only)"
    )
    refused = 0
    for shape in shapes:
        try:
            auto, configs = shape_configs(shape, args)
        except Exception as exc:  # the shape itself is refused
            print(f"\n{shape.label}  REFUSED {type(exc).__name__}: {exc}")
            refused += 1
            continue
        picked = window(len(configs), args.shard, args.offset, args.limit)
        print(
            f"\n{shape.label}  configurations={len(configs)} this process={len(picked)}"
        )
        for i in picked:
            result = configs[i]
            mark = "  <- auto" if result.spec.tuning_id == auto.spec.tuning_id else ""
            print(
                f"  {i:5d} {result.candidate.spec_id:<20} {result.spec.tuning_id}  "
                f"{json.dumps(dict(result.spec.knobs), sort_keys=True)}{mark}"
            )
    return 1 if refused else 0


# ---------------------------------------------------------------- GPU
def _driver():
    """The builder's host driver (imports torch): GPU paths only."""
    from builders.gfx950.gdn import gdn_decode

    return gdn_decode


class _ShapeInputs:
    """One shape's gate inputs and fp32 reference, timing inputs and cold value
    sets, shared by every configuration: they depend only on the problem, and
    every configuration of a problem takes the same kernel arguments.

    The gate runs on DISJOINT read and write slots (``make_inputs`` default):
    with ``read == write`` a kernel that reads its state from the write slot,
    or writes into the read slot, computes the same answer and passes, so the
    in-place set cannot see that bug. The pool keeps spare slots, which the
    gate checks stay untouched. Timing runs on the in-place set (read slots
    equal write slots, as in a serving engine's state update), which is also
    what :func:`touched_bytes` sizes the rotation from."""

    def __init__(self, shape: Shape, auto, args):
        g = _driver()
        self.spec = auto.spec.kernel_spec
        self.batch = shape.batch
        self.rotate_mb = args.rotate_mb
        self.gate_inp = g.make_inputs(
            self.spec, shape.batch, seed=args.seed, disjoint_writes=True
        )
        self.ref = g.reference(self.spec, self.gate_inp)
        self.inp = g.make_inputs(
            self.spec, shape.batch, seed=args.seed, disjoint_writes=False
        )
        self.touched = g.touched_bytes(self.spec, self.inp, shape.batch)
        self._sets = None

    def gate(self, result, row: dict):
        """Gate ``result`` and fill ``row``'s errors; its launcher if it passed."""
        g = _driver()
        spec = result.spec.kernel_spec
        launcher = g.launcher_for(spec, arch=ARCH)
        values = g.run_values(spec, self.gate_inp, launcher, self.batch)
        verdict = g.gate(spec, self.gate_inp, values, self.ref)
        del values
        row.update(
            out_err=verdict.out_err,
            state_err=verdict.state_err,
            conv_err=verdict.conv_err,
            spill=verdict.spill,
        )
        if verdict.passed:
            return launcher
        row.update(
            status=FAIL,
            reason=f"fp32 gate: worst={verdict.worst:.3e} spill={verdict.spill}",
        )
        return None

    def capture(self, result, launcher):
        """A cold graph of ``result`` on this shape's value sets (made on first
        use, then shared); raises the driver's ``CaptureUnavailable`` if
        capture fails."""
        g = _driver()
        if self._sets is None:
            self._sets = g.cold_value_sets(
                self.spec, self.inp, self.batch, self.rotate_mb
            )
            k = len(self._sets)
            print(
                f"ROTATION {self.touched / 2**20:.3f} MiB touched per launch x "
                f"{k} buffer sets = {k * self.touched / 2**20:.1f} MiB per cycle "
                f"(target >= {self.rotate_mb}); a set is reused after "
                f"{(k - 1) * self.touched / 2**20:.1f} MiB",
                flush=True,
            )
        return g.capture_cold(result.spec.kernel_spec, self.batch, self._sets, launcher)

    def release(self) -> None:
        """Drop the value sets; the next :meth:`capture` makes fresh ones."""
        self._sets = None
        _driver().drain()


def _reason(text: str) -> str:
    """A row's ``reason``: one line, at most 400 characters."""
    return text.replace("\n", " ")[:400]


def _error(row: dict, exc: Exception) -> None:
    row.update(status=ERROR, reason=_reason(f"{type(exc).__name__}: {exc}"))


def _no_graph(row: dict, exc: Exception) -> None:
    row.update(status=NO_GRAPH, reason=_reason(f"graph capture unavailable: {exc}"))


def _print_row(shape: Shape, row: dict) -> None:
    us = f"{row['us']:9.3f}us" if row["us"] is not None else f"{'-':>11}"
    extra = "" if row["status"] == OK else f"  {row['reason']}"
    print(
        f"S{row['stage']} {shape.label} {row['index']:5d} {row['tuning_id']} {us} "
        f"{row['status'].upper()}{'  <- auto' if row['auto'] else ''}{extra}",
        flush=True,
    )


def stage1_shape(shape: Shape, auto, configs, indices, args, meta) -> list:
    """Gate and time ``configs[i]`` for every ``i`` in ``indices``, each on its
    own: median of ``--rounds`` replays of its cold graph."""
    g = _driver()
    data = _ShapeInputs(shape, auto, args)
    rows = []
    for i in indices:
        result = configs[i]
        row = config_row(shape, result, i, auto.spec.tuning_id, dict(meta, stage=1))
        try:
            launcher = data.gate(result, row)
            if launcher is not None:
                try:
                    cold = data.capture(result, launcher)
                except g.CaptureUnavailable as exc:
                    cold = None
                    _no_graph(row, exc)
                if cold is not None:
                    try:
                        samples = [cold.replay_us() for _ in range(args.rounds)]
                    finally:
                        cold.close()
                    us = statistics.median(samples)
                    row.update(status=OK, us=us, runs=[us], copies=cold.copies)
        except Exception as exc:  # compile / launch failure: recorded, not timed
            _error(row, exc)
        finally:
            g.drain()
        rows.append(row)
        _print_row(shape, row)
    data.release()
    return rows


def stage2_shape(shape: Shape, auto, picks: list, args, meta) -> list:
    """Re-time ``picks`` (``(index, result)``) together, interleaved
    (:func:`interleaved_samples`), for ``--runs`` runs of ``--s2-rounds``
    rounds; each run captures fresh graphs on fresh value sets."""
    data = _ShapeInputs(shape, auto, args)
    entries = []  # (row, result, launcher)
    rows = []
    for index, result in picks:
        row = config_row(shape, result, index, auto.spec.tuning_id, dict(meta, stage=2))
        rows.append(row)
        try:
            launcher = data.gate(result, row)
            if launcher is not None:
                entries.append((row, result, launcher))
        except Exception as exc:
            _error(row, exc)
        finally:
            _driver().drain()
    for run in range(args.runs):
        graphs = []
        for row, result, launcher in entries:
            if row["status"] in (ERROR, NO_GRAPH):
                continue
            try:
                cold = data.capture(result, launcher)
            except _driver().CaptureUnavailable as exc:
                _no_graph(row, exc)
                continue
            except Exception as exc:
                _error(row, exc)
                continue
            graphs.append((row, cold))
        samples = interleaved_samples(
            [cold.replay_us for _, cold in graphs], args.s2_rounds, run
        )
        for (row, cold), got in zip(graphs, samples):
            row["runs"].append(statistics.median(got))
            row["copies"] = cold.copies
            cold.close()
        del graphs
        data.release()
    for row, _, _ in entries:
        if row["status"] not in (ERROR, NO_GRAPH):
            row.update(status=OK, us=statistics.median(row["runs"]))
    for row in rows:
        _print_row(shape, row)
    return rows


def pinned_result(shape: Shape, row: dict):
    """Re-dispatch a recorded configuration from its ``spec_id``, ``tuning_id``
    and ``knobs``; the pin must resolve to exactly that ``tuning_id``."""
    result = dispatch_gdn_decode(
        shape.request(
            spec_id=row["spec_id"],
            tuning_id=row["tuning_id"],
            tuning_knobs=row["knobs"],
        )
    )
    if result.spec.tuning_id != row["tuning_id"]:
        raise ValueError(
            f"{row['tuning_id']} re-dispatched as {result.spec.tuning_id} on {shape.label}"
        )
    return result


def _source_sha() -> str:
    try:
        return subprocess.run(
            [
                "git",
                "-C",
                os.path.dirname(os.path.abspath(__file__)),
                "rev-parse",
                "HEAD",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_meta(args, device: str) -> dict:
    """The run settings every row carries, so one row stands on its own."""
    return dict(
        arch=ARCH,
        device=device,
        sha=args.source_sha or _source_sha(),
        sweep_level=args.sweep_level,
        tuning_sample=args.tuning_sample,
        seed=args.seed,
        rotate_mb=args.rotate_mb,
        rounds=args.rounds,
        s2_rounds=args.s2_rounds,
        top_k=args.top_k,
    )


def sweep(args, shapes: list, meta: dict) -> list:
    """Stage 1 per shape, then (unless ``--stage1-only``) stage 2 on it."""
    rows = []
    for shape in shapes:
        try:
            auto, configs = shape_configs(shape, args)
        except Exception as exc:  # the shape itself is refused
            print(f"ERROR {shape.label}: {type(exc).__name__}: {exc}", flush=True)
            rows.append(shape_error_row(shape, exc, meta))
            continue
        indices = window(len(configs), args.shard, args.offset, args.limit)
        print(
            f"\n=== {shape.label}: {len(configs)} configurations, "
            f"{len(indices)} in this process; auto {auto.spec.tuning_id}",
            flush=True,
        )
        if not indices:
            continue
        stage1 = stage1_shape(shape, auto, configs, indices, args, meta)
        rows.extend(stage1)
        if not args.stage1_only:
            picks = stage2_picks(stage1, args.top_k)
            rows.extend(
                stage2_shape(
                    shape,
                    auto,
                    [(r["index"], configs[r["index"]]) for r in picks],
                    args,
                    meta,
                )
            )
        if args.output_json:
            write_rows(args.output_json, rows)
    return rows


def stage1_shapes(args):
    """``(by_shape, shapes)``: the ``--stage1-json`` stage-1 rows by shape, and
    the recorded shapes the shape filters keep. No shape left is an error:
    stage 2 would time nothing and report success."""
    by_shape = rows_by_shape(r for r in load_rows(args.stage1_json) if r["stage"] == 1)
    shapes = [s for s in sorted(by_shape) if matches(s, args)]
    if not shapes:
        raise ValueError("no recorded stage-1 shape matches the given filters")
    return by_shape, shapes


def retime(args, meta: dict) -> list:
    """Stage 2 from saved stage-1 rows (``--stage1-json``): every matching
    shape (``--shard`` splits the shapes), top ``--top-k`` plus ``auto``. A
    shape with no timed stage-1 row gets an ``error`` row, so the run fails."""
    by_shape, shapes = stage1_shapes(args)
    mine = [s for i, s in enumerate(shapes) if i % args.shard[1] == args.shard[0]]
    rows = []
    for shape in mine:
        recorded = by_shape[shape]
        if not any(r["status"] == OK for r in recorded):
            print(f"ERROR {shape.label}: no timed stage-1 row", flush=True)
            rows.append(
                shape_error_row(
                    shape, ValueError("no timed stage-1 row"), meta, stage=2
                )
            )
            continue
        auto = dispatch_gdn_decode(shape.request())
        if not any(
            r["auto"] and r["tuning_id"] == auto.spec.tuning_id for r in recorded
        ):
            print(
                f"WARNING {shape.label}: auto is {auto.spec.tuning_id} here, not the "
                "stage-1 auto row; it is re-timed as auto",
                flush=True,
            )
        picks = []
        for row in stage2_picks(recorded, args.top_k):
            picks.append((row["index"], pinned_result(shape, row)))
        if auto.spec.tuning_id not in {res.spec.tuning_id for _, res in picks}:
            picks.append((-1, auto))
        print(
            f"\n=== {shape.label}: stage 2 on {len(picks)} configurations", flush=True
        )
        rows.extend(stage2_shape(shape, auto, picks, args, meta))
        if args.output_json:
            write_rows(args.output_json, rows)
    return rows


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    shapes = ap.add_argument_group("shapes (filters when --shapes is given)")
    shapes.add_argument("--shapes", default="", help="JSON list of shape entries")
    shapes.add_argument("--gate-kind", choices=GATE_KINDS, default=None)
    shapes.add_argument("--mode", choices=tuple(MODES), default=None)
    shapes.add_argument("--state-dtype", choices=STATE_DTYPES, default=None)
    shapes.add_argument("--heads", type=parse_heads, default=None, help="Hk/Hv list")
    shapes.add_argument("--batches", type=parse_ints, default=None)
    shapes.add_argument("--head-dim", type=int, default=None)
    walk = ap.add_argument_group("configurations")
    walk.add_argument(
        "--sweep-level", choices=("production", "full"), default="production"
    )
    walk.add_argument(
        "--tuning-sample",
        type=int,
        default=0,
        help="with --sweep-level full: random legal configurations per tile, "
        "seeded by --seed (0 = every one); ignored at production",
    )
    walk.add_argument("--seed", type=int, default=0, help="sampling and input seed")
    walk.add_argument(
        "--bases-only", action="store_true", help="each tile's base configuration only"
    )
    walk.add_argument(
        "--tuning-ids",
        type=lambda text: tuple(t for t in text.split(",") if t),
        default=(),
        help="only these tuning_ids (plus auto), e.g. a shortlist to re-time",
    )
    walk.add_argument("--shard", type=parse_shard, default=(0, 1), metavar="I/N")
    walk.add_argument("--offset", type=int, default=0, help="configs to skip per shape")
    walk.add_argument(
        "--limit", type=int, default=0, help="configs per shape (0 = all)"
    )
    timing = ap.add_argument_group("timing")
    timing.add_argument(
        "--rotate-mb",
        type=int,
        default=None,
        help="MiB one cold-memory rotation cycle touches at least (default: the "
        "driver's ROTATE_MB; see gdn_decode.cold_copies); 0 = warm, debugging "
        "only, and says so",
    )
    timing.add_argument("--rounds", type=int, default=5, help="stage-1 replays")
    timing.add_argument(
        "--top-k", type=int, default=8, help="stage-2 arms besides auto"
    )
    timing.add_argument(
        "--s2-rounds",
        type=int,
        default=30,
        help="stage-2 rounds per run; an even count cancels linear drift exactly "
        "(the order reverses every round)",
    )
    timing.add_argument("--runs", type=int, default=3, help="stage-2 runs")
    timing.add_argument("--stage1-only", action="store_true")
    timing.add_argument(
        "--stage1-json",
        nargs="+",
        default=None,
        metavar="JSON",
        help="run stage 2 only, from these stage-1 files; --shard splits the shapes",
    )
    out = ap.add_argument_group("output")
    out.add_argument("--output-json", default="")
    out.add_argument(
        "--source-sha", default="", help="recorded per row (default: git HEAD)"
    )
    out.add_argument("--list-only", action="store_true", help="CPU listing; no GPU")
    out.add_argument(
        "--summarize",
        nargs="+",
        default=None,
        metavar="JSON",
        help="best configuration per shape vs auto from saved files; no GPU",
    )
    return ap


def options(**overrides) -> argparse.Namespace:
    """The command line's defaults with ``overrides``: the settings object the
    sweep functions read, for callers that drive them from Python."""
    args = build_parser().parse_args([])
    unknown = set(overrides) - set(vars(args))
    if unknown:
        raise TypeError(f"unknown sweep options {sorted(unknown)}")
    vars(args).update(overrides)
    return args


# Flags that only stage 2 reads.
_STAGE2_FLAGS = ("top_k", "s2_rounds", "runs")


def check_args(ap: argparse.ArgumentParser, args) -> None:
    """Refuse every flag the chosen mode would accept and then ignore."""
    defaults = vars(ap.parse_args([]))

    def changed(*names):
        return [
            "--" + name.replace("_", "-")
            for name in names
            if getattr(args, name) != defaults[name]
        ]

    if args.summarize:
        # --summarize reads saved rows and nothing else.
        ignored = changed(*(name for name in defaults if name != "summarize"))
        if ignored:
            ap.error(f"--summarize reads saved rows only: no {', '.join(ignored)}")
        return
    windowed = args.shard[1] > 1 or args.offset or args.limit
    if args.stage1_json:
        # Stage 2 from files re-times the recorded rows: nothing is walked, so
        # a walk flag would be accepted and ignored.
        ignored = changed(
            "stage1_only",
            "list_only",
            "offset",
            "limit",
            "shapes",
            "bases_only",
            "tuning_ids",
            "sweep_level",
            "tuning_sample",
        )
        if ignored:
            ap.error(
                f"--stage1-json runs stage 2 on recorded rows: no {', '.join(ignored)}"
            )
    elif windowed and not args.stage1_only and not args.list_only:
        ap.error(
            "--shard/--offset/--limit see part of each shape's configurations, so "
            "stage 2 would rank a part: pass --stage1-only, then run stage 2 with "
            "--stage1-json"
        )
    if args.stage1_only:
        ignored = changed(*_STAGE2_FLAGS)
        if ignored:
            ap.error(f"--stage1-only runs no stage 2: no {', '.join(ignored)}")
    for name in ("rounds", "s2_rounds", "runs", "top_k"):
        if getattr(args, name) < 1:
            ap.error(f"--{name.replace('_', '-')} must be >= 1")
    if args.offset < 0 or args.limit < 0:
        ap.error("--offset and --limit must be >= 0")
    if args.head_dim is not None and args.head_dim < 1:
        ap.error("--head-dim must be >= 1")
    if args.rotate_mb is not None and args.rotate_mb < 0:
        ap.error("--rotate-mb must be >= 0 (0 = warm)")


def main(argv: Iterable[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    check_args(ap, args)
    if args.summarize:
        return summarize(load_rows(args.summarize))
    if args.stage1_json:
        try:
            stage1_shapes(args)
        except (OSError, ValueError) as exc:
            ap.error(str(exc))
        shapes = []
    else:
        shapes = select_shapes(args)
        if not shapes:
            ap.error("no shape matches the given filters")
    if args.list_only:
        return list_configs(args, shapes)

    try:
        import torch
    except ImportError:
        print("torch is required to time configurations", file=sys.stderr)
        return 2
    from rocke.runtime.hip_module import get_device_arch

    device = get_device_arch() if torch.cuda.is_available() else None
    if not device or not device.startswith(ARCH):
        print(f"requires a {ARCH} device; visible: {device!r}", file=sys.stderr)
        return 2
    from builders.gfx950.gdn.gdn_decode import ROTATE_MB, timing_label

    if args.rotate_mb is None:
        args.rotate_mb = ROTATE_MB
    print(timing_label(args.rotate_mb), flush=True)
    meta = run_meta(args, device)
    rows = retime(args, meta) if args.stage1_json else sweep(args, shapes, meta)
    if args.output_json:
        write_rows(args.output_json, rows)
        print(f"\nwrote {len(rows)} rows to {args.output_json}")
    print()
    windowed = args.shard[1] > 1 or args.offset or args.limit
    return summarize(rows, require_auto=bool(args.stage1_json) or not windowed)


if __name__ == "__main__":
    raise SystemExit(main())
