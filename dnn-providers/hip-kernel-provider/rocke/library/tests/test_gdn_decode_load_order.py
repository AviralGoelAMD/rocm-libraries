# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Structural test: the warp-tiled GDN/KDA decode issues every global load
before any floating-point math (ALGORITHM.md §4.3, "Why loads first").

The golden hashes only say that the IR moved; the numeric tests pass in either
order. This test pins the order itself: inside the ``active`` region, in
program order, the last global load must come before the first float op. An
emitter that loads a tensor after the math has started (the interleaved order
the simple path keeps) fails it.

One tile is exempt by design: single-wave ``fuse_conv`` owns all ``DV`` rows,
and hoisting their inputs as well would spill, so it loads each row's ``v``
conv inputs, the state and the norm inputs where the parent order did
(ALGORITHM.md §4.3, "One exception"). Its zero-scratch contract is pinned in
``test_gdn_decode_fused.py``.

Builds IR only: no lowering, no GPU, no comgr.
"""

from __future__ import annotations

import dataclasses as dc
import sys
from pathlib import Path

import pytest

_LIB_ROOT = str(Path(__file__).resolve().parent.parent)
if sys.path and sys.path[0] != _LIB_ROOT:
    sys.path.insert(0, _LIB_ROOT)

from dispatch.gdn import GdnDecodeRequest, dispatch_gdn_decode_all  # noqa: E402
from dispatch.gdn.gfx950 import _TUNED_TILES_KDA  # noqa: E402
from kernels.gfx950.gdn_decode import GdnDecodeSpec, build_gdn_decode  # noqa: E402

_ARCH = "gfx950"
_FLOAT_OPS = {
    "arith.fadd",
    "arith.fsub",
    "arith.fmul",
    "arith.fma",
    "arith.fneg",
    "arith.fmin",
    "arith.fmax",
    "arith.fcmp",
}


def _is_global_load(name: str) -> bool:
    return "global_load" in name or "buffer_load" in name


def _is_float_math(name: str) -> bool:
    return name in _FLOAT_OPS or name.startswith("math.")


def _flatten(ops):
    for op in ops:
        yield op
        for region in op.regions:
            yield from _flatten(region.ops)


def _active_region_ops(kernel):
    """Ops of the ``scf_if(active)`` body, nested regions inlined in order."""
    guards = [op for op in kernel.body.ops if op.name == "scf.if"]
    assert len(guards) == 1, [op.name for op in kernel.body.ops]
    return [op.name for op in _flatten(guards[0].regions[0].ops)]


# Every tile knob on that applies to the spec it is added to; the knob code
# must keep the loads-first order too.
_TILE_KNOBS = dict(
    state_load_hint="streaming",
    state_store_hint="streaming",
    dpp_reduce=True,
    xcd_remap=True,
    stream_rows=True,
    interleave_cols=True,
    waves_per_eu=4,
)


def _specs():
    out = {}
    for result in dispatch_gdn_decode_all(GdnDecodeRequest(batch=16, arch=_ARCH)):
        out[f"registered_{result.candidate.spec_id}"] = result.spec
    base = GdnDecodeSpec()
    for _, tile, spec_id in _TUNED_TILES_KDA:
        for state_dtype in ("bf16", "f16", "f32"):
            spec = dc.replace(
                base,
                gate_kind="kda",
                state_dtype=state_dtype,
                num_warps=tile[0],
                warp_threads_k=tile[1],
                blocks_per_v_dim=tile[2],
            )
            out[f"tuned_{spec_id}_st{state_dtype}"] = spec
            out[f"tuned_{spec_id}_st{state_dtype}_knobs"] = dc.replace(
                spec, **_TILE_KNOBS
            )
    out["kda_raw_gate"] = dc.replace(base, gate_kind="kda", fuse_gate=False)
    out["no_l2norm"] = dc.replace(base, use_qk_l2norm=False)
    for gate in ("gdn", "kda"):
        for state_dtype in ("bf16", "f16", "f32"):
            for conv, norm in ((True, False), (False, True), (True, True)):
                for nw in (1, 4):
                    if conv and nw == 1:
                        continue  # the exempt tile (module docstring)
                    tag = ("_cv" if conv else "") + ("_rn" if norm else "")
                    spec = dc.replace(
                        base,
                        num_k_heads=16,
                        num_v_heads=16,
                        gate_kind=gate,
                        state_dtype=state_dtype,
                        num_warps=nw,
                        warp_threads_k=16,
                        blocks_per_v_dim=1,
                        fuse_conv=conv,
                        fuse_out_norm=norm,
                    )
                    case = f"fused_{gate}_st{state_dtype}{tag}_w{nw}"
                    out[case] = spec
                    # the fused knobs with every lane loading its own norm
                    # gate, then with the once-per-row LDS products
                    lane = dc.replace(spec, conv_once=conv, out_lds=norm, **_TILE_KNOBS)
                    out[f"{case}_knobs"] = lane
                    if conv and norm:
                        out[f"{case}_knobs_ngo"] = dc.replace(lane, norm_gate_once=True)
    return out


_SPECS = _specs()


@pytest.mark.parametrize("case", sorted(_SPECS))
def test_every_global_load_precedes_the_first_float_op(case):
    names = _active_region_ops(build_gdn_decode(_SPECS[case], arch=_ARCH))
    loads = [i for i, n in enumerate(names) if _is_global_load(n)]
    maths = [i for i, n in enumerate(names) if _is_float_math(n)]
    assert loads and maths, case
    late = [names[i] for i in loads if i > maths[0]]
    assert not late, (
        f"{case}: {len(late)} global load(s) after the first float op "
        f"({names[maths[0]]} at op {maths[0]}): {late[:4]}"
    )
