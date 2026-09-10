# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-device numeric checks for the GDN decode kernel (gfx942).

Compares the kernel against the same whole-tensor fp32 reference the gfx950
lanes use. The reference shares no algebra with the kernel -- no tiling, no
warp structure, no cross-lane reductions -- so agreement is evidence about the
algorithm rather than a restatement of the implementation.

Output error and state error are reported and asserted *separately*, never
collapsed with ``max()``. The kernel writes ``out`` and mutates the recurrent
state pool in place, and the in-place bf16 state write carries roughly thirty
times the absolute error of the output; one combined number would let a state
regression hide behind a healthy output.

Specs are built through the real dispatch path (``make_spec`` fed by
``tile_for_batch``) and every builder entry point is given ``arch=ARCH``, so
these lanes grade the gfx942 kernel rather than silently grading gfx950.

These lanes need a real gfx942 and ROCm torch, so they are marked ``gpu``. The
spec rules, emission and dispatch selection are covered by CPU-only tests.
"""

from __future__ import annotations

import dataclasses as dc

import pytest

torch = pytest.importorskip("torch", reason="ROCm torch required")

from dispatch.gdn import GdnDecodeRequest, dispatch_gdn_decode  # noqa: E402
from dispatch.gdn.gfx942 import (  # noqa: E402
    ARCH,
    TUNED_SPEC_IDS,
    make_spec,
    tile_for_batch,
)

pytestmark = pytest.mark.gpu


def _device_is_gfx942() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return ARCH in torch.cuda.get_device_properties(0).gcnArchName
    except Exception:
        return False


requires_gfx942 = pytest.mark.skipif(
    not _device_is_gfx942(), reason=f"needs a {ARCH} device"
)


def _req(batch: int, **kw) -> GdnDecodeRequest:
    kw.setdefault("arch", ARCH)
    return GdnDecodeRequest(batch=batch, **kw)


def _spec_for(batch: int, **kw):
    """The spec the dispatcher would pick for ``batch`` on gfx942."""
    return make_spec(_req(batch, **kw), tile_for_batch(batch))


def _tile(spec):
    return (spec.num_warps, spec.warp_threads_k, spec.blocks_per_v_dim)


@pytest.fixture(scope="module")
def harness():
    from builders.gfx950.gdn.gdn_decode import TOL, check, launch, launcher_for
    from builders.gfx950.gdn.gdn_decode import make_inputs, prepare, ref_fp32

    return {
        "TOL": TOL,
        "check": check,
        "launch": launch,
        "launcher_for": launcher_for,
        "make_inputs": make_inputs,
        "prepare": prepare,
        "ref_fp32": ref_fp32,
    }


@requires_gfx942
@pytest.mark.parametrize("batch", [1, 3, 16, 64])
def test_matches_fp32_reference(harness, batch):
    """Output and updated state both agree with the fp32 oracle."""
    spec = _spec_for(batch)
    out_err, state_err = harness["check"](spec, batch, arch=ARCH)
    assert out_err <= harness["TOL"], f"batch {batch}: output error {out_err:.3e}"
    assert state_err <= harness["TOL"], f"batch {batch}: state error {state_err:.3e}"


@requires_gfx942
def test_every_dispatched_tile_is_correct(harness):
    """Every tile the gfx942 table can select must be numerically sound.

    One batch inside each band, so all four tuned tiles are really built and
    run. A tuning table that can route a request to a wrong kernel is worse
    than no tuning at all.
    """
    seen = set()
    for batch in (1, 16, 64, 256):
        result = dispatch_gdn_decode(_req(batch))
        spec = result.spec
        assert _tile(spec) == tile_for_batch(batch), (
            f"batch {batch}: dispatcher picked {_tile(spec)}, "
            f"table says {tile_for_batch(batch)}"
        )
        seen.add(result.candidate.spec_id)
        out_err, state_err = harness["check"](spec, batch, arch=ARCH)
        assert out_err <= harness["TOL"], (
            f"batch {batch} tile {_tile(spec)}: output error {out_err:.3e}"
        )
        assert state_err <= harness["TOL"], (
            f"batch {batch} tile {_tile(spec)}: state error {state_err:.3e}"
        )
    assert seen == set(TUNED_SPEC_IDS), f"tiles never exercised: {set(TUNED_SPEC_IDS) - seen}"


@requires_gfx942
def test_padding_lanes_are_skipped_and_leave_state_untouched(harness):
    """A negative index means 'skip', and must leave the pool bit-identical.

    This is the continuous-batching contract: idle slots in a ragged request
    cost nothing and must come back unchanged to the last bit -- not merely
    close. The pool is deepened past ``batch`` so slots no live write index
    names at all are checked too, not just the padded lane's own slot.
    """
    spec = _spec_for(8)
    batch = 8
    inp = harness["make_inputs"](spec, batch)

    hv, dv, dk = spec.num_v_heads, spec.head_v_dim, spec.head_k_dim
    dev, dtype = inp["state"].device, inp["state"].dtype
    gen = torch.Generator(device=dev).manual_seed(7)
    tail = (
        torch.randn(4, hv, dv, dk, device=dev, generator=gen, dtype=torch.float32) * 0.01
    ).to(dtype)
    inp["state"] = torch.cat([inp["state"], tail], dim=0)

    padded = 3
    inp["read_indices"][padded] = -1
    inp["write_indices"][padded] = -1

    before = inp["state"].clone()
    values, cfg = harness["prepare"](spec, inp, batch, arch=ARCH)
    harness["launch"](harness["launcher_for"](spec, arch=ARCH), values, cfg)
    torch.cuda.synchronize()

    after = values["state"]
    live = {int(i) for i in inp["write_indices"].tolist() if i >= 0}
    assert padded not in live

    for slot in range(after.shape[0]):
        if slot in live:
            continue
        assert torch.equal(after[slot], before[slot]), (
            f"pool slot {slot} was modified even though no live write index "
            f"named it (live={sorted(live)})"
        )

    # Guard against the vacuous pass: a kernel that wrote nothing at all would
    # satisfy every assertion above.
    assert not torch.equal(
        after[0], before[0]
    ), "the kernel did not write any live slot; the skip check would be vacuous"


@requires_gfx942
def test_results_are_deterministic(harness):
    """Same inputs, same answer -- no dependence on scheduling or leftovers."""
    from builders.gfx950.gdn.gdn_decode import run

    spec = _spec_for(16)
    batch = 16
    launcher = harness["launcher_for"](spec, arch=ARCH)

    out_a, state_a = run(
        spec, harness["make_inputs"](spec, batch, seed=0), launcher, batch, arch=ARCH
    )
    out_b, state_b = run(
        spec, harness["make_inputs"](spec, batch, seed=0), launcher, batch, arch=ARCH
    )

    assert torch.equal(out_a, out_b), "output differed between identical runs"
    assert torch.equal(state_a, state_b), "state differed between identical runs"


@requires_gfx942
def test_state_dtype_variant_is_correct(harness):
    """An f16 recurrent state is a distinct kernel; it must be checked too."""
    from kernels.common.gdn_decode import is_valid_spec

    spec = dc.replace(_spec_for(8), state_dtype="f16")
    ok, why = is_valid_spec(spec, arch=ARCH)
    assert ok, why
    out_err, state_err = harness["check"](spec, 8, arch=ARCH)
    assert out_err <= harness["TOL"], f"f16 state: output error {out_err:.3e}"
    assert state_err <= harness["TOL"], f"f16 state: state error {state_err:.3e}"


@requires_gfx942
def test_end_to_end_through_the_dispatch_result(harness):
    """Drive a launch from the dispatch result alone, as a caller would.

    Every other lane reaches into the builder for the launcher and grid. This
    one uses only what ``dispatch_gdn_decode`` hands back -- the gfx942
    candidate, its build, signature, grid and block -- so a disagreement
    between the dispatcher's launch contract and the kernel it selected shows
    up as wrong numbers rather than passing unnoticed.
    """
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig, no_fence

    batch = 16
    result = dispatch_gdn_decode(_req(batch))
    assert ARCH in result.candidate.name, (
        f"dispatch returned {result.candidate.name!r}, not a {ARCH} candidate"
    )

    artifact = compile_kernel(result.build(), arch=ARCH)
    launcher = KernelLauncher(
        hsaco=artifact.hsaco,
        kernel_name=artifact.kernel_name,
        signature=result.signature,
    )

    inp = harness["make_inputs"](result.spec, batch)
    ref_out, ref_state = harness["ref_fp32"](result.spec, inp)
    values, _ = harness["prepare"](result.spec, inp, batch, arch=ARCH)
    cfg = LaunchConfig(grid=result.grid, block=result.block, stream=0)
    with no_fence():
        launcher(values, config=cfg)
    torch.cuda.synchronize()

    written = inp["write_indices"].long()
    out_err = (values["out"].float() - ref_out).abs().max().item()
    state_err = (values["state"].float()[written] - ref_state).abs().max().item()
    assert out_err <= harness["TOL"], f"dispatch-driven output error {out_err:.3e}"
    assert state_err <= harness["TOL"], f"dispatch-driven state error {state_err:.3e}"
