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
``tile_for_work``) and every builder entry point is given ``arch=ARCH``, so
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
    spec_id_for_work,
    tile_for_work,
    work_for,
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
    req = _req(batch, **kw)
    return make_spec(req, tile_for_work(work_for(batch, req.num_v_heads)))


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


def _one_case_per_band() -> dict:
    """spec_id -> (batch, num_k_heads, num_v_heads) landing in that band.

    Derived from the table rather than hard-coded. A hard-coded list silently
    stops covering a tile the moment the bands are re-tuned, which is exactly
    when coverage matters most.

    The search varies HEAD COUNT as well as batch, because the table is keyed
    on work = batch * num_v_heads. At the production geometry (Hv=32) the
    smallest band is unreachable -- batch 1 already means work 32 -- so a
    batch-only search would silently never exercise it. Tensor-parallel shards
    (Hv of 16/8/4) are what reach the low-work bands, and they are real
    deployments, not synthetic cases.
    """
    geometries = ((16, 32), (8, 16), (4, 8), (2, 4))
    seen: dict = {}
    for hk, hv in geometries:
        for batch in (1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 4096):
            seen.setdefault(spec_id_for_work(work_for(batch, hv)), (batch, hk, hv))
    return seen


@requires_gfx942
def test_every_dispatched_tile_is_correct(harness):
    """Every tile the gfx942 table can select must be numerically sound.

    One batch inside each band, so every tuned tile is really built and run. A
    tuning table that can route a request to a wrong kernel is worse than no
    tuning at all.
    """
    seen = set()
    for spec_id, (batch, hk, hv) in sorted(_one_case_per_band().items()):
        result = dispatch_gdn_decode(_req(batch, num_k_heads=hk, num_v_heads=hv))
        spec = result.spec
        work = work_for(batch, hv)
        assert _tile(spec) == tile_for_work(work), (
            f"batch {batch} x {hv} v-heads (work {work}): dispatcher picked "
            f"{_tile(spec)}, table says {tile_for_work(work)}"
        )
        seen.add(result.candidate.spec_id)
        out_err, state_err = harness["check"](spec, batch, arch=ARCH)
        assert out_err <= harness["TOL"], (
            f"batch {batch} x {hv} heads tile {_tile(spec)}: "
            f"output error {out_err:.3e}"
        )
        assert state_err <= harness["TOL"], (
            f"batch {batch} x {hv} heads tile {_tile(spec)}: "
            f"state error {state_err:.3e}"
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


@requires_gfx942
def test_exp2_fast_matches_guarded_exp2(harness):
    """GDN's exponential arguments are not bounded above by zero, unlike every
    other exp2_fast site in this tree, so the fast path needs its own evidence
    rather than inheriting the softmax argument's justification.

    ``exp2_fast`` (``v_exp_f32``, no range-reduction clamp) is safe elsewhere
    because the softmax argument is always <= 0. GDN exponentiates unbounded
    ``A_log``, ``a + dt_bias`` and ``-b``. This runs both kernels on identical
    inputs and compares them against the same fp32 oracle.
    """
    base = _spec_for(8)
    fast = dc.replace(base, use_exp2_fast=True)
    slow = dc.replace(base, use_exp2_fast=False)

    # Without distinct names the launcher cache would hand back one kernel
    # twice and the comparison below would be vacuous.
    assert fast.kernel_name() != slow.kernel_name(), (
        f"both variants named {fast.kernel_name()!r}; the A/B would compare "
        "one compiled kernel against itself"
    )

    fast_out, fast_state = harness["check"](fast, 8, arch=ARCH)
    slow_out, slow_state = harness["check"](slow, 8, arch=ARCH)
    print(
        f"\nexp2_fast : out_err={fast_out:.3e} state_err={fast_state:.3e}"
        f"\nexp2 guard: out_err={slow_out:.3e} state_err={slow_state:.3e}"
    )

    tol = harness["TOL"]
    # Both must be correct on their own terms.
    assert fast_out <= tol, f"exp2_fast output error {fast_out:.3e}"
    assert fast_state <= tol, f"exp2_fast state error {fast_state:.3e}"
    assert slow_out <= tol, f"guarded exp2 output error {slow_out:.3e}"
    assert slow_state <= tol, f"guarded exp2 state error {slow_state:.3e}"
    # The fast path may be less accurate, but not by an order of magnitude.
    assert fast_out < max(10 * slow_out, tol), (
        f"exp2_fast output error {fast_out:.3e} is more than 10x the guarded "
        f"path's {slow_out:.3e}"
    )
    assert fast_state < max(10 * slow_state, tol), (
        f"exp2_fast state error {fast_state:.3e} is more than 10x the guarded "
        f"path's {slow_state:.3e}"
    )


# Gate values chosen to break the exponential, not to look plausible. Each
# tuple is (A_log, a, -b sweep) applied head-by-head, so the pairing is exact.
#   "wide"       exp(A_log) stays finite (2e-9 .. 5e8) and decay sweeps its
#                whole range; a + dt_bias crosses the softplus threshold from
#                both sides; b saturates the sigmoid at both ends.
#   "saturating" exp(A_log) overflows to +inf, so decay is 0 everywhere and
#                beta is pinned to 0 or 1. ``a`` is kept positive here on
#                purpose: inf * softplus(0) would be NaN in the oracle too,
#                which would test the reference, not the kernel.
_GATE_REGIMES = {
    "wide": (
        [-20.0, -8.0, -1.0, 0.0, 1.0, 4.0, 12.0, 20.0],
        [-90.0, -30.0, -5.0, 0.0, 5.0, 19.0, 21.0, 60.0],
        [-100.0, -40.0, -8.0, 0.0, 8.0, 40.0, 100.0, 200.0],
    ),
    "saturating": (
        [60.0, 88.0, 100.0, 200.0, 60.0, 88.0, 100.0, 200.0],
        [0.5, 1.0, 2.0, 5.0, 19.0, 21.0, 40.0, 60.0],
        [-300.0, -120.0, -89.0, 0.0, 89.0, 120.0, 300.0, 1000.0],
    ),
}


def _extreme_gate_inputs(harness, spec, batch, regime):
    """Nominal q/k/v/state, but gate scalars pushed out to the exponent limits."""
    inp = harness["make_inputs"](spec, batch, 0)
    hv = spec.num_v_heads
    device = inp["A_log"].device
    a_log, a, b = _GATE_REGIMES[regime]

    def per_head(values, dtype):
        t = torch.tensor(values, device=device, dtype=torch.float32)
        return t.repeat(-(-hv // t.numel()))[:hv].to(dtype)

    inp["A_log"] = per_head(a_log, torch.float32)
    inp["dt_bias"] = torch.zeros_like(inp["dt_bias"])
    for key, values in (("a", a), ("b", b)):
        row = per_head(values, inp[key].dtype)
        inp[key] = row.view(1, 1, hv).expand(batch, 1, hv).contiguous()
    return inp


def _run_spec(harness, spec, inp, batch):
    values, cfg = harness["prepare"](spec, inp, batch, arch=ARCH)
    harness["launch"](harness["launcher_for"](spec, arch=ARCH), values, cfg)
    torch.cuda.synchronize()
    return values["out"], values["state"]


@requires_gfx942
@pytest.mark.parametrize("regime", sorted(_GATE_REGIMES))
def test_exp2_fast_matches_guarded_exp2_on_extreme_gates(harness, regime):
    """The same A/B where the missing range-reduction guard could actually bite.

    ``test_exp2_fast_matches_guarded_exp2`` uses the ordinary input
    distribution, where every exponent argument is small; that cannot tell the
    two lowerings apart. Here ``exp(A_log)`` reaches +inf, ``a + dt_bias``
    straddles the softplus threshold by +-70, and ``-b`` saturates the sigmoid,
    so an unguarded ``v_exp_f32`` has somewhere to diverge.
    """
    batch = 8
    base = _spec_for(batch)
    tol = harness["TOL"]
    results = {}
    for flag in (True, False):
        spec = dc.replace(base, use_exp2_fast=flag)
        inp = _extreme_gate_inputs(harness, spec, batch, regime)
        # The point of the case is the extreme gate; assert it is really there.
        assert torch.exp(inp["A_log"]).max().item() > 1e8, (
            "gate override did not reach the kernel inputs"
        )
        ref_out, ref_state = harness["ref_fp32"](spec, inp)
        out, state = _run_spec(harness, spec, inp, batch)
        state = state[inp["write_indices"].long()]
        assert torch.isfinite(out).all(), f"{regime}/{flag}: non-finite output"
        assert torch.isfinite(state).all(), f"{regime}/{flag}: non-finite state"
        results[flag] = (out.float(), state.float(), ref_out, ref_state)

    for flag, (out, state, ref_out, ref_state) in results.items():
        out_err = (out - ref_out).abs().max().item()
        state_err = (state - ref_state).abs().max().item()
        label = "exp2_fast " if flag else "exp2 guard"
        print(f"\n{regime} {label}: out_err={out_err:.3e} state_err={state_err:.3e}")
        assert out_err <= tol, f"{regime}/{flag}: output error {out_err:.3e}"
        assert state_err <= tol, f"{regime}/{flag}: state error {state_err:.3e}"

    fast_out, fast_state, _, _ = results[True]
    slow_out, slow_state, _, _ = results[False]
    out_gap = (fast_out - slow_out).abs().max().item()
    state_gap = (fast_state - slow_state).abs().max().item()
    print(f"{regime} fast-vs-guarded: out={out_gap:.3e} state={state_gap:.3e}")
    assert out_gap <= tol, f"{regime}: paths disagree on output by {out_gap:.3e}"
    assert state_gap <= tol, f"{regime}: paths disagree on state by {state_gap:.3e}"
