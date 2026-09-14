# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-side input-validation guard for the GDN decode kernel, without a GPU.

``prepare`` rejects inputs that would make the kernel read or write outside the
state pool. The decode kernel bounds-checks nothing on device beyond the ``-1``
skip sentinel, so this host guard *is* the memory-safety contract. The checks
raise before any launch, so they are pure host logic that runs on a CPU box --
which is exactly where a "this guard must not be silently dropped" regression
test belongs, rather than behind the on-device ``gpu`` gate.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch required (CPU build is fine)")

from builders.gfx950.gdn.gdn_decode import make_inputs, prepare
from kernels.gfx950.gdn_decode import GdnDecodeSpec

DEVICE = "cpu"


def test_out_of_range_index_is_rejected():
    """An index past the pool depth is an OOB access; prepare() must refuse it.

    ``-1`` stays legal (skip); any other out-of-pool value is rejected before a
    launch can touch it, for both the read and the write index.
    """
    spec = GdnDecodeSpec()
    batch = 8
    pool_depth = make_inputs(spec, batch, device=DEVICE)["state"].shape[0]

    for name, bad in (
        ("read_indices", pool_depth),  # == depth: the first OOB slot
        ("write_indices", pool_depth + 5),
        ("read_indices", -2),  # below the -1 skip sentinel
    ):
        inp = make_inputs(spec, batch, device=DEVICE)
        inp[name][0] = bad
        with pytest.raises(ValueError, match="out of range"):
            prepare(spec, inp, batch)


def test_the_skip_sentinel_is_accepted():
    """``-1`` marks an idle continuous-batching slot and must pass the guard."""
    spec = GdnDecodeSpec()
    batch = 8
    inp = make_inputs(spec, batch, device=DEVICE)
    inp["read_indices"][1::2] = -1
    inp["write_indices"][1::2] = -1
    prepare(spec, inp, batch)  # must not raise


def test_wrong_state_head_dims_are_rejected():
    """A state pool whose head dims disagree with the spec is a shape bug, and
    the check is sync-free so it runs regardless of the value-range flag."""
    spec = GdnDecodeSpec()
    batch = 8
    inp = make_inputs(spec, batch, device=DEVICE)
    # Drop a slice of the K dim so the pool no longer matches the spec.
    inp["state"] = inp["state"][..., :-8].contiguous()
    with pytest.raises(ValueError, match="head dims"):
        prepare(spec, inp, batch, validate_indices=False)


def test_validate_indices_flag_skips_the_range_check():
    """The value-range check reads the index extrema (a device sync on GPU), so
    it is flag-gated for the hot path.

    With it off, prepare() does not inspect the values and an out-of-pool index
    slips past; the sync-free shape checks still run. This pins the flag
    contract so the default-on guard cannot be silently lost.
    """
    spec = GdnDecodeSpec()
    batch = 8
    inp = make_inputs(spec, batch, device=DEVICE)
    inp["read_indices"][0] = inp["state"].shape[0]  # OOB, but unchecked
    prepare(spec, inp, batch, validate_indices=False)


@pytest.mark.parametrize(
    "key,mutate,match",
    [
        ("a", lambda x: x[..., :1], r"a shape"),
        (
            "a",
            lambda x: x.transpose(-1, -2).contiguous().transpose(-1, -2),
            r"a.*contiguous",
        ),
        ("dt_bias", lambda x: x[..., :1], r"dt_bias shape"),
        ("dt_bias", lambda x: x.to(torch.bfloat16), r"dt_bias.*float32"),
        (
            "dt_bias",
            lambda x: x.transpose(-1, -2).contiguous().transpose(-1, -2),
            r"dt_bias.*contiguous",
        ),
    ],
)
def test_kda_gate_input_contract_is_rejected_before_launch(key, mutate, match):
    """KDA widens the gate buffers; legacy/malformed allocations must not launch.

    The emitter vector-loads ``a`` as ``[B,1,HV,DK]`` and ``dt_bias`` as f32
    ``[HV,DK]`` using spec-derived offsets. A legacy GDN-shaped or strided
    allocation is smaller/differently laid out than that compiled range.
    """
    import dataclasses as dc

    spec = dc.replace(GdnDecodeSpec(), gate_kind="kda")
    batch = 2
    inp = make_inputs(spec, batch, device=DEVICE)
    inp[key] = mutate(inp[key])

    with pytest.raises(ValueError, match=match):
        prepare(spec, inp, batch)


@pytest.mark.parametrize("key", ["a", "dt_bias"])
def test_kda_gate_inputs_must_match_query_device(key):
    """The launch contract rejects gate buffers on a different device."""
    import dataclasses as dc

    spec = dc.replace(GdnDecodeSpec(), gate_kind="kda")
    inp = make_inputs(spec, batch=2, device=DEVICE)
    inp[key] = torch.empty_like(inp[key], device="meta")

    with pytest.raises(ValueError, match=rf"{key} device"):
        prepare(spec, inp, batch=2)


def test_gdn_gate_input_contract_stays_scalar():
    """The KDA checks must not reject the existing scalar GDN ABI."""
    spec = GdnDecodeSpec()
    inp = make_inputs(spec, batch=2, device=DEVICE)

    assert tuple(inp["a"].shape) == (2, 1, spec.num_v_heads)
    assert tuple(inp["dt_bias"].shape) == (spec.num_v_heads,)
    prepare(spec, inp, batch=2)
