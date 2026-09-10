# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-side input-validation guard for the GDN decode kernel, without a GPU.

``prepare`` rejects inputs that would make the kernel read or write outside the
state pool. The decode kernel bounds-checks nothing on device beyond the ``-1``
skip sentinel, so this host guard *is* the memory-safety contract. The checks
raise before any launch, so they are pure host logic that runs on a CPU box --
which is exactly where a "this guard must not be silently dropped" regression
test belongs, rather than behind the on-device ``gpu`` gate.

It also pins the *arch threading* contract of the same driver. The driver used
to carry a module-level ``_ARCH = "gfx950"``, so ``check``/``bench`` compiled
and validated against gfx950 no matter which architecture the caller wanted.
Those tests need no torch at all, so they are not gated behind it.
"""

from __future__ import annotations

import contextlib
import importlib.util
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

from kernels.common.gdn_decode import GdnDecodeSpec

_HAVE_TORCH = importlib.util.find_spec("torch") is not None

# Per-test rather than module-level: the arch-threading tests at the bottom of
# this file touch no tensor API, and a module-level ``importorskip`` would skip
# them on a torch-free CPU box -- which is exactly where they are meant to run.
requires_torch = pytest.mark.skipif(
    not _HAVE_TORCH, reason="torch required (CPU build is fine)"
)

if _HAVE_TORCH:
    from builders.gfx950.gdn.gdn_decode import make_inputs, prepare

DEVICE = "cpu"


@requires_torch
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


@requires_torch
def test_the_skip_sentinel_is_accepted():
    """``-1`` marks an idle continuous-batching slot and must pass the guard."""
    spec = GdnDecodeSpec()
    batch = 8
    inp = make_inputs(spec, batch, device=DEVICE)
    inp["read_indices"][1::2] = -1
    inp["write_indices"][1::2] = -1
    prepare(spec, inp, batch)  # must not raise


@requires_torch
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


@requires_torch
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


# ---------------------------------------------------------------------------
# Arch threading. A gfx942 build must validate and compile against gfx942, not
# against whatever the host driver happens to default to.
# ---------------------------------------------------------------------------

_DRIVER_SRC = (
    Path(__file__).resolve().parents[1]
    / "builders"
    / "gfx950"
    / "gdn"
    / "gdn_decode.py"
)


def _load_driver():
    """Load a private copy of the host driver, stubbing torch when absent.

    Every path exercised below reaches the builder without touching a single
    tensor API, so on a torch-free box a bare stub module is enough to satisfy
    the driver's top-level ``import torch``.

    The copy runs under a private module name and is never registered as
    ``builders.gfx950.gdn.gdn_decode``. That keeps the stub from leaking into a
    real-torch import elsewhere in the session, and keeps the copy's
    ``_LAUNCHER_CACHE`` from handing a fake launcher to another test.
    """
    name = "_gdn_decode_driver_under_test"
    spec = importlib.util.spec_from_file_location(name, _DRIVER_SRC)
    module = importlib.util.module_from_spec(spec)
    stubbed = not _HAVE_TORCH
    if stubbed:
        sys.modules["torch"] = types.ModuleType("torch")
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[name]
        if stubbed:
            del sys.modules["torch"]
    return module


class _NullTensor:
    """Stands in for every tensor the driver's error arithmetic touches.

    The arithmetic itself is covered by the on-device numeric test; what is
    under test here is the arch handed to the launcher, so every operation
    returns the same object and both errors come out as ``0.0``.
    """

    def float(self):
        return self

    def long(self):
        return self

    def abs(self):
        return self

    def max(self):
        return self

    def item(self):
        return 0.0

    def __sub__(self, other):
        return self

    def __getitem__(self, key):
        return self


@contextlib.contextmanager
def _fake_toolchain(drv, seen):
    """Record the arch at each arch-consuming boundary of ``launcher_for``.

    ``is_valid_spec`` and ``build_gdn_decode`` run for real -- so the arch has
    to be one the emitter actually accepts -- while ``compile_kernel`` and
    ``KernelLauncher`` are replaced, because those two need comgr and a GPU.
    """
    real_is_valid_spec = drv.is_valid_spec
    real_build = drv.build_gdn_decode

    def recording_is_valid_spec(spec, arch):
        seen["is_valid_spec"] = arch
        return real_is_valid_spec(spec, arch=arch)

    def recording_build(spec, arch):
        seen["build_gdn_decode"] = arch
        return real_build(spec, arch=arch)

    def fake_compile(kernel_def, arch):
        seen["compile_kernel"] = arch
        return types.SimpleNamespace(hsaco=b"", kernel_name="gdn_decode")

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(drv, "is_valid_spec", recording_is_valid_spec)
        )
        stack.enter_context(mock.patch.object(drv, "build_gdn_decode", recording_build))
        stack.enter_context(mock.patch.object(drv, "compile_kernel", fake_compile))
        stack.enter_context(
            mock.patch.object(drv, "KernelLauncher", lambda **kw: object())
        )
        yield


def _cpu_torch():
    """The only torch surface ``run``/``bench`` need once tensors are faked."""
    return types.SimpleNamespace(
        cuda=types.SimpleNamespace(synchronize=lambda: None),
    )


def test_launcher_for_forwards_the_requested_arch():
    """gfx942 in, gfx942 all the way to the validator and the emitter."""
    drv = _load_driver()
    seen = {}
    with _fake_toolchain(drv, seen):
        drv.launcher_for(GdnDecodeSpec(), arch="gfx942")

    assert seen == {
        "is_valid_spec": "gfx942",
        "build_gdn_decode": "gfx942",
        "compile_kernel": "gfx942",
    }


def test_launcher_for_still_defaults_to_gfx950():
    """Existing gfx950 callers pass no arch and must be unaffected."""
    drv = _load_driver()
    seen = {}
    with _fake_toolchain(drv, seen):
        drv.launcher_for(GdnDecodeSpec())

    assert set(seen.values()) == {"gfx950"}
    assert drv.DEFAULT_ARCH == "gfx950"


def test_no_module_level_arch_constant_survives():
    """``_ARCH`` was replaced by an explicit parameter; a surviving constant
    would mean some path can still quietly default to gfx950."""
    drv = _load_driver()
    assert not hasattr(drv, "_ARCH")


def test_check_forwards_the_requested_arch_to_the_launcher():
    """``check`` used to call ``launcher_for(spec)`` with no arch, so a gfx942
    correctness run silently graded the gfx950 kernel."""
    drv = _load_driver()
    null = _NullTensor()
    seen = []

    def recording_launcher_for(spec, arch):
        seen.append(arch)
        return "launcher"

    with mock.patch.object(drv, "launcher_for", recording_launcher_for), mock.patch.object(
        drv, "make_inputs", lambda spec, batch, seed=0: {"write_indices": null}
    ), mock.patch.object(
        drv, "ref_fp32", lambda spec, inp: (null, null)
    ), mock.patch.object(
        drv, "run", lambda spec, inp, launcher, batch, arch=None: (null, null)
    ):
        result = drv.check(GdnDecodeSpec(), 4, arch="gfx942")

    assert seen == ["gfx942"]
    # The pair must survive. The in-place bf16 state write carries roughly 30x
    # the output error, so collapsing the two with max() would hide a state bug.
    assert result == (0.0, 0.0)


def test_bench_forwards_the_requested_arch_to_the_launcher():
    """Same defaulting bug as ``check``, but it times the wrong kernel."""
    drv = _load_driver()
    seen = []

    def recording_launcher_for(spec, arch):
        seen.append(arch)
        return "launcher"

    with mock.patch.object(drv, "launcher_for", recording_launcher_for), mock.patch.object(
        drv, "make_inputs", lambda spec, batch: {}
    ), mock.patch.object(
        drv, "prepare", lambda spec, inp, batch, arch=None: ({}, None)
    ), mock.patch.object(
        drv, "launch", lambda launcher, values, cfg: None
    ), mock.patch.object(
        drv, "torch", _cpu_torch()
    ):
        drv.bench(GdnDecodeSpec(), 4, reps=1, arch="gfx942")

    assert seen == ["gfx942"]


def test_prepare_validates_against_the_arch_it_is_given():
    """``prepare`` freezes the launch geometry, so it must judge the spec by the
    arch the caller named. An arch the emitter does not know is rejected there,
    before a single byte is allocated."""
    drv = _load_driver()
    with pytest.raises(ValueError, match="gfx000"):
        drv.prepare(GdnDecodeSpec(), {}, 4, arch="gfx000")


def test_run_forwards_the_arch_to_prepare():
    drv = _load_driver()
    seen = []

    def recording_prepare(spec, inp, batch, arch=None):
        seen.append(arch)
        return {"out": 1, "state": 2}, None

    with mock.patch.object(drv, "prepare", recording_prepare), mock.patch.object(
        drv, "launch", lambda launcher, values, cfg: None
    ), mock.patch.object(drv, "torch", _cpu_torch()):
        drv.run(GdnDecodeSpec(), {}, "launcher", 4, arch="gfx942")

    assert seen == ["gfx942"]
