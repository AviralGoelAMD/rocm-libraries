# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Golden LLVM-IR stability test for the GDN decode kernel.

Hashes the lowered IR for a fixed set of specs and compares against a recorded
fixture. This catches a class of change nothing else does: a refactor of the
emitter that alters the generated code *without* making it wrong. The numeric
test would still pass, because the kernel is still correct -- just different.

The two checks are complementary, not redundant:

* the numeric test catches a kernel that is **wrong**, but not one that merely
  **changed**;
* this test catches a kernel that **changed**, but says nothing about whether
  either version is correct.

A failure here is not automatically a bug. It is a claim that the emitted code
moved, and it demands an answer: intended, or not? When intended, re-record in
the same change so the diff states it out loud::

    python3 library/tests/test_gdn_decode_golden.py --write

Each arch that registers GDN decode has its own fixture, covering that arch's
legal registered tiles. KDA decode is gfx950-only, so only the gfx950 fixture
carries KDA cases.

Lowering needs no GPU and no comgr, so this runs anywhere.
"""

from __future__ import annotations

import dataclasses as dc
import hashlib
import json
import sys
from pathlib import Path

import pytest

_ARCHES = ("gfx942", "gfx950")
_GOLDENS = {
    arch: Path(__file__).resolve().parent
    / "golden"
    / f"gdn_decode_{arch}_ir_sha256.json"
    for arch in _ARCHES
}
_FLAVORS = ("llvm20", "llvm22", "llvm23")

# Pin the library root ahead of everything on sys.path so that running this file
# directly does not let tests/dispatch/ shadow the real library/dispatch package.
_LIB_ROOT = str(Path(__file__).resolve().parent.parent)
if sys.path and sys.path[0] != _LIB_ROOT:
    sys.path.insert(0, _LIB_ROOT)


def _cases(arch):
    """case id -> zero-arg builder returning a KernelDef.

    Covers the default spec, the reference path, and every legal registered
    tile, so a change to any selectable configuration is visible.
    """
    from dispatch.gdn import GdnDecodeRequest, dispatch_gdn_decode_all
    from dispatch.gdn.gfx950 import _TUNED_TILES_KDA
    from kernels.common.gdn_decode import GdnDecodeSpec, build_gdn_decode

    def build(**overrides):
        spec = dc.replace(GdnDecodeSpec(), **overrides)
        return lambda: build_gdn_decode(spec, arch=arch)

    cases = {
        "default": build(),
        "simple": build(simple=True),
        "no_l2norm": build(use_qk_l2norm=False),
    }
    if arch == "gfx950":
        # KDA gate kind. Pinned for the same reason the GDN cases are: the
        # per-channel gate is emitted code, and a refactor that changed it
        # without breaking it would pass every other test in the tree.
        cases["kda_default"] = build(gate_kind="kda")
        cases["kda_simple"] = build(gate_kind="kda", simple=True)
        cases["kda_raw_gate"] = build(gate_kind="kda", fuse_gate=False)
    request = GdnDecodeRequest(batch=16, arch=arch)
    for result in dispatch_gdn_decode_all(request):
        cases[f"registered_{result.candidate.spec_id}"] = (
            lambda spec=result.spec: build_gdn_decode(spec, arch=arch)
        )
    if arch == "gfx950":
        for _, tile, spec_id in _TUNED_TILES_KDA:
            cases[f"tuned_{spec_id}"] = build(
                gate_kind="kda",
                num_warps=tile[0],
                warp_threads_k=tile[1],
                blocks_per_v_dim=tile[2],
            )
    return cases


def _current_flavor():
    from rocke.core.lower_llvm import _resolve_llvm_flavor

    return _resolve_llvm_flavor()


def _sha_for(build, flavor, arch):
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    llvm = _lower_kernel_to_llvm_python(build(), arch=arch, llvm_flavor=flavor)
    data = llvm.encode("utf-8")
    return hashlib.sha256(data).hexdigest(), len(data)


def _build_doc(arch):
    doc = {"schema": f"gdn_decode_{arch}.ir_golden_sha256/v1", "flavors": {}}
    failures = []
    for flavor in _FLAVORS:
        cases = {}
        for cid, build in _cases(arch).items():
            try:
                sha, nbytes = _sha_for(build, flavor, arch)
            except Exception as exc:  # pragma: no cover - diagnostic only
                failures.append(f"{flavor}/{cid}: {exc}")
                continue
            cases[cid] = {"sha256": sha, "bytes": nbytes}
        doc["flavors"][flavor] = {"cases": cases}
    if failures:
        raise RuntimeError(
            "refusing to write golden fixture with lowering failures:\n  "
            + "\n  ".join(failures)
        )
    return doc


def _recorded(arch):
    golden = _GOLDENS[arch]
    if not golden.exists():
        pytest.skip(f"gdn_decode {arch} golden fixture missing; generate with --write")
    flavor = _current_flavor()
    recorded = json.loads(golden.read_text()).get("flavors", {}).get(flavor)
    if not recorded:
        pytest.skip(f"no gdn_decode {arch} golden recorded for llvm flavor {flavor!r}")
    return flavor, recorded


@pytest.mark.parametrize("arch", _ARCHES)
def test_gdn_decode_ir_matches_golden(arch):
    flavor, recorded = _recorded(arch)
    drift = []
    for cid, build in _cases(arch).items():
        entry = recorded["cases"].get(cid, {})
        want = entry.get("sha256")
        if not want:
            drift.append(f"{cid}: no sha256 recorded ({entry})")
            continue
        got, _ = _sha_for(build, flavor, arch)
        if got != want:
            drift.append(f"{cid}: {want} -> {got}")
    assert not drift, (
        f"gdn_decode {arch} IR drift vs golden (re-record with --write if "
        "intended):\n  " + "\n  ".join(drift)
    )


@pytest.mark.parametrize("arch", _ARCHES)
def test_every_shipped_configuration_is_recorded(arch):
    """A new tuned tile must arrive with a golden entry, not silently uncovered."""
    _, recorded = _recorded(arch)
    missing = sorted(
        cid for cid in _cases(arch) if not recorded["cases"].get(cid, {}).get("sha256")
    )
    assert not missing, f"configurations without a SHA-256: {missing}"


@pytest.mark.parametrize("arch", _ARCHES)
def test_gdn_decode_ir_cpp_python_byte_identity(arch, monkeypatch):
    """The C++ engine lowers every golden case to the same bytes as Python.

    The fixtures above pin the *Python* lowering; production defaults to the C++
    engine, so this is the other half over the same case set. There is no
    ``gdn_decode_emit.c`` mirror, so the comparison starts from the Python-built
    kernel's serialized IR. ``ROCKE_CPP_STRICT=1`` disables the silent Python
    fallback, so an absent or stale engine cannot pass. Failure handling follows
    ``test_attention_ir_cpp_parity.py``: an undeclared engine is a whole-run
    skip, an arch in ``CPP_UNPORTED_ARCHES`` a counted skip, anything else red.
    """
    from rocke.core.backend import BackendCoverageGap, BackendError
    from rocke.helpers.compile import _lower_llvm_via_backend

    monkeypatch.setenv("ROCKE_CPP_STRICT", "1")
    mismatched = []
    for cid, build in _cases(arch).items():
        kernel = build()
        py = _lower_llvm_via_backend(kernel, arch=arch, backend="python", spec=None)
        try:
            cpp = _lower_llvm_via_backend(kernel, arch=arch, backend="cpp", spec=None)
        except BackendCoverageGap as e:  # subclass of BackendError: catch first
            pytest.skip(f"{arch} is a declared C++ engine gap: {str(e)[:200]}")
        except BackendError as e:
            pytest.skip(f"C++ engine not importable: {str(e)[:200]}")
        if py != cpp:
            mismatched.append(cid)
    assert (
        not mismatched
    ), f"gdn_decode {arch} cpp/python IR byte-mismatch:\n  " + "\n  ".join(mismatched)


def _synthetic_fixture(monkeypatch, tmp_path):
    """Point the gfx950 golden at a fixture whose only case failed to lower."""
    monkeypatch.setattr(
        sys.modules[__name__], "_cases", lambda arch: {"default": object()}
    )
    fixture = tmp_path / "gdn_decode_gfx950_ir_sha256.json"
    fixture.write_text(
        json.dumps(
            {
                "flavors": {
                    _current_flavor(): {
                        "cases": {"default": {"error": "synthetic lowering failure"}}
                    }
                }
            }
        )
    )
    monkeypatch.setitem(_GOLDENS, "gfx950", fixture)


def test_golden_ir_check_rejects_entry_without_sha256(monkeypatch, tmp_path):
    _synthetic_fixture(monkeypatch, tmp_path)
    with pytest.raises(AssertionError, match="no sha256 recorded"):
        test_gdn_decode_ir_matches_golden("gfx950")


def test_config_coverage_rejects_entry_without_sha256(monkeypatch, tmp_path):
    _synthetic_fixture(monkeypatch, tmp_path)
    with pytest.raises(AssertionError, match="without a SHA-256"):
        test_every_shipped_configuration_is_recorded("gfx950")


def test_build_doc_refuses_lowering_failure(monkeypatch):
    monkeypatch.setattr(
        sys.modules[__name__], "_cases", lambda arch: {"default": object()}
    )

    def fail_lowering(*_):
        raise RuntimeError("synthetic lowering failure")

    monkeypatch.setattr(sys.modules[__name__], "_sha_for", fail_lowering)

    with pytest.raises(RuntimeError, match="refusing to write"):
        _build_doc("gfx950")


def test_gate_kind_actually_moves_the_ir():
    """A mutation check: the golden gate must be able to detect this change.

    "Golden untouched" only means something if the golden *could* have moved.
    Flipping gate_kind changes emitted code, so it must change both the IR hash
    and the kernel name -- otherwise the KDA cases above are pinning nothing and
    two different kernels would share one compile-cache entry.
    """
    from kernels.common.gdn_decode import GdnDecodeSpec, build_gdn_decode

    flavor = _current_flavor()
    gdn = GdnDecodeSpec()
    kda = dc.replace(gdn, gate_kind="kda")

    gdn_sha, _ = _sha_for(
        lambda: build_gdn_decode(gdn, arch="gfx950"), flavor, "gfx950"
    )
    kda_sha, _ = _sha_for(
        lambda: build_gdn_decode(kda, arch="gfx950"), flavor, "gfx950"
    )

    assert gdn_sha != kda_sha, "gate_kind did not change the emitted IR"
    assert gdn.kernel_name() != kda.kernel_name(), "gate_kind did not change the name"


def test_gdn_cases_carry_no_kda_marker():
    """Every pre-existing GDN entry must stay a GDN entry.

    Guards the additive claim from the fixture side: if a GDN case id ever
    starts resolving to a KDA spec, the "GDN goldens unchanged" evidence is
    quietly measuring the wrong kernel.

    KDA appears in an id two ways -- as a prefix for the hand-written cases
    (``kda_default``) and as an infix for the tuned ones (``tuned_kda_w128``,
    which inherits its gate kind from the spec id in the KDA table) -- so the
    split is on containment, not prefix.
    """
    from kernels.common.gdn_decode import GdnDecodeSpec

    assert GdnDecodeSpec().gate_kind == "gdn"
    ids = list(_cases("gfx950"))
    gdn_ids = [cid for cid in ids if "kda" not in cid]
    kda_ids = [cid for cid in ids if "kda" in cid]

    # The original GDN set: default, simple, no_l2norm + one per GDN tuned tile.
    assert len(gdn_ids) >= 7, f"expected the original GDN case set, got {gdn_ids}"
    assert kda_ids, "the KDA gate kind is unpinned"
    assert not set(gdn_ids) & set(kda_ids)
    assert not [
        cid for cid in _cases("gfx942") if "kda" in cid
    ], "gfx942 serves the GDN gate only; a KDA case there pins nothing shipped"


if __name__ == "__main__":
    if "--write" in sys.argv:
        for arch in _ARCHES:
            _GOLDENS[arch].parent.mkdir(parents=True, exist_ok=True)
            _GOLDENS[arch].write_text(
                json.dumps(_build_doc(arch), indent=2, sort_keys=True) + "\n"
            )
            print(f"wrote {_GOLDENS[arch]}")
    else:
        for arch in _ARCHES:
            test_gdn_decode_ir_matches_golden(arch)
            test_every_shipped_configuration_is_recorded(arch)
