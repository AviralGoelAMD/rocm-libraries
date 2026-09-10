# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Golden LLVM-IR stability test for the GDN decode kernel on gfx942.

The gfx950 sibling (``test_gdn_decode_golden.py``) documents why a golden IR
hash is worth having: it catches an emitter refactor that changes the emitted
code without making it wrong, which no numeric test can see.

This file is the gfx942 arm of that check. The gfx942 tile table it covers is
**measured on gfx942 silicon**, not inherited: every tile the emitter accepts
was enumerated, correctness-gated, and timed. If the table is ever re-tuned and
the tiles move, these hashes must move with them, in the same change, out loud.

Measured note, so nobody reads more into a match than is there: for the cases
the two arches share, the lowered IR for gfx942 is **byte-identical** to gfx950.
The arch does not reach the emitter or the Python lowerer; it enters later, at
assembly. (The ``tuned_*`` case ids no longer overlap between the arches, since
each now carries its own band names, so the identity claim covers the shared
cases only.) This fixture is therefore a *drift detector for the gfx942
configuration set*, not evidence that anything arch-specific happens here. The
arch-specific evidence is the assembled ISA, not this file.

Re-record when a change to the emitted code is intended::

    python3 library/tests/test_gdn_decode_gfx942_golden.py --write

Lowering needs no GPU and no comgr, so this runs anywhere.
"""

from __future__ import annotations

import dataclasses as dc
import hashlib
import json
import sys
from pathlib import Path

_GOLDEN = (
    Path(__file__).resolve().parent / "golden" / "gdn_decode_gfx942_ir_sha256.json"
)
_FLAVORS = ("llvm20", "llvm22", "llvm23")
_ARCH = "gfx942"
_SCHEMA = "gdn_decode_gfx942.ir_golden_sha256/v1"

# Pin the library root ahead of everything on sys.path so that running this file
# directly does not let tests/dispatch/ shadow the real library/dispatch package.
_LIB_ROOT = str(Path(__file__).resolve().parent.parent)
if sys.path and sys.path[0] != _LIB_ROOT:
    sys.path.insert(0, _LIB_ROOT)


def _cases():
    """case id -> zero-arg builder returning a KernelDef.

    Covers the default spec, the reference path, and every tile the gfx942
    dispatcher can select, so a change to any shipped configuration is visible.
    """
    from dispatch.gdn.gfx942 import _TUNED_TILES
    from kernels.common.gdn_decode import GdnDecodeSpec, build_gdn_decode

    def build(**overrides):
        spec = dc.replace(GdnDecodeSpec(), **overrides)
        return lambda: build_gdn_decode(spec, arch=_ARCH)

    cases = {
        "default": build(),
        "simple": build(simple=True),
        "no_l2norm": build(use_qk_l2norm=False),
    }
    for _, tile, spec_id in _TUNED_TILES:
        cases[f"tuned_{spec_id}"] = build(
            num_warps=tile[0],
            warp_threads_k=tile[1],
            blocks_per_v_dim=tile[2],
        )
    return cases


def _current_flavor():
    from rocke.core.lower_llvm import _resolve_llvm_flavor

    return _resolve_llvm_flavor()


def _sha_for(build, flavor):
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    llvm = _lower_kernel_to_llvm_python(build(), arch=_ARCH, llvm_flavor=flavor)
    data = llvm.encode("utf-8")
    return hashlib.sha256(data).hexdigest(), len(data)


def _build_doc():
    doc = {"schema": _SCHEMA, "flavors": {}}
    for flavor in _FLAVORS:
        cases = {}
        for cid, build in _cases().items():
            try:
                sha, nbytes = _sha_for(build, flavor)
                cases[cid] = {"sha256": sha, "bytes": nbytes}
            except Exception as exc:  # pragma: no cover - diagnostic only
                cases[cid] = {"error": str(exc)[:160]}
        doc["flavors"][flavor] = {"cases": cases}
    return doc


def test_gdn_decode_gfx942_ir_matches_golden():
    import pytest

    if not _GOLDEN.exists():
        pytest.skip("gdn_decode gfx942 golden fixture missing; generate with --write")
    golden = json.loads(_GOLDEN.read_text())
    flavor = _current_flavor()
    recorded = golden.get("flavors", {}).get(flavor)
    if not recorded:
        pytest.skip(f"no gdn_decode gfx942 golden recorded for llvm flavor {flavor!r}")
    drift = []
    for cid, build in _cases().items():
        want = recorded["cases"].get(cid, {}).get("sha256")
        if want is None:
            continue
        got, _ = _sha_for(build, flavor)
        if got != want:
            drift.append(f"{cid}: {want} -> {got}")
    assert not drift, (
        "gdn_decode gfx942 IR drift vs golden (re-record with --write if intended):\n  "
        + "\n  ".join(drift)
    )


def test_every_shipped_gfx942_configuration_is_recorded():
    """A new tuned tile must arrive with a golden entry, not silently uncovered."""
    import pytest

    if not _GOLDEN.exists():
        pytest.skip("gdn_decode gfx942 golden fixture missing; generate with --write")
    golden = json.loads(_GOLDEN.read_text())
    flavor = _current_flavor()
    recorded = golden.get("flavors", {}).get(flavor)
    if not recorded:
        pytest.skip(f"no gdn_decode gfx942 golden recorded for llvm flavor {flavor!r}")
    missing = sorted(set(_cases()) - set(recorded["cases"]))
    assert not missing, f"configurations with no golden entry: {missing}"


if __name__ == "__main__":
    if "--write" in sys.argv:
        _GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        _GOLDEN.write_text(json.dumps(_build_doc(), indent=2, sort_keys=True) + "\n")
        print(f"wrote {_GOLDEN}")
    else:
        test_gdn_decode_gfx942_ir_matches_golden()
        test_every_shipped_gfx942_configuration_is_recorded()
