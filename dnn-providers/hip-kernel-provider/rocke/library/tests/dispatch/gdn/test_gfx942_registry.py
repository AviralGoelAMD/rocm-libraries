# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU contract tests for the gfx942 GDN decode candidate registry."""

from __future__ import annotations

from dataclasses import asdict, replace

import pytest

from dispatch.gdn import (
    GdnDecodeRequest,
    dispatch_gdn_decode,
    dispatch_gdn_decode_all,
    gdn_candidates,
)
from dispatch.gdn.common import make_spec
from dispatch.gdn.gfx942 import ARCH, CONFIGURED_TILES, DEFAULT_TILE
from kernels.common.gdn_decode import is_valid_spec
from rocke.dispatch.core import stable_json_hash


def _req(batch: int = 16, **changes) -> GdnDecodeRequest:
    return GdnDecodeRequest(batch=batch, arch=ARCH, **changes)


def _tile(spec) -> tuple[int, int, int]:
    return spec.num_warps, spec.warp_threads_k, spec.blocks_per_v_dim


def test_configured_count_and_identity_are_stable_and_unique():
    candidates = tuple(
        candidate
        for candidate in gdn_candidates()
        if candidate.name.startswith(f"gdn_decode_{ARCH}_")
    )
    expected_ids = tuple(
        f"nw{nw}_wtk{wtk}_bpv{bpv}" for nw, wtk, bpv in CONFIGURED_TILES
    )
    assert len(CONFIGURED_TILES) == 180
    assert tuple(candidate.spec_id for candidate in candidates) == expected_ids
    assert len({candidate.name for candidate in candidates}) == 180


def test_legal_d128_count_is_54_and_validator_is_authority():
    req = _req()
    results = dispatch_gdn_decode_all(req)
    expected = {
        tile
        for tile in CONFIGURED_TILES
        if is_valid_spec(make_spec(req, tile), arch=ARCH)[0]
    }
    assert len(expected) == 54
    assert {_tile(result.spec) for result in results} == expected
    assert all(
        result.candidate.name.startswith(f"gdn_decode_{ARCH}_") for result in results
    )


def test_legal_compile_identities_are_unique():
    results = dispatch_gdn_decode_all(_req())
    identity_sets = (
        {result.candidate.name for result in results},
        {stable_json_hash(asdict(result.spec), n=16) for result in results},
        {result.kernel_id.compile_key for result in results},
    )
    assert all(len(identities) == 54 for identities in identity_sets)


def test_compile_keys_differ_from_gfx950_for_the_same_tile():
    """Same spec, different arch: the cache must never hand one arch's code to the other.

    The spec itself is arch-neutral, so its hash is shared; the arch must enter
    the compile key, or the two arches would collide on one cache entry. The
    tile is pinned so that a change of either arch's default cannot move it.
    """
    pinned = _req(algorithm="warp_tiled", spec_id="nw4_wtk16_bpv8")
    gfx942 = dispatch_gdn_decode(pinned)
    gfx950 = dispatch_gdn_decode(replace(pinned, arch="gfx950"))
    assert _tile(gfx942.spec) == _tile(gfx950.spec) == (4, 16, 8)
    assert gfx942.kernel_id.spec_hash == gfx950.kernel_id.spec_hash
    assert gfx942.kernel_id.compile_key != gfx950.kernel_id.compile_key
    assert gfx942.candidate.name != gfx950.candidate.name


def test_every_legal_pin_round_trips_and_illegal_pin_fails_loudly():
    req = _req()
    for expected in dispatch_gdn_decode_all(req):
        result = dispatch_gdn_decode(
            replace(
                req,
                algorithm=expected.candidate.algorithm,
                spec_id=expected.candidate.spec_id,
            )
        )
        assert result.candidate.name == expected.candidate.name
    with pytest.raises(ValueError, match="nw4_wtk16_bpv8"):
        dispatch_gdn_decode(
            _req(head_k_dim=64, algorithm="warp_tiled", spec_id="nw4_wtk16_bpv8")
        )


def test_auto_is_static_default_independent_of_batch():
    nw, wtk, bpv = DEFAULT_TILE
    for batch in (1, 16, 64, 256):
        result = dispatch_gdn_decode(_req(batch))
        assert result.candidate.name == f"gdn_decode_{ARCH}_nw{nw}_wtk{wtk}_bpv{bpv}"
        assert _tile(result.spec) == DEFAULT_TILE


def test_auto_admits_only_the_default_when_it_is_legal():
    """``auto`` is decided by admission, not by registry order.

    Asking each candidate directly takes priority order out of the picture:
    with a legal default, every other legal tile must refuse ``auto`` and
    still accept its own pin.
    """
    auto = _req()
    for result in dispatch_gdn_decode_all(auto):
        candidate = result.candidate
        admitted, why = candidate.admits(auto)
        if _tile(result.spec) == DEFAULT_TILE:
            assert admitted, why
            continue
        assert not admitted and "static GDN auto tile" in why, (candidate.name, why)
        pin = replace(auto, algorithm=candidate.algorithm, spec_id=candidate.spec_id)
        assert candidate.admits(pin)[0], candidate.name


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("use_qk_l2norm", "False"),
        ("head_k_dim", 128.9),
        ("batch", 16.7),
        ("num_v_heads", True),
        ("gate_kind", None),
    ],
)
def test_malformed_request_field_is_rejected_not_coerced(arch, field, value):
    """``bool("False")`` is ``True`` and ``int(128.9)`` is ``128``: a cast would
    serve a different kernel than the caller described, so dispatch refuses."""
    req = replace(_req(), arch=arch, **{field: value})
    with pytest.raises(ValueError, match=field):
        dispatch_gdn_decode(req)
    with pytest.raises(TypeError, match=field):
        make_spec(req, DEFAULT_TILE)


def test_d64_auto_falls_back_to_a_validator_approved_candidate():
    result = dispatch_gdn_decode(_req(head_k_dim=64))
    assert is_valid_spec(result.spec, arch=ARCH)[0]
    assert _tile(result.spec) != DEFAULT_TILE


def test_kda_gate_is_not_yet_implemented_on_gfx942():
    """A scope-out, not a hardware limit: the validator accepts a KDA spec on
    gfx942, so dispatch must say ``NOT_YET_IMPLEMENTED`` rather than imply the
    arch cannot run it."""
    kda = _req(gate_kind="kda")
    assert is_valid_spec(make_spec(kda, DEFAULT_TILE), arch=ARCH)[0]
    assert dispatch_gdn_decode_all(kda) == ()
    with pytest.raises(ValueError, match="NOT_YET_IMPLEMENTED"):
        dispatch_gdn_decode(kda)
