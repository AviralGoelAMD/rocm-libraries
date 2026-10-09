# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Code-object resource gates for the exact KDA tiles dispatch ships on gfx950."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from dispatch.gdn import GdnDecodeRequest, dispatch_gdn_decode

ARCH = "gfx950"
# (state dtype, expected candidate, expected tile): the static KDA ``auto``
# default for each state width. Every other registered KDA tile is compiled for
# scratch by test_gdn_decode_spec.py.
CASES = (
    ("bf16", "kda_nw4_wtk16_bpv4", (4, 16, 4)),
    ("f16", "kda_nw4_wtk16_bpv4", (4, 16, 4)),
    ("f32", "kda_nw8_wtk16_bpv4", (8, 16, 4)),
)


def _resources_for(kernel):
    try:
        from rocke.analysis.isa import analyze_hsaco
        from rocke.helpers.compile import compile_kernel
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"comgr/resource tools unavailable: {exc}")

    try:
        artifact = compile_kernel(kernel, arch=ARCH, capture_ir_text=False)
    except ImportError as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"comgr toolchain unavailable: {exc}")

    with tempfile.NamedTemporaryFile(suffix=".hsaco") as fh:
        fh.write(bytes(artifact.hsaco))
        fh.flush()
        try:
            return analyze_hsaco(Path(fh.name)).resources
        except (FileNotFoundError, RuntimeError) as exc:  # pragma: no cover
            pytest.skip(f"HSACO introspection unavailable: {exc}")


def _assert_scratch_free(resources, *, spec_id: str, tile) -> None:
    assert resources.scratch_bytes is not None, "scratch metadata was not parsed"
    assert (
        resources.scratch_bytes == 0
    ), f"{spec_id} tile {tile} spills {resources.scratch_bytes} bytes to scratch"


def test_scratch_gate_rejects_nonzero_metadata():
    """Mutation-level proof that the gate detects a spilling code object."""
    from rocke.analysis.isa import ResourceInfo

    with pytest.raises(AssertionError, match="spills 16 bytes"):
        _assert_scratch_free(
            ResourceInfo(scratch_bytes=16),
            spec_id="mutated",
            tile=(1, 1, 1),
        )


@pytest.mark.parametrize("state_dtype,expected_spec_id,expected_tile", CASES)
def test_dispatched_kda_tile_is_scratch_free(
    state_dtype, expected_spec_id, expected_tile
):
    """Compile each auto default and reject register spills."""
    result = dispatch_gdn_decode(
        GdnDecodeRequest(
            batch=8,
            arch=ARCH,
            gate_kind="kda",
            num_k_heads=32,
            num_v_heads=32,
            head_k_dim=128,
            head_v_dim=128,
            state_dtype=state_dtype,
        )
    )
    tile = (
        result.spec.num_warps,
        result.spec.warp_threads_k,
        result.spec.blocks_per_v_dim,
    )
    assert result.candidate.spec_id == expected_spec_id
    assert tile == expected_tile

    resources = _resources_for(result.build())
    _assert_scratch_free(resources, spec_id=expected_spec_id, tile=tile)


@pytest.mark.parametrize("gate_kind", ["gdn", "kda"])
@pytest.mark.parametrize("state_dtype", ["bf16", "f32"])
def test_dispatched_fused_default_is_scratch_free(gate_kind, state_dtype):
    """The static fused (conv + out-norm) ``auto`` tile compiles without spills."""
    from dispatch.gdn.gfx950 import FUSED_DEFAULT_TILES

    result = dispatch_gdn_decode(
        GdnDecodeRequest(
            batch=8,
            arch=ARCH,
            gate_kind=gate_kind,
            num_k_heads=32,
            num_v_heads=32,
            state_dtype=state_dtype,
            fuse_conv=True,
            fuse_out_norm=True,
        )
    )
    tile = (
        result.spec.num_warps,
        result.spec.warp_threads_k,
        result.spec.blocks_per_v_dim,
    )
    assert tile == FUSED_DEFAULT_TILES[(gate_kind, state_dtype)]
    resources = _resources_for(result.build())
    _assert_scratch_free(resources, spec_id=result.candidate.spec_id, tile=tile)
