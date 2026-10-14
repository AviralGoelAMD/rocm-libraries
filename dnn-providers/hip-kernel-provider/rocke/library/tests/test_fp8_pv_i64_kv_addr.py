# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Native-fp8 PV cannot address a paged KV cache above the i32 bound.

The fp8 PV V loads use the 32-bit buffer voffset and ignore the 64-bit
per-block base, so ``use_fp8_mfma_pv`` with ``use_i64_kv_addr`` would read
zeros / wrapped rows. Both engines must reject the combination.
"""

from __future__ import annotations

import pytest

_BASE = dict(
    head_size=64,
    block_size=32,
    num_query_heads=32,
    num_kv_heads=32,
    dtype="bf16",
    use_sinks=False,
    sliding_window=0,
    has_softcap=False,
    kv_storage_dtype="fp8e4m3",
    use_fp8_mfma_pv=True,
)


def test_python_gfx950_rejects_fp8_pv_with_i64():
    from kernels.gfx950.attention_tiled_2d import UnifiedAttention2DTiledSpec

    UnifiedAttention2DTiledSpec(**_BASE)  # i32 addressing stays legal
    with pytest.raises(ValueError, match="use_fp8_mfma_pv does not support"):
        UnifiedAttention2DTiledSpec(**_BASE, use_i64_kv_addr=True)


def test_tuned_fp8_pv_spec_refuses_large_cache():
    """A tuned fp8-PV spec retargeted to a > 2 GiB cache fails loudly."""
    from dataclasses import replace

    from kernels.gfx950.attention_tiled_2d import UnifiedAttention2DTiledSpec

    spec = UnifiedAttention2DTiledSpec(**_BASE)
    with pytest.raises(ValueError, match="use_fp8_mfma_pv does not support"):
        replace(spec, use_i64_kv_addr=True)


def test_cpp_gfx950_rejects_fp8_pv_with_i64():
    try:
        import rocke_engine
    except ImportError:
        pytest.skip("rocke_engine (C++) not importable")
    rocke_engine.gfx950_attention_tiled_2d_lower_llvm(dict(_BASE), "gfx950")
    with pytest.raises(RuntimeError, match="use_fp8_mfma_pv does not support"):
        rocke_engine.gfx950_attention_tiled_2d_lower_llvm(
            dict(_BASE, use_i64_kv_addr=True), "gfx950"
        )
