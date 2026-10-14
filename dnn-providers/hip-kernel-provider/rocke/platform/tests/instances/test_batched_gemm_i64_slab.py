# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Batched universal GEMM folds the batch slab into a 64-bit base pointer.

``batch * stride`` in i32 elements wraps once a tensor passes 2^31 elements,
so the batched kernel must offset A/B/C by a 64-bit byte product instead.
"""

from __future__ import annotations

import pytest

from rocke.helpers.compile import _lower_llvm_via_backend
from rocke.instances.common.batched_gemm import check_batched_strides_fit
from rocke.instances.common.gemm_universal import (
    DataSpec,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
)


def _ll(batched: bool) -> str:
    spec = UniversalGemmSpec(
        name="slab64",
        tile=TileSpec(256, 256, 64, 2, 2, 1, 32, 32, 16),
        trait=TraitSpec(pipeline="compv4", epilogue="default"),
        data=DataSpec(dtype_a="bf16", dtype_b="bf16", dtype_c="bf16"),
        batched=batched,
    )
    k = build_universal_gemm(spec, arch="gfx950")
    return _lower_llvm_via_backend(k, arch="gfx950", backend="python", spec=None)


def test_batched_offsets_are_i64_byte_bases():
    ll = _ll(batched=True)
    for p in ("%A", "%B", "%C"):
        assert f"getelementptr inbounds i8, ptr addrspace(1) {p}, i64" in ll, p
    for s in ("stride_a", "stride_b", "stride_c"):
        assert f"zext i32 %{s} to i64" in ll, s


def test_non_batched_has_no_slab_fold():
    ll = _ll(batched=False)
    assert "getelementptr inbounds i8, ptr addrspace(1) %A, i64" not in ll


def test_strides_guard():
    # 136 batches of 4096^2 crosses 2^31 elements: now legal.
    check_batched_strides_fit(4096, 4096, 4096, 4096 * 4096, 4096 * 4096, 4096 * 4096)
    with pytest.raises(ValueError, match="outside"):
        check_batched_strides_fit(16, 16, 16, -1, 256, 256)
    with pytest.raises(ValueError, match="outside"):
        check_batched_strides_fit(16, 16, 16, 1 << 32, 256, 256)
    with pytest.raises(ValueError, match="within-slab"):
        check_batched_strides_fit(1 << 16, 1 << 16, 16, 1 << 20, 1 << 20, 1 << 31)
