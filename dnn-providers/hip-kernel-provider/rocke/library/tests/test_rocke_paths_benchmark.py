# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
import torch

import benchmarks.gfx942.attention.prefill.rocke_vs_aiter.rocke_paths as rocke_paths


def test_configs_match_the_ten_row_benchmark_cohort() -> None:
    assert [
        (config.id, config.B, config.S, config.Hq, config.Hkv)
        for config in rocke_paths.CONFIGS
    ] == [
        (1, 1, 4096, 32, 8),
        (2, 1, 4096, 32, 16),
        (3, 1, 8192, 32, 8),
        (4, 1, 8192, 32, 16),
        (5, 1, 16384, 32, 8),
        (6, 16, 4096, 32, 8),
        (7, 16, 8192, 32, 8),
        (8, 16, 4096, 32, 16),
        (9, 64, 4096, 32, 8),
        (10, 64, 8192, 32, 8),
    ]


def test_identity_paged_kv_preserves_each_cpu_token() -> None:
    k = torch.arange(2 * 128 * 2 * 4, dtype=torch.float32).reshape(2, 128, 2, 4)
    v = k + k.numel()

    k_cache, v_cache, block_table = rocke_paths.make_identity_paged_kv(k, v)

    assert k_cache.shape == (4, 64, 2, 4)
    assert v_cache.shape == (4, 64, 2, 4)
    assert torch.equal(k_cache.reshape_as(k), k)
    assert torch.equal(v_cache.reshape_as(v), v)
    assert block_table.dtype is torch.int32
    assert block_table.device == k.device
    assert torch.equal(block_table, torch.tensor([[0, 1], [2, 3]], dtype=torch.int32))


def test_identity_paged_kv_materializes_noncontiguous_inputs() -> None:
    k = torch.arange(2 * 128 * 2 * 8, dtype=torch.float32).reshape(2, 128, 2, 8)[
        ..., ::2
    ]
    v = torch.arange(2 * 128 * 2 * 8, dtype=torch.float32).add_(2 * 128 * 2 * 8).reshape(
        2, 128, 2, 8
    )[..., ::2]

    k_cache, v_cache, _ = rocke_paths.make_identity_paged_kv(k, v)

    assert not k.is_contiguous()
    assert not v.is_contiguous()
    assert k_cache.is_contiguous()
    assert v_cache.is_contiguous()
    assert torch.equal(k_cache.reshape_as(k), k)
    assert torch.equal(v_cache.reshape_as(v), v)


def test_identity_paged_kv_rejects_kv_on_different_devices() -> None:
    k = torch.empty(2, 128, 2, 4)
    v = torch.empty(2, 128, 2, 4, device="meta")

    with pytest.raises(ValueError, match="same device"):
        rocke_paths.make_identity_paged_kv(k, v)


def test_identity_paged_kv_rejects_mismatched_shapes() -> None:
    k = torch.empty(2, 128, 2, 4)
    v = torch.empty(2, 128, 2, 5)

    with pytest.raises(ValueError, match="identical"):
        rocke_paths.make_identity_paged_kv(k, v)


def test_identity_paged_kv_rejects_partial_page() -> None:
    k = torch.empty(2, 65, 2, 4)

    with pytest.raises(ValueError, match="divisible"):
        rocke_paths.make_identity_paged_kv(k, k)
