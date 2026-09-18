# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared workload and paged-KV primitives for the gfx942 attention benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import torch

PAGE_SIZE = 64
MAX_ABS_TOL = 4e-2


@dataclass(frozen=True)
class Config:
    id: int
    B: int
    S: int
    Hq: int
    Hkv: int


CONFIGS = (
    Config(1, 1, 4096, 32, 8),
    Config(2, 1, 4096, 32, 16),
    Config(3, 1, 8192, 32, 8),
    Config(4, 1, 8192, 32, 16),
    Config(5, 1, 16384, 32, 8),
    Config(6, 16, 4096, 32, 8),
    Config(7, 16, 8192, 32, 8),
    Config(8, 16, 4096, 32, 16),
    Config(9, 64, 4096, 32, 8),
    Config(10, 64, 8192, 32, 8),
)


def make_identity_paged_kv(
    k: torch.Tensor, v: torch.Tensor, page_size: int = PAGE_SIZE
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack dense BSHD K/V tensors into identity-ordered page caches."""
    if k.shape != v.shape:
        raise ValueError("k and v must have identical [B, S, Hkv, D] shapes")
    if k.ndim != 4:
        raise ValueError("k and v must be rank-4 [B, S, Hkv, D] tensors")
    if page_size <= 0:
        raise ValueError("page_size must be positive")

    batch, sequence_length, kv_heads, head_dim = k.shape
    if sequence_length % page_size:
        raise ValueError("sequence length must be divisible by page_size")

    pages_per_sequence = sequence_length // page_size
    cache_shape = (batch * pages_per_sequence, page_size, kv_heads, head_dim)
    k_cache = k.reshape(cache_shape)
    v_cache = v.reshape(cache_shape)
    block_table = torch.arange(
        batch * pages_per_sequence, device=k.device, dtype=torch.int32
    ).reshape(batch, pages_per_sequence)
    return k_cache, v_cache, block_table
