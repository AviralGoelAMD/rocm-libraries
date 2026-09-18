# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared workload and paged-KV primitives for the gfx942 attention benchmark."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
    if k.device != v.device:
        raise ValueError("k and v must be on the same device")
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
    k_cache = k.contiguous().view(cache_shape)
    v_cache = v.contiguous().view(cache_shape)
    block_table = torch.arange(
        batch * pages_per_sequence, device=k.device, dtype=torch.int32
    ).reshape(batch, pages_per_sequence)
    return k_cache, v_cache, block_table


_HEAD_SIZE = 128
_ROW_FIELDS = (
    "id",
    "B",
    "S",
    "Hq",
    "Hkv",
    "GQA",
    "ms",
    "tflops",
    "max_abs",
    "status",
    "kernel",
    "path",
    "settings",
    "reason",
)


def dense_spec(config: Config):
    """Resolve the shipped gfx942 BF16 D128 causal dense spec for ``config``."""
    from builders.gfx942.attention.prefill.attention_dense_prefill import (
        dense_request,
        resolve_dense_spec,
    )

    defaults = SimpleNamespace(
        persistent=None,
        num_persistent=None,
        persist_decode=None,
        block_n=None,
        waves_per_eu=None,
        interleave=None,
        lds_k_group_pad=None,
        sliding_window=None,
    )
    request = dense_request(
        defaults,
        batch=config.B,
        seqlen_q=config.S,
        seqlen_kv=config.S,
        num_query_heads=config.Hq,
        num_kv_heads=config.Hkv,
        head_size=_HEAD_SIZE,
        causal=True,
        dtype="bf16",
    )
    return resolve_dense_spec(request)


def _row(config: Config, *, path: str) -> dict[str, Any]:
    return {
        "id": config.id,
        "B": config.B,
        "S": config.S,
        "Hq": config.Hq,
        "Hkv": config.Hkv,
        "GQA": f"{config.Hq // config.Hkv}:1",
        "ms": None,
        "tflops": None,
        "max_abs": None,
        "status": "ERROR",
        "kernel": "",
        "path": path,
        "settings": "",
        "reason": "",
    }


def _error_text(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def _tflops(config: Config, ms: float) -> float:
    causal_pairs = config.S * (config.S + 1) // 2
    flops = config.B * 4 * config.Hq * _HEAD_SIZE * causal_pairs
    return flops / (ms * 1e-3) / 1e12


def _finish_row(
    row: dict[str, Any], output: torch.Tensor, reference: torch.Tensor
) -> None:
    max_abs = (output.float() - reference).abs().max().item()
    row["max_abs"] = max_abs
    if max_abs < MAX_ABS_TOL:
        row["status"] = "PASS"
        row["reason"] = ""
    else:
        row["status"] = "FAIL"
        row["reason"] = (
            f"max_abs={max_abs:.8g} exceeds MAX_ABS_TOL={MAX_ABS_TOL:.8g}"
        )


def _unified_identity(problem) -> tuple[str, str, str]:
    """Return the normal auto-route's path, concrete kernel names, and settings."""
    import kernels.common.attention_unified as unified

    selected = problem.select_path()
    if selected == "3d":
        supported, _ = unified.supports_native_unified_attention_3d_tiled(problem)
        if supported:
            segment = unified._tiled_3d_spec_from_problem(problem)
            (
                _,
                reduce_spec_type,
                _,
                _,
                _,
            ) = unified._tiled_3d_impl(unified._resolve_attention_arch())
            reduction = reduce_spec_type(
                head_size=problem.head_size,
                num_query_heads=problem.num_query_heads,
                num_kv_heads=problem.num_kv_heads,
                dtype=problem.dtype,
                num_segments=unified._num_segments(problem),
                waves_per_eu=unified._select_3d_waves_per_eu(problem),
            )
            return "auto:3d", (
                f"{segment.kernel_name()} + {reduction.kernel_name()}"
            ), repr(segment)

    supported, _ = unified.supports_native_unified_attention_tiled(problem)
    if supported:
        spec = unified._tiled_spec_from_problem(problem)
        return "auto:2d", spec.kernel_name(), repr(spec)

    return "auto:scalar", "rocke_unified_attention_2d_scalar", repr(problem)


def run_rocke_pair(
    config: Config, *, warmup: int, iters: int, seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run dense and normal auto-unified attention from one BF16 Q/K/V fixture."""
    from kernels.gfx942.attention_dense import (
        gfx942_kernel_name,
        run_attention_dense_torch,
        supports_attention_dense,
    )

    dense = _row(config, path="attention_dense")
    unified = _row(config, path="auto")
    preflight_succeeded = False
    supported = False
    try:
        spec = dense_spec(config)
        dense["kernel"] = gfx942_kernel_name(spec)
        dense["settings"] = repr(spec)
        supported, reason = supports_attention_dense(spec, arch="gfx942")
        preflight_succeeded = True
    except Exception as error:  # dispatch failure is a real runner error
        dense["reason"] = _error_text(error)
    if preflight_succeeded and not supported:
        dense["status"] = "UNSUPPORTED"
        dense["reason"] = str(reason)

    if not torch.cuda.is_available():
        message = "CUDA is unavailable"
        if dense["status"] != "UNSUPPORTED":
            dense["reason"] = message
        unified["reason"] = message
        return dense, unified

    try:
        device = torch.device("cuda")
        generator = torch.Generator(device=device).manual_seed(seed)
        q = torch.randn(
            config.B, config.S, config.Hq, _HEAD_SIZE,
            dtype=torch.bfloat16, device=device, generator=generator,
        ).contiguous()
        k = torch.randn(
            config.B, config.S, config.Hkv, _HEAD_SIZE,
            dtype=torch.bfloat16, device=device, generator=generator,
        ).contiguous()
        v = torch.randn(
            config.B, config.S, config.Hkv, _HEAD_SIZE,
            dtype=torch.bfloat16, device=device, generator=generator,
        ).contiguous()
        scale = 1.0 / math.sqrt(_HEAD_SIZE)
        gqa = config.Hq // config.Hkv
        reference = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2).float(),
            k.transpose(1, 2).float().repeat_interleave(gqa, dim=1),
            v.transpose(1, 2).float().repeat_interleave(gqa, dim=1),
            is_causal=True,
            scale=scale,
        )
        k_cache, v_cache, block_table = make_identity_paged_kv(k, v)
        cu_seqlens_q = (
            torch.arange(config.B + 1, dtype=torch.int32, device=device) * config.S
        )
        seqused_k = torch.full(
            (config.B,), config.S, dtype=torch.int32, device=device
        )
        q_paged = q.reshape(config.B * config.S, config.Hq, _HEAD_SIZE)
        from kernels import UnifiedAttentionProblem, run_unified_attention_torch

        problem = UnifiedAttentionProblem(
            total_q=config.B * config.S,
            num_seqs=config.B,
            num_query_heads=config.Hq,
            num_kv_heads=config.Hkv,
            head_size=_HEAD_SIZE,
            block_size=PAGE_SIZE,
            max_seqlen_q=config.S,
            max_seqlen_k=config.S,
            dtype="bf16",
            q_dtype="bf16",
            num_cus=torch.cuda.get_device_properties(0).multi_processor_count,
        )
        unified["path"], unified["kernel"], unified["settings"] = _unified_identity(
            problem
        )
        stream = int(torch.cuda.current_stream().cuda_stream)
    except Exception as error:
        message = _error_text(error)
        if dense["status"] != "UNSUPPORTED":
            dense["reason"] = message
        unified["reason"] = message
        return dense, unified

    from rocke.runtime import synchronize_and_release, time_launches

    dense_launchable = preflight_succeeded and supported

    if dense_launchable:
        try:
            dense_out = torch.empty_like(q)

            def run_dense() -> None:
                run_attention_dense_torch(
                    spec=spec, q=q, k=k, v=v, out=dense_out, scale=scale, stream=stream
                )

            run_dense()  # compile and launch outside the timed region
            torch.cuda.synchronize()
            _finish_row(dense, dense_out.transpose(1, 2), reference)
            dense["ms"] = time_launches(
                run_dense, warmup=warmup, iters=iters, stream=stream
            )
            dense["tflops"] = _tflops(config, dense["ms"])
            synchronize_and_release(stream)
        except Exception as error:
            dense["status"] = "ERROR"
            dense["reason"] = _error_text(error)

    try:
        unified_out = torch.empty_like(q_paged)

        def run_unified() -> None:
            run_unified_attention_torch(
                problem=problem,
                q=q_paged,
                k=k_cache,
                v=v_cache,
                out=unified_out,
                cu_seqlens_q=cu_seqlens_q,
                seqused_k=seqused_k,
                softmax_scale=scale,
                block_table=block_table,
                softcap=0.0,
                sinks=None,
                backend="auto",
                stream=stream,
            )

        run_unified()  # compile and launch outside the timed region
        torch.cuda.synchronize()
        _finish_row(
            unified,
            unified_out.view(config.B, config.S, config.Hq, _HEAD_SIZE).transpose(1, 2),
            reference,
        )
        unified["ms"] = time_launches(
            run_unified, warmup=warmup, iters=iters, stream=stream
        )
        unified["tflops"] = _tflops(config, unified["ms"])
        synchronize_and_release(stream)
    except Exception as error:
        unified["status"] = "ERROR"
        unified["reason"] = _error_text(error)
    return dense, unified


def _write_tsv(path: str, rows: list[dict[str, Any]]) -> None:
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_ROW_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dense", required=True)
    parser.add_argument("--out-unified", required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    dense_rows: list[dict[str, Any]] = []
    unified_rows: list[dict[str, Any]] = []
    for config in CONFIGS:
        dense, unified = run_rocke_pair(
            config, warmup=args.warmup, iters=args.iters, seed=args.seed
        )
        dense_rows.append(dense)
        unified_rows.append(unified)
        print(dense)
        print(unified)
    _write_tsv(args.out_dense, dense_rows)
    _write_tsv(args.out_unified, unified_rows)
    return int(
        any(
            row["status"] in {"FAIL", "ERROR"}
            for row in (*dense_rows, *unified_rows)
        )
    )


if __name__ == "__main__":
    sys.exit(main())
