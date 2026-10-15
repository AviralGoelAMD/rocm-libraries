#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/fmha_bwd_emit.py -- Python reference emitter for the
# FMHA-backward parity harness. Selects one sampled config by argv[1]:
# 0..5 are bwd_mode="atomic" FmhaBwdSpec configs built via build_fmha_bwd;
# 6.. are (bwd_mode="mfma" spec, launch-plan stage) pairs built via
# build_fmha_bwd_kernels(spec)[stage]. Prints lower_kernel_to_llvm(arch='gfx950')
# to stdout so it can be byte-compared with the C emitter fmha_bwd_emit.c.
from kernels.common._fmha_common import FmhaCommonSpec, FmhaShape
from kernels.common.fmha_bwd import FmhaBwdSpec, build_fmha_bwd, build_fmha_bwd_kernels
from _emit_common import run_emit

# (head_size, HQ, HK, dtype, mask, seqlen, knob overrides, stages)
_MFMA = [
    (128, 2, 2, "bf16", "none", 256, {}, (0, 1, 2)),
    (128, 4, 2, "bf16", "causal", 256, {}, (0, 1, 2)),
    (
        128, 2, 2, "f16", "causal", 256,
        dict(mfma_causal_order="pair", mfma_causal_split=False, output_grad_dtype="f16",
             mfma_natural_rows=False, mfma_exp2_fast=False, mfma_waves_per_eu=0,
             mfma_dq_qtiles=1, mfma_lds_pad=8),
        (1, 2),
    ),
    (128, 2, 2, "bf16", "none", 256, dict(mfma_fused_dq=True, mfma_dkdv_ktiles=4), (0, 1)),
    (128, 4, 2, "bf16", "causal", 256, dict(mfma_fused_dq=True, mfma_dkdv_ktiles=2), (1,)),
    (128, 2, 2, "bf16", "none", 256, dict(output_grad_dtype="bf16"), (1, 2)),
    (64, 1, 1, "bf16", "causal", 128, dict(mfma_causal_order="none"), (1, 2)),
]
_MFMA_CFGS = [(row, st) for row in _MFMA for st in row[7]]
_STAGE = [None]


def _mfma_spec(idx: int) -> FmhaBwdSpec:
    (h, hq, hk, dt, mask, sq, knobs, _), stage = _MFMA_CFGS[idx]
    _STAGE[0] = stage
    return FmhaBwdSpec(
        common=FmhaCommonSpec(
            shape=FmhaShape(head_size=h, num_query_heads=hq, num_kv_heads=hk),
            dtype=dt,
            mask_mode=mask,
        ),
        seqlen_q=sq,
        seqlen_k=sq,
        bwd_mode="mfma",
        **knobs,
    )


def _build(spec: FmhaBwdSpec, arch: str = "gfx950"):
    if spec.bwd_mode == "atomic":
        return build_fmha_bwd(spec, arch=arch)
    return build_fmha_bwd_kernels(spec, arch)[_STAGE[0]][0]


def _spec(idx: int) -> FmhaBwdSpec:
    if 6 <= idx < 6 + len(_MFMA_CFGS):
        return _mfma_spec(idx - 6)
    if idx == 0:
        return FmhaBwdSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(head_size=64, num_query_heads=4, num_kv_heads=4),
                dtype="f16",
                mask_mode="none",
            ),
            seqlen_q=16,
            seqlen_k=16,
        )
    if idx == 1:
        return FmhaBwdSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(head_size=128, num_query_heads=8, num_kv_heads=4),
                dtype="f16",
                mask_mode="none",
            ),
            seqlen_q=32,
            seqlen_k=64,
        )
    if idx == 2:
        return FmhaBwdSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(head_size=256, num_query_heads=16, num_kv_heads=8),
                dtype="bf16",
                mask_mode="causal",
            ),
            seqlen_q=64,
            seqlen_k=64,
        )
    if idx == 3:
        return FmhaBwdSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(head_size=64, num_query_heads=2, num_kv_heads=2),
                dtype="f16",
                mask_mode="sliding_window",
                sliding_window=16,
            ),
            seqlen_q=128,
            seqlen_k=128,
        )
    if idx == 4:
        return FmhaBwdSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(head_size=192, num_query_heads=8, num_kv_heads=2),
                dtype="bf16",
                mask_mode="none",
            ),
            seqlen_q=256,
            seqlen_k=256,
        )
    if idx == 5:
        return FmhaBwdSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(head_size=128, num_query_heads=1, num_kv_heads=1),
                dtype="f16",
                mask_mode="causal",
            ),
            seqlen_q=128,
            seqlen_k=256,
        )
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    return run_emit(
        _spec,
        _build,
        usage=f"usage: fmha_bwd_emit.py <config_index 0..{5 + len(_MFMA_CFGS)}>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
