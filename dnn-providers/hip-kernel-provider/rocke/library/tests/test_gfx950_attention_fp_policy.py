# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx950 unified attention: f32 denormal flush + FMA contraction policy.

The gfx950 dispatcher turns ``use_f32_denormal_flush`` on for the paged 2D
prefill kernel and both split-KV 3D decode kernels (segment + reduce), and
``use_fp_contract`` on for the 2D kernel only. Every other arch keeps both off.

CPU part: routing, cache keys, kernel names and the emitted LLVM attributes.
GPU part (gfx950 only): each shape runs through the production dispatcher with
the policy on and with it forced off, both are gated against the fp32 paged
reference, and the two outputs are compared directly so a policy-induced error
cannot hide inside the reference tolerance.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest import mock

import pytest

from kernels import UnifiedAttentionProblem
from kernels.common import attention_unified as au
from builders.common import attention_spec_builder as sb
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

_FLUSH_ATTR = '"denormal-fp-math-f32"="preserve-sign,preserve-sign"'


def _problem(**kw) -> UnifiedAttentionProblem:
    base = dict(
        total_q=2048,
        num_seqs=1,
        num_query_heads=32,
        num_kv_heads=8,
        head_size=128,
        block_size=16,
        max_seqlen_q=2048,
        max_seqlen_k=2048,
        dtype="bf16",
        num_cus=256,
    )
    base.update(kw)
    return UnifiedAttentionProblem(**base)


def _decode_problem(**kw) -> UnifiedAttentionProblem:
    return _problem(
        **{"total_q": 32, "num_seqs": 32, "max_seqlen_q": 1, "max_seqlen_k": 8192, **kw}
    )


@pytest.fixture
def arch():
    def _set(name):
        return mock.patch.object(au, "_resolve_attention_arch", return_value=name)

    return _set


# ---------------------------------------------------------------------------
# CPU: selection, keys, names, emitted attributes
# ---------------------------------------------------------------------------


def test_gfx950_2d_spec_turns_both_flags_on(arch):
    with arch("gfx950"):
        spec = sb._tiled_spec_from_problem(_problem())
    assert spec.use_f32_denormal_flush and spec.use_fp_contract
    assert "_ftz_fpc" in spec.kernel_name()


def test_gfx950_3d_spec_turns_flush_on_only(arch):
    with arch("gfx950"):
        spec = sb._tiled_3d_spec_from_problem(_decode_problem())
    assert spec.use_f32_denormal_flush
    assert spec.kernel_name().endswith("_ftz")


@pytest.mark.parametrize("name", ["gfx942", "gfx1201"])
def test_other_arches_keep_the_policy_off(arch, name):
    with arch(name):
        spec2d = sb._tiled_spec_from_problem(_problem())
        spec3d = sb._tiled_3d_spec_from_problem(_decode_problem())
        assert not au._enable_f32_denormal_flush(_problem())
        assert not au._enable_fp_contract(_problem())
    for spec in (spec2d, spec3d):
        assert not getattr(spec, "use_f32_denormal_flush", False)
        assert not getattr(spec, "use_fp_contract", False)
        assert "_ftz" not in spec.kernel_name()


def test_cache_keys_follow_the_predicates(arch):
    with arch("gfx950"):
        on2d, on3d = au._tiled_cache_key(_problem()), au._tiled_3d_cache_key(
            _decode_problem()
        )
        with mock.patch.object(
            au, "_enable_f32_denormal_flush", return_value=False
        ), mock.patch.object(au, "_enable_fp_contract", return_value=False):
            off2d = au._tiled_cache_key(_problem())
            off3d = au._tiled_3d_cache_key(_decode_problem())
            spec2d = sb._tiled_spec_from_problem(_problem())
            spec3d = sb._tiled_3d_spec_from_problem(_decode_problem())
    assert on2d != off2d and on3d != off3d
    assert not spec2d.use_f32_denormal_flush and not spec2d.use_fp_contract
    assert not spec3d.use_f32_denormal_flush


def test_flag_on_ir_carries_the_backend_policy(arch):
    from kernels.gfx950.attention_tiled_2d import build_unified_attention_2d_tiled
    from kernels.gfx950.attention_tiled_3d import (
        UnifiedAttentionReduceTiledSpec,
        build_unified_attention_3d_tiled,
        build_unified_attention_reduce_tiled,
    )

    with arch("gfx950"):
        spec2d = sb._tiled_spec_from_problem(_problem())
        spec3d = sb._tiled_3d_spec_from_problem(_decode_problem())
    ll2d = _lower_kernel_to_llvm_python(
        build_unified_attention_2d_tiled(spec2d), arch="gfx950"
    )
    assert _FLUSH_ATTR in ll2d
    assert "= fmul contract " in ll2d and "= fadd contract " in ll2d
    assert "= fdiv contract " not in ll2d

    ll3d = _lower_kernel_to_llvm_python(
        build_unified_attention_3d_tiled(spec3d), arch="gfx950"
    )
    assert _FLUSH_ATTR in ll3d and " contract " not in ll3d

    reduce_spec = UnifiedAttentionReduceTiledSpec(
        head_size=128,
        num_query_heads=32,
        num_kv_heads=8,
        dtype="bf16",
        num_segments=spec3d.num_segments,
        use_f32_denormal_flush=True,
    )
    llr = _lower_kernel_to_llvm_python(
        build_unified_attention_reduce_tiled(reduce_spec), arch="gfx950"
    )
    assert _FLUSH_ATTR in llr
    assert reduce_spec.kernel_name().endswith("_ftz")


def test_flag_off_ir_has_no_policy(arch):
    from kernels.gfx950.attention_tiled_2d import build_unified_attention_2d_tiled
    from dataclasses import replace

    with arch("gfx950"):
        spec = sb._tiled_spec_from_problem(_problem())
    off = replace(spec, use_f32_denormal_flush=False, use_fp_contract=False)
    ll = _lower_kernel_to_llvm_python(
        build_unified_attention_2d_tiled(off), arch="gfx950"
    )
    assert "denormal-fp-math" not in ll and " contract " not in ll


@pytest.mark.parametrize("value", [1, "true", None])
def test_spec_rejects_non_bool_policy(value):
    from kernels.gfx950.attention_tiled_2d import UnifiedAttention2DTiledSpec
    from kernels.gfx950.attention_tiled_3d import UnifiedAttention3DTiledSpec

    common = dict(
        head_size=128,
        block_size=16,
        num_query_heads=8,
        num_kv_heads=8,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
    )
    with pytest.raises(ValueError, match="must be a bool"):
        UnifiedAttention2DTiledSpec(**common, use_fp_contract=value)
    with pytest.raises(ValueError, match="must be a bool"):
        UnifiedAttention3DTiledSpec(
            **common, num_segments=4, use_f32_denormal_flush=value
        )


# ---------------------------------------------------------------------------
# GPU (gfx950): production dispatcher, policy on vs forced off, fp32 oracle
# ---------------------------------------------------------------------------


def _gfx950_gpu() -> bool:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950")


requires_gfx950 = pytest.mark.skipif(
    not _gfx950_gpu(), reason="needs a gfx950 GPU with ROCm torch"
)


def _harness():
    path = (
        Path(__file__).resolve().parents[1]
        / "builders/gfx950/attention/prefill/parity_unified_attention.py"
    )
    spec = importlib.util.spec_from_file_location("parity_unified_attention_fp", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["parity_unified_attention_fp"] = mod
    spec.loader.exec_module(mod)
    return mod


# (name, seq_lens, hq, hk, d, bs, dtype, extra Scenario fields, q_scale)
_CASES = [
    ("p2d_d128_gqa4_bf16", [(2048, 2048)], 32, 8, 128, 16, "bf16", {}, 1.0),
    ("p2d_d128_gqa4_fp16", [(2048, 2048)], 32, 8, 128, 16, "fp16", {}, 1.0),
    ("p2d_d64_gqa8_sinks", [(1024, 1024)], 64, 8, 64, 32, "bf16", {"use_sinks": True}, 1.0),
    ("p2d_d256_gqa8_bf16", [(1024, 1024)], 16, 2, 256, 16, "bf16", {}, 1.0),
    ("p2d_d128_sw_fp16", [(2048, 2048)], 32, 8, 128, 16, "fp16", {"sliding_window": 512}, 1.0),
    ("p2d_d128_softcap", [(1024, 1024)], 16, 4, 128, 16, "bf16", {"softcap": 30.0}, 1.0),
    ("p2d_d128_mixed_batch", [(512, 1536), (1, 2048), (300, 300)], 32, 8, 128, 16, "bf16", {}, 1.0),
    ("p2d_d128_logit_stress", [(2048, 2048)], 32, 8, 128, 16, "bf16", {}, 32.0),
    ("d3d_d128_gqa4_bf16", [(1, 4096)] * 8, 32, 8, 128, 16, "bf16", {}, 1.0),
    ("d3d_d128_gqa4_fp16", [(1, 4096)] * 8, 32, 8, 128, 16, "fp16", {}, 1.0),
    ("d3d_d64_sw", [(1, 4096)] * 8, 16, 2, 64, 16, "bf16", {"sliding_window": 1024}, 1.0),
    ("d3d_d128_logit_stress", [(1, 8192)] * 4, 32, 8, 128, 16, "bf16", {}, 8.0),
]


@requires_gfx950
@pytest.mark.parametrize(
    "name,seq_lens,hq,hk,d,bs,dtype,extra,q_scale", _CASES, ids=[c[0] for c in _CASES]
)
def test_policy_numeric_on_vs_off_vs_fp32(
    name, seq_lens, hq, hk, d, bs, dtype, extra, q_scale
):
    import torch

    H = _harness()
    tdt = torch.bfloat16 if dtype == "bf16" else torch.float16
    blocks = sum(-(-kv // bs) for _q, kv in seq_lens) + 64
    s = H.Scenario(
        name=name,
        seq_lens=seq_lens,
        num_query_heads=hq,
        num_kv_heads=hk,
        head_size=d,
        block_size=bs,
        dtype=tdt,
        num_blocks=blocks,
        **extra,
    )
    data = H.make_inputs(s, seed=0)
    if q_scale != 1.0:
        data["query"] = (data["query"].float() * q_scale).to(tdt)
    ref = H.run_reference(s, data)

    out_on, _ = H.run_unified("rocke", s, data, warmup=0, attempts=1)
    out_on = out_on.clone()
    with mock.patch.object(
        au, "_enable_f32_denormal_flush", return_value=False
    ), mock.patch.object(au, "_enable_fp_contract", return_value=False):
        out_off, _ = H.run_unified("rocke", s, data, warmup=0, attempts=1)
    torch.cuda.synchronize()

    m_on = H.compare(ref, out_on)
    m_off = H.compare(ref, out_off)
    assert torch.isfinite(out_on.float()).all(), f"{name}: non-finite output"
    # Repo-wide attention gates: fp16 3e-3, bf16 the bf16 output ulp budget.
    # The logit-stress cases (scaled Q) exceed that budget with the policy
    # off as well, by the same amount, so they are gated relative to the
    # policy-off kernel below instead.
    tol = 3e-3 if dtype == "fp16" else 4e-2
    if q_scale == 1.0:
        assert m_on["max_abs"] <= tol, f"{name}: policy-on max_abs {m_on}"
    # The policy may move a fused pair by one ulp; it must not add error
    # beyond one output ulp of the largest output.
    ulp = torch.finfo(tdt).eps * max(1.0, float(out_off.float().abs().max()))
    direct = float((out_on.float() - out_off.float()).abs().max())
    assert direct <= 2 * ulp, f"{name}: on vs off differ by {direct} (> 2 ulp {ulp})"
    assert m_on["max_abs"] <= m_off["max_abs"] + 2 * ulp, (name, m_on, m_off)
