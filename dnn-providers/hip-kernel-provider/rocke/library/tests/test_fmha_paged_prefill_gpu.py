# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-GPU numeric lane for the paged-KV prefill FMHA kernel (both bodies).

Multi-sequence batches with ragged q / k lengths (not multiples of the
16-row MFMA tile), GQA, and block tables shuffled over one shared page
pool. Every unused page slot (past a sequence's ``seqlen_k`` and every
unassigned page) holds NaN, so a kernel that reads K / V past the
sequence end, or skips an O row (O is pre-filled with NaN), fails.

Every test is marked ``gpu`` and gated on a gfx950 / gfx942 device, so it is
a graceful skip on a CPU CI box. Run standalone:

    HIP_VISIBLE_DEVICES=0 python -m pytest tests/test_fmha_paged_prefill_gpu.py
"""

from __future__ import annotations

import math

import pytest
from kernels.common._fmha_common import FmhaCommonSpec, FmhaShape
from kernels.common.fmha_paged_prefill import (
    FmhaFwdPagedPrefillSpec,
    build_fmha_fwd_paged_prefill,
    fmha_fwd_paged_prefill_grid,
    fmha_fwd_paged_prefill_signature,
)

torch = pytest.importorskip("torch", reason="ROCm torch required")


def _gpu_arch():
    if not torch.cuda.is_available():
        return None
    arch = torch.cuda.get_device_properties(0).gcnArchName.lower()
    for a in ("gfx950", "gfx942"):
        if a in arch:
            return a
    return None


requires_cdna_gpu = pytest.mark.skipif(
    _gpu_arch() is None, reason="needs a gfx950 / gfx942 GPU with ROCm torch"
)

_D, _HQ, _HK = 128, 16, 4

# (case id, q lengths, k lengths, page_block_size)
_CASES = [
    ("ragged_self", [100, 37, 250, 16, 1], [100, 37, 250, 16, 1], 16),
    ("ragged_prefix", [100, 37, 250, 16, 1], [131, 37, 260, 40, 17], 16),
    ("ragged_page8", [100, 37, 250, 16, 1], [100, 45, 250, 16, 9], 8),
    ("ragged_page32", [100, 37, 250, 16, 1], [100, 37, 250, 33, 1], 32),
    ("aligned", [256, 512, 128], [256, 512, 128], 16),
]


def _run(ql, kl, page, causal, use_mfma_body):
    from rocke.helpers import compile_kernel
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    dev = "cuda"
    batch = len(ql)
    gen = torch.Generator(device=dev).manual_seed(7)
    mbs = max(math.ceil(k / page) for k in kl)
    pool = batch * mbs + 5
    kc = torch.full(
        (pool, page, _HK, _D), float("nan"), device=dev, dtype=torch.float16
    )
    vc = torch.full_like(kc, float("nan"))
    perm = torch.randperm(pool, device=dev, generator=gen).to(torch.int32)
    bt = perm[: batch * mbs].view(batch, mbs).contiguous()
    ks, vs = [], []
    for i, k_len in enumerate(kl):
        k = torch.randn(k_len, _HK, _D, device=dev, generator=gen).half()
        v = torch.randn(k_len, _HK, _D, device=dev, generator=gen).half()
        for t in range(k_len):
            blk = int(bt[i, t // page])
            kc[blk, t % page] = k[t]
            vc[blk, t % page] = v[t]
        ks.append(k)
        vs.append(v)
    total_q = sum(ql)
    q = torch.randn(total_q, _HQ, _D, device=dev, generator=gen).half()
    o = torch.full_like(q, float("nan"))
    cuq = torch.tensor(
        [0, *torch.tensor(ql).cumsum(0).tolist()], device=dev, dtype=torch.int32
    )
    seqlens_k = torch.tensor(kl, device=dev, dtype=torch.int32)

    arch = _gpu_arch()
    spec = FmhaFwdPagedPrefillSpec(
        common=FmhaCommonSpec(
            shape=FmhaShape(_D, _HQ, _HK),
            dtype="f16",
            mask_mode="causal" if causal else "none",
        ),
        page_block_size=page,
        max_blocks_per_seq=mbs,
        batch=batch,
        use_mfma_body=use_mfma_body,
    )
    kernel = build_fmha_fwd_paged_prefill(spec, arch=arch)
    art = compile_kernel(kernel, arch=arch)
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=kernel.name,
        signature=fmha_fwd_paged_prefill_signature(spec),
    )
    sb = page * _HK * _D
    launcher(
        {
            "Q": q,
            "K_cache": kc,
            "V_cache": vc,
            "O": o,
            "block_table": bt,
            "cu_seqlens_q": cuq,
            "seqlens_k": seqlens_k,
            "scale_log2": math.log2(math.e) / math.sqrt(_D),
            "total_q": total_q,
            "batch": batch,
            "stride_q_token": _HQ * _D,
            "stride_q_head": _D,
            "stride_block": sb,
            "stride_page": _HK * _D,
            "stride_kv_head": _D,
            "stride_v_block": sb,
            "stride_v_page": _HK * _D,
            "stride_v_kv_head": _D,
            "stride_o_token": _HQ * _D,
            "stride_o_head": _D,
            "block_table_stride": mbs,
        },
        config=LaunchConfig(
            grid=fmha_fwd_paged_prefill_grid(spec, total_q=total_q), block=(64, 1, 1)
        ),
    )
    torch.cuda.synchronize()

    group = _HQ // _HK
    errs = []
    for i in range(batch):
        qs = q[int(cuq[i]) : int(cuq[i + 1])].float()
        k = ks[i].float().repeat_interleave(group, dim=1)
        v = vs[i].float().repeat_interleave(group, dim=1)
        s = torch.einsum("qhd,khd->hqk", qs, k) / math.sqrt(_D)
        if causal:
            # Top-left alignment (k <= local q), the kernel's paged semantics.
            qi = torch.arange(ql[i], device=dev)[:, None]
            ki = torch.arange(kl[i], device=dev)[None]
            s = s.masked_fill(ki > qi, float("-inf"))
        ref = torch.einsum("hqk,khd->qhd", s.softmax(-1), v)
        out = o[int(cuq[i]) : int(cuq[i + 1])].float()
        errs.append((out - ref).abs().max().item())
    return errs


@requires_cdna_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("use_mfma_body", [True, False], ids=["mfma", "warp"])
@pytest.mark.parametrize("causal", [False, True], ids=["nc", "causal"])
@pytest.mark.parametrize("_name,ql,kl,page", _CASES, ids=[c[0] for c in _CASES])
def test_paged_prefill_multiseq_vs_fp32(_name, ql, kl, page, causal, use_mfma_body):
    errs = _run(ql, kl, page, causal, use_mfma_body)
    assert all(e < 3e-3 for e in errs), f"per-sequence max_abs {errs}"
