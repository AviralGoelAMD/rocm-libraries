# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx950 paged 3D decode: NaN in a page's unused slots must not reach the output.

A KV cache page holds ``block_size`` slots; a sequence whose length is not a
multiple of the page size leaves the tail slots of its last page unused, and a
serving cache may hold anything there (stale data, NaN). The kernel must treat
keys ``>= seq_len`` as absent: masking the scores alone is not enough, because
``P = 0`` times ``V = NaN`` is NaN inside the PV MFMA.

Set ROCKE_REQUIRE_DECODE_GPU=gfx950 to require hardware (no skips).
"""

import math
import os
from dataclasses import replace

import pytest

BLOCK = 16
HQ, HKV, D = 32, 8, 128
# Odd lengths: every sequence's last page has unused slots. One length sits a
# single key past a 32-key tile boundary, one a single key short of it.
LENGTHS = [1201, 33, 1023, 511]


@pytest.fixture(scope="module")
def gpu():
    required = os.environ.get("ROCKE_REQUIRE_DECODE_GPU")
    if required:
        import torch
    else:
        torch = pytest.importorskip("torch")
    from rocke.runtime.hip_module import get_device_arch

    if not torch.cuda.is_available():
        if required:
            pytest.fail("required decode GPU is unavailable")
        pytest.skip("requires a ROCm GPU")
    arch = get_device_arch()
    if arch != "gfx950":
        if required:
            pytest.fail(f"requested {required}, found {arch}")
        pytest.skip("gfx950 3D decode kernel")
    return torch


def _inputs(torch, fp8):
    g = torch.Generator().manual_seed(1234)
    nblk = (max(LENGTHS) + BLOCK - 1) // BLOCK
    npages = len(LENGTHS) * nblk + 3
    kv_dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    k = (torch.randn(npages, BLOCK, HKV, D, generator=g) * (2.0 if fp8 else 1.0)).to(kv_dtype)
    v = (torch.randn(npages, BLOCK, HKV, D, generator=g) * (2.0 if fp8 else 1.0)).to(kv_dtype)
    # Shuffled pages; the table is padded with the sequence's own last page.
    perm = torch.randperm(npages, generator=g)[: len(LENGTHS) * nblk].view(len(LENGTHS), nblk)
    table = perm.clone()
    for i, n in enumerate(LENGTHS):
        used = (n + BLOCK - 1) // BLOCK
        table[i, used:] = table[i, used - 1]
        last = int(table[i, used - 1])
        for t in (k, v):
            if fp8:
                t.view(torch.uint8)[last, n % BLOCK :] = 0x7F  # e4m3fn NaN
            else:
                t[last, n % BLOCK :] = float("nan")
    q = torch.randn(len(LENGTHS), HQ, D, generator=g).to(torch.bfloat16)
    k_scale, v_scale = (0.25, 0.5) if fp8 else (1.0, 1.0)
    ref = torch.zeros(len(LENGTHS), HQ, D)
    for i, n in enumerate(LENGTHS):
        kk = k[table[i].long()].reshape(-1, HKV, D)[:n].float() * k_scale
        vv = v[table[i].long()].reshape(-1, HKV, D)[:n].float() * v_scale
        qq = q[i].float().view(HKV, HQ // HKV, D)
        s = torch.einsum("hqd,khd->hqk", qq, kk) / math.sqrt(D)
        ref[i] = torch.einsum("hqk,khd->hqd", s.softmax(-1), vv).reshape(HQ, D)
    assert torch.isfinite(ref).all()
    return dict(
        q=q.cuda(),
        k=k.cuda(),
        v=v.cuda(),
        table=table.to(torch.int32).cuda(),
        lens=torch.tensor(LENGTHS, dtype=torch.int32).cuda(),
        cu_q=torch.arange(len(LENGTHS) + 1, dtype=torch.int32).cuda(),
        k_scale=k_scale,
        v_scale=v_scale,
        ref=ref,
    )


def _problem(fp8):
    from dispatch.attention import AttentionRequest
    from dispatch.attention.common import _problem as build

    return build(
        AttentionRequest(
            batch=len(LENGTHS), nhead_q=HQ, nhead_k=HKV, seqlen_q=1,
            seqlen_k=max(LENGTHS), hdim_q=D, hdim_v=D, arch="gfx950", dtype="bf16",
            mask_type=1, kv_block_size=BLOCK, use_fp8=fp8,
        )
    )


def _check(torch, out, inp, fp8):
    got = out.float().cpu()
    assert torch.isfinite(got).all(), "NaN from unused page slots reached the output"
    err = float((got - inp["ref"]).abs().max() / inp["ref"].abs().max())
    assert err <= (5e-2 if fp8 else 3e-2), err


@pytest.mark.parametrize("fp8", [False, True])
def test_production_3d_path_ignores_nan_slack(gpu, fp8):
    from kernels.common import attention_unified as au

    torch = gpu
    inp = _inputs(torch, fp8)
    out = torch.empty_like(inp["q"])
    au.run_unified_attention_torch(
        problem=_problem(fp8), q=inp["q"], k=inp["k"], v=inp["v"], out=out,
        cu_seqlens_q=inp["cu_q"], seqused_k=inp["lens"], softmax_scale=D**-0.5,
        block_table=inp["table"], softcap=0.0, backend="3d",
        k_scale=inp["k_scale"], v_scale=inp["v_scale"], use_graph=False,
    )
    torch.cuda.synchronize()
    _check(torch, out, inp, fp8)


_FAST = {"decode_fast_path": True, "kv_prefetch": True, "kv_tile_mult": 2}
_KNOBS = [
    ({}, False),
    (_FAST, False),
    ({**_FAST, "transposed_softmax": True}, False),
    (_FAST, True),
    ({**_FAST, "transposed_softmax": True}, True),
]


@pytest.mark.parametrize("nseg", [4, 32])
@pytest.mark.parametrize("over,fp8", _KNOBS)
def test_3d_variants_ignore_nan_slack(gpu, over, fp8, nseg):
    from kernels.common import attention_unified as au
    from rocke.helpers import compile_kernel
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    torch = gpu
    inp = _inputs(torch, fp8)
    B = len(LENGTHS)
    p = replace(_problem(fp8), num_kv_blocks=int(inp["k"].shape[0]))
    Spec3D, RedSpec, build_seg, build_red, _ = au._tiled_3d_impl("gfx950")
    seg = replace(au._tiled_3d_spec_from_problem(p), num_segments=nseg, **over)
    red = RedSpec(head_size=D, num_query_heads=HQ, num_kv_heads=HKV, dtype="bf16",
                  num_segments=nseg)
    sa = compile_kernel(build_seg(seg, arch="gfx950"), arch="gfx950")
    ra = compile_kernel(build_red(red, arch="gfx950"), arch="gfx950")
    segL = KernelLauncher(hsaco=sa.hsaco, kernel_name=sa.kernel_name,
                          signature=au._3d_signature("bf16", kv_dtype=seg.kv_storage_dtype),
                          cache_key=("nan_slack_seg", sa.kernel_name))
    redL = KernelLauncher(hsaco=ra.hsaco, kernel_name=ra.kernel_name,
                          signature=au._reduce_signature("bf16"),
                          cache_key=("nan_slack_red", ra.kernel_name))
    gx = B if seg.decode_fast_path else B // (16 // (HQ // HKV)) + B
    so = torch.empty(B, HQ, nseg, D, dtype=torch.float32, device="cuda")
    sm = torch.empty(B, HQ, nseg, dtype=torch.float32, device="cuda")
    se = torch.empty_like(sm)
    out = torch.empty_like(inp["q"])
    st = torch.cuda.current_stream().cuda_stream
    segL({"segm_output_ptr": so, "segm_max_ptr": sm, "segm_expsum_ptr": se,
          "query_ptr": inp["q"], "key_cache_ptr": inp["k"], "value_cache_ptr": inp["v"],
          "sink_ptr": torch.zeros(HQ, dtype=torch.float32, device="cuda"),
          "block_tables_ptr": inp["table"], "seq_lens_ptr": inp["lens"],
          "alibi_slopes_ptr": 0, "qq_bias_ptr": 0, "query_start_len_ptr": inp["cu_q"],
          "scale": D**-0.5, "k_scale": inp["k_scale"], "v_scale": inp["v_scale"],
          "softcap": 0.0, "num_seqs": B, "block_table_stride": int(inp["table"].stride(0)),
          "qq_bias_stride_0": 0},
         config=LaunchConfig(grid=(gx, HKV, nseg), block=(64, 1, 1), stream=st))
    redL({"output_ptr": out, "segm_output_ptr": so, "segm_max_ptr": sm,
          "segm_expsum_ptr": se, "seq_lens_ptr": inp["lens"]},
         config=LaunchConfig(grid=(B, HQ, 1), block=(64, 1, 1), stream=st))
    torch.cuda.synchronize()
    _check(torch, out, inp, fp8)
