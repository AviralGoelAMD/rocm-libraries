# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx950 pipelined paged prefill (``_gfx950_prefill_pipeline``) on the GPU.

The production dispatcher routes d128 / block_size 16 causal prefill (single
sequence, multi-sequence and ragged batches) to the transposed-32x32 body with
the pipeline knob stack: XOR-swizzled K/V LDS, heaviest-first Q blocks, V
double buffer with the next-tile DMA kept in flight (scoped-noalias LDS tags,
LDS-only publish barriers), block ids prefetched a tile ahead, the causal mask
peeled to boundary tiles and the folded-scale softmax.

Every case:
  * asserts the problem really takes the pipeline route;
  * pre-fills the output with NaN so an unwritten row cannot pass;
  * checks the result against the fp32 paged reference;
  * re-runs with V == 1 (the exact answer is 1 for every row with at least one
    key), which isolates the softmax row sum: a dropped or double-counted half
    shows up as an O(0.1-1) error.
Under ``ROCKE_BACKEND=both ROCKE_CPP_STRICT=1`` each kernel is also lowered by
the C++ engine and must match the Python ``.ll`` byte for byte.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from kernels.common import attention_unified as au

pytestmark = pytest.mark.gpu


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
    spec = importlib.util.spec_from_file_location("parity_unified_attention_pipe", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["parity_unified_attention_pipe"] = mod
    spec.loader.exec_module(mod)
    return mod


_RAGGED7 = [(n, n) for n in (1000, 3000, 37, 2048, 517, 1, 1600)]

# (name, seq_lens, hq, hk, dtype)
_CASES = [
    ("single_s2048_gqa4_bf16", [(2048, 2048)], 32, 8, "bf16"),
    ("single_s2048_gqa4_fp16", [(2048, 2048)], 32, 8, "fp16"),
    ("single_s4096_gqa8_bf16", [(4096, 4096)], 8, 1, "bf16"),
    ("single_s1000_tail_fp16", [(1000, 1000)], 32, 8, "fp16"),
    ("multi_4x2048_bf16", [(2048, 2048)] * 4, 32, 8, "bf16"),
    ("multi_4x2048_fp16", [(2048, 2048)] * 4, 32, 8, "fp16"),
    ("ragged7_bf16", _RAGGED7, 32, 8, "bf16"),
    ("ragged7_fp16", _RAGGED7, 32, 8, "fp16"),
    ("chunked_prefill_bf16", [(512, 1536), (300, 300), (700, 2100)], 32, 8, "bf16"),
]


def _problem(data, s, dtype):
    from kernels import UnifiedAttentionProblem

    return UnifiedAttentionProblem(
        total_q=int(data["query"].shape[0]),
        num_seqs=len(s.seq_lens),
        num_query_heads=s.num_query_heads,
        num_kv_heads=s.num_kv_heads,
        head_size=s.head_size,
        block_size=s.block_size,
        max_seqlen_q=data["max_query_len"],
        max_seqlen_k=data["max_kv_len"],
        dtype=dtype,
    )


def _launch(problem, data, value_cache):
    import torch
    from kernels import run_unified_attention_torch

    out = torch.full_like(data["query"], float("nan"))
    run_unified_attention_torch(
        problem=problem,
        q=data["query"],
        k=data["key_cache"],
        v=value_cache,
        out=out,
        cu_seqlens_q=data["cu_q"],
        seqused_k=data["kv_lens"],
        softmax_scale=data["scale"],
        block_table=data["block_tables"],
        softcap=0.0,
        backend="auto",
    )
    torch.cuda.synchronize()
    return out


@requires_gfx950
@pytest.mark.parametrize(
    "name,seq_lens,hq,hk,dtype", _CASES, ids=[c[0] for c in _CASES]
)
def test_pipeline_numeric(name, seq_lens, hq, hk, dtype):
    import torch

    H = _harness()
    tdt = torch.bfloat16 if dtype == "bf16" else torch.float16
    bs = 16
    max_blocks = -(-max(kv for _q, kv in seq_lens) // bs)
    blocks = len(seq_lens) * max_blocks + 64
    s = H.Scenario(
        name=name,
        seq_lens=seq_lens,
        num_query_heads=hq,
        num_kv_heads=hk,
        head_size=128,
        block_size=bs,
        dtype=tdt,
        num_blocks=blocks,
    )
    data = H.make_inputs(s, seed=0)
    # Distinct physical blocks in a shuffled order (no aliasing between
    # sequences, non-monotone block table).
    perm = torch.randperm(blocks, device="cuda", dtype=torch.int32)
    bt = data["block_tables"]
    data["block_tables"] = perm[: bt.numel()].reshape(bt.shape).contiguous()
    problem = _problem(data, s, dtype)
    assert au._gfx950_prefill_pipeline(problem), f"{name}: not on the pipeline route"
    assert "_dmaov_bidpf" in au._tiled_spec_from_problem(problem).kernel_name()

    tol = 3e-3 if dtype == "fp16" else 4e-2
    out = _launch(problem, data, data["value_cache"])
    assert torch.isfinite(out.float()).all(), f"{name}: unwritten / non-finite rows"
    m = H.compare(H.run_reference(s, data), out)
    assert m["max_abs"] <= tol, f"{name}: {m}"

    ones = torch.ones_like(data["value_cache"])
    out1 = _launch(problem, data, ones)
    assert torch.isfinite(out1.float()).all(), f"{name}: V=1 non-finite rows"
    err1 = float((out1.float() - 1.0).abs().max())
    assert err1 <= 2 * torch.finfo(tdt).eps, f"{name}: V=1 row-sum error {err1}"
