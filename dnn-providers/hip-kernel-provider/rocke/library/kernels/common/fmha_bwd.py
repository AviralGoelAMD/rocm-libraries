# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""FMHA backward pass (CK Tile ``01_fmha`` bwd parity).

Computes::

 dQ[i, d] = sum_j P[i, j] * scale * (dS[i, j] - rowsum) * K[j, d]
 dK[j, d] = sum_i P[i, j] * scale * (dS[i, j] - rowsum) * Q[i, d]
 dV[j, d] = sum_i P[i, j] * dO[i, d]

where ``P = softmax(Q @ K^T / sqrt(d))`` (recomputed from saved
``(M, L)``) and ``dS[i, j] = dO[i, :] . V[j, :]``.

Warp-distributed body: one CTA per ``(q_token, head)``, one warp
per CTA, lane ``t`` owns head-dim slot ``t * EPT + k``. The first
K-loop computes ``rowsum_dp`` via a loop-carried iter_arg; the
second loop atomic-adds dV / dK to global and accumulates dQ per
lane (single store per lane at the end). Both loops apply the
spec's ``mask_mode`` consistently with the forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from rocke.core.ir import F32, KernelDef
from rocke.helpers.attention import apply_attention_mask, warp_xor_reduce_sum
from rocke.helpers.io import load_lane_slice_f32
from rocke.helpers.spec import kernel_name_join
from ._fmha_common import FmhaCommonSpec, FmhaKernelBuilder, validate_common_spec
from ._fmha_warp_body import WARP_SIZE


__all__ = [
    "FmhaBwdSpec",
    "build_fmha_bwd",
    "build_fmha_bwd_mfma",
    "build_fmha_bwd_split",
    "fmha_bwd_grid",
    "fmha_bwd_signature",
    "is_valid_spec",
]


@dataclass(frozen=True)
class FmhaBwdSpec:
    common: FmhaCommonSpec
    seqlen_q: int
    seqlen_k: int
    name: str = "rocke_fmha_bwd"
    # P69: when True, the kernel uses the new
    # :mod:`rocke.helpers.mfma_attention_bwd` MFMA-tiled body
    # (``mfma_attention_bwd_dq_dk_dv_inner_body``) instead of the
    # warp-distributed scalar inner. Same parity contract; ~32× density
    # for the QK / dP MFMAs once the kernel body wires it up.
    use_mfma_body: bool = False
    # P70: ``output_grad_dtype`` selects the gradient atomic accumulator
    # dtype. ``"f32"`` (default) routes through
    # ``global_atomic_add_f32``; ``"bf16"`` routes through
    # ``global_atomic_add_pk_bf16`` for halved atomic engine pressure.
    # The bf16 path is a real numerical change so callers gate on
    # parity; H=256 workloads that are atomic-engine-bound see the
    # biggest relative improvement.
    output_grad_dtype: str = "f32"
    # ``"atomic"`` (default): one Q-major kernel, dK / dV via f32 atomics
    # (:func:`build_fmha_bwd`). ``"split"``: atomic-free dQ + dK/dV kernel
    # pair (:func:`build_fmha_bwd_split`). ``"mfma"``: FA2-style MFMA
    # kernels (:func:`build_fmha_bwd_mfma`): a ``D`` / log-sum-exp pre-pass,
    # a KV-major dK/dV kernel and a Q-major dQ kernel.
    bwd_mode: str = "atomic"
    # ``bwd_mode="mfma"`` knobs. dK/dV kernel: waves per CTA and 16-key
    # tiles per wave (the CTA owns ``16 * waves * ktiles`` keys). dQ kernel:
    # waves per CTA and 16-row Q tiles per wave. ``mfma_lds_pad`` pads each
    # staged LDS row by that many elements.
    mfma_dkdv_waves: int = 4
    mfma_dkdv_ktiles: int = 1
    mfma_dq_waves: int = 4
    mfma_dq_qtiles: int = 1
    mfma_lds_pad: int = 8

    def kernel_name(self) -> str:
        s = self.common.shape
        return kernel_name_join(
            self.name,
            f"H{s.head_size}",
            f"HQ{s.num_query_heads}",
            f"HK{s.num_kv_heads}",
            self.common.dtype,
            f"Q{self.seqlen_q}",
            f"K{self.seqlen_k}",
            self.common.mask_mode,
        )


def is_valid_spec(spec: FmhaBwdSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for one FMHA backward config on ``arch``.

    The backward body is warp-distributed scalar (one wave64 warp per
    CTA, head-dim distributed across lanes): it issues no MFMA atoms and
    accumulates dQ / dK / dV via f32 global atomics, both of which are
    shared CDNA3 / CDNA4 hardware. The only architecture facts consulted
    are the per-WG thread cap and the wave size (the warp body needs
    ``head_size % wave_size == 0``), sourced from
    :class:`rocke.core.arch.ArchTarget` so an unknown arch or an
    impossible head-size / wave-size combination is rejected with a
    structured reason rather than failing later at lower time. gfx950
    behavior is unchanged.
    """
    from rocke.core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)

    ok, why = validate_common_spec(spec.common)
    if not ok:
        return False, why
    if spec.seqlen_q <= 0 or spec.seqlen_k <= 0:
        return False, (
            f"seqlen_q / seqlen_k must be > 0 (got {spec.seqlen_q}, {spec.seqlen_k})"
        )
    if spec.common.shape.head_size % target.wave_size != 0:
        return False, (
            f"fmha_bwd warp body needs head_size % {target.wave_size} == 0 "
            f"(got {spec.common.shape.head_size}) on {arch}"
        )
    if target.wave_size > target.max_threads_per_block:
        return False, (
            f"wave_size {target.wave_size} > max_threads_per_block "
            f"{target.max_threads_per_block} on {arch}"
        )
    return True, "ok"


def _declare_params(kb: FmhaKernelBuilder) -> None:
    """Declare the bwd kernel ABI (shared between build + sig).

    dQ / dK / dV are f32 accumulator tensors targeted by atomic_add;
    their head stride is implicit (== K / V head stride for dK / dV,
    == Q head stride for dQ) so the ABI only carries the *token*
    stride for each gradient tensor.
    """
    kb.add_tensor("Q", readonly=True)
    kb.add_tensor("K", readonly=True)
    kb.add_tensor("V", readonly=True)
    kb.add_tensor("dO", readonly=True)
    kb.add_ptr("M_saved", dtype="f32", readonly=True)
    kb.add_ptr("L_saved", dtype="f32", readonly=True)
    kb.add_ptr("dQ", dtype="f32", readonly=False)
    kb.add_ptr("dK", dtype="f32", readonly=False)
    kb.add_ptr("dV", dtype="f32", readonly=False)
    kb.add_scalar("scale_log2", "f32")
    kb.add_scalar("scale_inv", "f32")
    kb.add_scalar("seqlen_q", "i32")
    kb.add_scalar("seqlen_k", "i32")
    kb.add_strides("q", "k", "v")
    kb.add_strides("do")
    kb.add_scalar("stride_dq_token", "i32")
    kb.add_scalar("stride_dk_token", "i32")
    kb.add_scalar("stride_dv_token", "i32")


def build_fmha_bwd(spec: FmhaBwdSpec, arch: str = "gfx950") -> KernelDef:
    """FMHA backward kernel (warp-distributed, with mask support).

    Per CTA (one warp): lane ``t`` owns head-dim slot ``t * EPT + k``.

    1. Pre-load Q[q_token, head, lane_slice] / dO[..., lane_slice].
    2. Load saved m, l for this row.
    3. **First K-loop**: per-lane partial Q.K + dO.V, warp-reduce,
       apply mask, accumulate rowsum_dp via iter_arg.
    4. **Second K-loop**: recompute p, dp = p * (dO.V - rowsum_dp);
       atomic-add dV[k, head, lane_d] += p * dO; atomic-add dK[k, head,
       lane_d] += dp * scale * Q; per-lane register dq slice updates.
    5. Final: atomic-add per-lane dq into dQ[q_token, head, lane_d].

    The body is warp-distributed scalar (f32 global atomics + warp
    butterfly reductions) and issues **no** MFMA atoms, so it is
    arch-portable: ``arch`` is threaded through :func:`is_valid_spec`
    only to validate the per-WG thread cap / wave-size facts against
    :class:`rocke.core.arch.ArchTarget`. The emitted IR is identical
    on gfx942 and gfx950 (default ``arch="gfx950"`` is byte-for-byte
    backward-compatible).
    """
    ok, why = is_valid_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid fmha_bwd spec: {why}")
    s = spec.common
    dtype = s.dtype
    H = s.shape.head_size
    if H % WARP_SIZE != 0:
        raise ValueError(
            f"fmha_bwd warp body needs head_size % {WARP_SIZE} == 0; got {H}"
        )
    ept = H // WARP_SIZE

    kb = FmhaKernelBuilder(spec.kernel_name(), s)
    kb.block_size(WARP_SIZE)
    _declare_params(kb)
    kb.decode_grid()
    b = kb.builder

    Q = kb.tensor("Q")
    K = kb.tensor("K")
    V = kb.tensor("V")
    dO = kb.tensor("dO")
    M_saved = kb.ptr("M_saved")
    L_saved = kb.ptr("L_saved")
    dQ = kb.ptr("dQ")
    dK = kb.ptr("dK")
    dV = kb.ptr("dV")
    scale_log2 = kb.scalar("scale_log2")
    scale_inv = kb.scalar("scale_inv")
    seqlen_k = kb.scalar("seqlen_k")
    q_token = kb.q_token
    head_idx = kb.head_idx
    kv_head_idx = kb.kv_head_idx

    tid = b.thread_id_x()
    lane_d_base = b.mul(tid, b.const_i32(ept))

    q_row = kb.q_row_base()
    do_row = kb.row_base("do", q_token, head_idx)

    # Saved M / L are indexed by (q_token, head) -- linear (q * HQ + h).
    # Hoist the ``q * HQ + h`` row index so we don't rematerialise the
    # ``HQ`` constant twice.
    ml_row = b.add(b.mul(q_token, b.const_i32(s.shape.num_query_heads)), head_idx)
    m_qt = b.global_load_f32(M_saved, ml_row)
    l_qt = b.global_load_f32(L_saved, ml_row)
    inv_l = b.rcp(l_qt)

    # Pre-load Q + dO lane slices as one vectorised global load per
    # tensor (``global_load_vN`` when EPT in {2, 4, 8}, scalar fallback
    # otherwise). Mirrors CK Tile's ``load_tile`` over a distributed
    # ``rt<bf16, 16, D>`` register tile -- the per-CTA Q / dO buffers
    # are constant over the K-loop, so the loads happen once per CTA
    # and the results live in registers.
    q_lane = load_lane_slice_f32(b, Q, q_row, lane_d_base, dtype=dtype, ept=ept)
    do_lane = load_lane_slice_f32(b, dO, do_row, lane_d_base, dtype=dtype, ept=ept)

    zero_f = b.const_f32(0.0)

    # First K-loop: rowsum_dp accumulated via loop-carried iter_arg.
    k_loop_1 = b.scf_for_iter(
        b.const_i32(0),
        seqlen_k,
        b.const_i32(1),
        iter_args=[("rs", zero_f)],
        iv_name="k_idx",
    )
    with k_loop_1 as (k_idx, (rowsum_carry,)):
        k_row = kb.k_row_base(k_idx)
        v_row = kb.v_row_base(k_idx)
        # Vectorised per-lane K / V loads (one transaction each when
        # EPT in {2, 4, 8}); the QK + dO.V partial accumulators stay
        # scalar f32 because the warp-distributed body uses a
        # ``warp_xor_reduce_sum`` butterfly to fold across the warp.
        k_lane = load_lane_slice_f32(b, K, k_row, lane_d_base, dtype=dtype, ept=ept)
        v_lane = load_lane_slice_f32(b, V, v_row, lane_d_base, dtype=dtype, ept=ept)
        partial_qk = zero_f
        partial_dov = zero_f
        for k in range(ept):
            # ``fma`` guarantees ``v_fma_f32`` (single VALU + single
            # rounding) for the per-lane dot partials; the bare
            # fadd(fmul()) only fuses opportunistically. Strictly more
            # accurate (one rounding) and one fewer issue slot per term.
            partial_qk = b.fma(q_lane[k], k_lane[k], partial_qk)
            partial_dov = b.fma(do_lane[k], v_lane[k], partial_dov)
        dot_qk = warp_xor_reduce_sum(b, partial_qk, stages=6)
        dot_dov = warp_xor_reduce_sum(b, partial_dov, stages=6)
        s_log2 = b.fmul(dot_qk, scale_log2)
        s_log2 = apply_attention_mask(
            b,
            s_log2,
            mask_mode=s.mask_mode,
            k_idx=k_idx,
            query_pos=q_token,
            sliding_window=s.sliding_window,
        )
        p = b.fmul(b.exp2(b.fsub(s_log2, m_qt)), inv_l)
        b.scf_yield(b.fma(p, dot_dov, rowsum_carry))
    rowsum_dp = k_loop_1.results[0]

    # Second K-loop: atomic-add dV / dK; accumulate dQ in per-lane regs.
    iter_args2 = [(f"dq{k}", zero_f) for k in range(ept)]
    k_loop_2 = b.scf_for_iter(
        b.const_i32(0),
        seqlen_k,
        b.const_i32(1),
        iter_args=iter_args2,
        iv_name="k_idx2",
    )
    with k_loop_2 as (k_idx, dq_state):
        k_row = kb.k_row_base(k_idx)
        v_row = kb.v_row_base(k_idx)
        dk_row = b.add(
            b.mul(k_idx, kb.scalar("stride_dk_token")),
            b.mul(kv_head_idx, kb.stride_head("k")),
        )
        dv_row = b.add(
            b.mul(k_idx, kb.scalar("stride_dv_token")),
            b.mul(kv_head_idx, kb.stride_head("v")),
        )
        # Vectorised K / V loads. The QK and dO.V partials reuse the
        # same lane slice the first K-loop's pattern uses; the second
        # K-loop additionally feeds dK = dp * scale * Q (no extra K
        # touch) and the dQ accumulator update (re-uses K once per
        # K-step). One ``global_load_vN`` per tensor keeps the inner
        # loop's address-arithmetic / MEM-channel pressure as low as
        # the CK Tile ``BlockFmhaBwd*`` register-tile loaders.
        k_lane = load_lane_slice_f32(b, K, k_row, lane_d_base, dtype=dtype, ept=ept)
        v_lane = load_lane_slice_f32(b, V, v_row, lane_d_base, dtype=dtype, ept=ept)
        partial_qk = zero_f
        partial_dov = zero_f
        for k in range(ept):
            partial_qk = b.fma(q_lane[k], k_lane[k], partial_qk)
            partial_dov = b.fma(do_lane[k], v_lane[k], partial_dov)
        dot_qk = warp_xor_reduce_sum(b, partial_qk, stages=6)
        dot_dov = warp_xor_reduce_sum(b, partial_dov, stages=6)
        s_log2 = b.fmul(dot_qk, scale_log2)
        s_log2 = apply_attention_mask(
            b,
            s_log2,
            mask_mode=s.mask_mode,
            k_idx=k_idx,
            query_pos=q_token,
            sliding_window=s.sliding_window,
        )
        p = b.fmul(b.exp2(b.fsub(s_log2, m_qt)), inv_l)
        dp = b.fmul(p, b.fsub(dot_dov, rowsum_dp))
        dp_scale = b.fmul(dp, scale_inv)

        # f32 atomic adds remain scalar (no packed-f32 atomic exists
        # on gfx950; ``global_atomic_add_pk_bf16`` would help only if
        # the ABI lowered dV/dK/dQ from f32 to bf16). One scalar
        # atomic per (lane, head-dim slot) matches what CK Tile's
        # bwd kernel emits when its accumulator dtype is f32.
        new_dq = []
        for k in range(ept):
            d = b.add(lane_d_base, b.const_i32(k))
            b.global_atomic_add(dV, b.add(dv_row, d), b.fmul(p, do_lane[k]))
            b.global_atomic_add(dK, b.add(dk_row, d), b.fmul(dp_scale, q_lane[k]))
            # fma for the per-lane dQ register accumulator (loop-carried):
            # guaranteed single-rounding v_fma_f32, one fewer issue slot.
            new_dq.append(b.fma(dp_scale, k_lane[k], dq_state[k]))
        b.scf_yield(*new_dq)

    # Final: write dQ. Only this CTA writes dQ[q_token, head, :].
    dq_row = b.add(
        b.mul(q_token, kb.scalar("stride_dq_token")),
        b.mul(head_idx, kb.stride_head("q")),
    )
    for k in range(ept):
        d = b.add(lane_d_base, b.const_i32(k))
        b.global_atomic_add(dQ, b.add(dq_row, d), k_loop_2.results[k])

    b.ret()
    return kb.kernel


def fmha_bwd_grid(spec: FmhaBwdSpec) -> Tuple[int, int, int]:
    return (spec.seqlen_q, spec.common.shape.num_query_heads, 1)


def fmha_bwd_signature(spec: FmhaBwdSpec):
    kb = FmhaKernelBuilder("rocke_fmha_bwd_sig_probe", spec.common)
    _declare_params(kb)
    return kb.signature()


# ---------------------------------------------------------------------
# Split backward (default-off lever): no global atomics.
#
# The atomic body above recomputes P per (q, k) inside a Q-major CTA and
# atomic-adds every dK / dV element: ``2 * seqlen_q * seqlen_k * head_size``
# f32 atomics per head. The split form instead runs two warp-distributed
# kernels that each own their output rows:
#
#   1. ``_dq`` (grid ``(seqlen_q, HQ)``, Q-major): the same two K-loops as
#      the atomic body, but loop 1's ``rowsum_dp`` is also written to the
#      ``Dsum`` workspace and loop 2 only accumulates dQ (plain store).
#   2. ``_dkdv`` (grid ``(seqlen_k, HK)``, KV-major): loops over the query
#      rows (and the ``HQ / HK`` query heads of the KV group), recomputes P
#      from the saved ``(M, L)`` and reads ``Dsum``; dK / dV accumulate in
#      registers and are stored once.
#
# Same ABI tensors as the atomic kernel plus the ``Dsum`` f32
# ``[seqlen_q, HQ]`` workspace; dQ / dK / dV are written (not accumulated),
# so callers need not zero them. Causal masks bound the loops instead of
# masking dead iterations.
# ---------------------------------------------------------------------


def _declare_split_dq_params(kb: FmhaKernelBuilder) -> None:
    kb.add_tensor("Q", readonly=True)
    kb.add_tensor("K", readonly=True)
    kb.add_tensor("V", readonly=True)
    kb.add_tensor("dO", readonly=True)
    kb.add_ptr("M_saved", dtype="f32", readonly=True)
    kb.add_ptr("L_saved", dtype="f32", readonly=True)
    kb.add_ptr("dQ", dtype="f32", readonly=False)
    kb.add_ptr("Dsum", dtype="f32", readonly=False)
    kb.add_scalar("scale_log2", "f32")
    kb.add_scalar("scale_inv", "f32")
    kb.add_scalar("seqlen_k", "i32")
    kb.add_strides("q", "k", "v")
    kb.add_strides("do")
    kb.add_scalar("stride_dq_token", "i32")


def _declare_split_dkdv_params(kb: FmhaKernelBuilder) -> None:
    kb.add_tensor("Q", readonly=True)
    kb.add_tensor("K", readonly=True)
    kb.add_tensor("V", readonly=True)
    kb.add_tensor("dO", readonly=True)
    kb.add_ptr("M_saved", dtype="f32", readonly=True)
    kb.add_ptr("L_saved", dtype="f32", readonly=True)
    kb.add_ptr("Dsum", dtype="f32", readonly=True)
    kb.add_ptr("dK", dtype="f32", readonly=False)
    kb.add_ptr("dV", dtype="f32", readonly=False)
    kb.add_scalar("scale_log2", "f32")
    kb.add_scalar("scale_inv", "f32")
    kb.add_scalar("seqlen_q", "i32")
    kb.add_strides("q", "k", "v")
    kb.add_strides("do")
    kb.add_scalar("stride_dk_token", "i32")
    kb.add_scalar("stride_dv_token", "i32")


def _build_split_dq(spec: FmhaBwdSpec) -> KernelDef:
    s = spec.common
    dtype = s.dtype
    ept = s.shape.head_size // WARP_SIZE
    kb = FmhaKernelBuilder(spec.kernel_name() + "_split_dq", s)
    kb.block_size(WARP_SIZE)
    _declare_split_dq_params(kb)
    kb.decode_grid()
    b = kb.builder
    Q, K, V, dO = kb.tensor("Q"), kb.tensor("K"), kb.tensor("V"), kb.tensor("dO")
    scale_log2 = kb.scalar("scale_log2")
    scale_inv = kb.scalar("scale_inv")
    q_token, head_idx = kb.q_token, kb.head_idx
    tid = b.thread_id_x()
    lane_d_base = b.mul(tid, b.const_i32(ept))
    ml_row = b.add(b.mul(q_token, b.const_i32(s.shape.num_query_heads)), head_idx)
    m_qt = b.global_load_f32(kb.ptr("M_saved"), ml_row)
    inv_l = b.rcp(b.global_load_f32(kb.ptr("L_saved"), ml_row))
    q_lane = load_lane_slice_f32(b, Q, kb.q_row_base(), lane_d_base, dtype=dtype, ept=ept)
    do_lane = load_lane_slice_f32(
        b, dO, kb.row_base("do", q_token, head_idx), lane_d_base, dtype=dtype, ept=ept
    )
    zero_f = b.const_f32(0.0)
    k_hi = (
        b.add(q_token, b.const_i32(1)) if s.mask_mode == "causal" else kb.scalar("seqlen_k")
    )

    def _p_and_dov(k_idx):
        k_lane = load_lane_slice_f32(b, K, kb.k_row_base(k_idx), lane_d_base, dtype=dtype, ept=ept)
        v_lane = load_lane_slice_f32(b, V, kb.v_row_base(k_idx), lane_d_base, dtype=dtype, ept=ept)
        pqk, pdov = zero_f, zero_f
        for k in range(ept):
            pqk = b.fma(q_lane[k], k_lane[k], pqk)
            pdov = b.fma(do_lane[k], v_lane[k], pdov)
        dot_qk = warp_xor_reduce_sum(b, pqk, stages=6)
        dot_dov = warp_xor_reduce_sum(b, pdov, stages=6)
        s_log2 = apply_attention_mask(
            b,
            b.fmul(dot_qk, scale_log2),
            mask_mode=s.mask_mode,
            k_idx=k_idx,
            query_pos=q_token,
            sliding_window=s.sliding_window,
        )
        return b.fmul(b.exp2(b.fsub(s_log2, m_qt)), inv_l), dot_dov, k_lane

    loop1 = b.scf_for_iter(b.const_i32(0), k_hi, b.const_i32(1), iter_args=[("rs", zero_f)], iv_name="k_idx")
    with loop1 as (k_idx, (rs,)):
        p, dot_dov, _ = _p_and_dov(k_idx)
        b.scf_yield(b.fma(p, dot_dov, rs))
    rowsum_dp = loop1.results[0]
    with b.scf_if(b.cmp_eq(tid, b.const_i32(0))):
        b.global_store(kb.ptr("Dsum"), ml_row, rowsum_dp, align=4)

    loop2 = b.scf_for_iter(
        b.const_i32(0), k_hi, b.const_i32(1),
        iter_args=[(f"dq{k}", zero_f) for k in range(ept)], iv_name="k_idx2",
    )
    with loop2 as (k_idx, dq_state):
        p, dot_dov, k_lane = _p_and_dov(k_idx)
        dp_scale = b.fmul(b.fmul(p, b.fsub(dot_dov, rowsum_dp)), scale_inv)
        b.scf_yield(*[b.fma(dp_scale, k_lane[k], dq_state[k]) for k in range(ept)])
    dq_row = b.add(
        b.mul(q_token, kb.scalar("stride_dq_token")), b.mul(head_idx, kb.stride_head("q"))
    )
    for k in range(ept):
        b.global_store(
            kb.ptr("dQ"), b.add(dq_row, b.add(lane_d_base, b.const_i32(k))), loop2.results[k], align=4
        )
    b.ret()
    return kb.kernel


def _build_split_dkdv(spec: FmhaBwdSpec) -> KernelDef:
    s = spec.common
    dtype = s.dtype
    HQ = s.shape.num_query_heads
    G = s.shape.num_queries_per_kv
    ept = s.shape.head_size // WARP_SIZE
    kb = FmhaKernelBuilder(spec.kernel_name() + "_split_dkdv", s)
    kb.block_size(WARP_SIZE)
    _declare_split_dkdv_params(kb)
    # Grid (seqlen_k, HK): block_x = key row, block_y = KV head.
    kb.decode_grid(num_queries_per_kv=1)
    b = kb.builder
    Q, K, V, dO = kb.tensor("Q"), kb.tensor("K"), kb.tensor("V"), kb.tensor("dO")
    M_saved, L_saved, Dsum = kb.ptr("M_saved"), kb.ptr("L_saved"), kb.ptr("Dsum")
    scale_log2 = kb.scalar("scale_log2")
    scale_inv = kb.scalar("scale_inv")
    k_idx, kv_head = kb.q_token, kb.head_idx
    tid = b.thread_id_x()
    lane_d_base = b.mul(tid, b.const_i32(ept))
    k_lane = load_lane_slice_f32(b, K, kb.row_base("k", k_idx, kv_head), lane_d_base, dtype=dtype, ept=ept)
    v_lane = load_lane_slice_f32(b, V, kb.row_base("v", k_idx, kv_head), lane_d_base, dtype=dtype, ept=ept)
    zero_f = b.const_f32(0.0)
    q_lo = k_idx if s.mask_mode == "causal" else b.const_i32(0)
    iter_args = [(f"dk{k}", zero_f) for k in range(ept)] + [(f"dv{k}", zero_f) for k in range(ept)]
    loop = b.scf_for_iter(q_lo, kb.scalar("seqlen_q"), b.const_i32(1), iter_args=iter_args, iv_name="q_idx")
    with loop as (q_idx, state):
        dk = list(state[:ept])
        dv = list(state[ept:])
        for hg in range(G):
            h = b.add(b.mul(kv_head, b.const_i32(G)), b.const_i32(hg)) if G > 1 else kv_head
            q_lane = load_lane_slice_f32(b, Q, kb.row_base("q", q_idx, h), lane_d_base, dtype=dtype, ept=ept)
            do_lane = load_lane_slice_f32(b, dO, kb.row_base("do", q_idx, h), lane_d_base, dtype=dtype, ept=ept)
            ml_row = b.add(b.mul(q_idx, b.const_i32(HQ)), h)
            m_q = b.global_load_f32(M_saved, ml_row)
            inv_l = b.rcp(b.global_load_f32(L_saved, ml_row))
            d_q = b.global_load_f32(Dsum, ml_row)
            pqk, pdov = zero_f, zero_f
            for k in range(ept):
                pqk = b.fma(q_lane[k], k_lane[k], pqk)
                pdov = b.fma(do_lane[k], v_lane[k], pdov)
            dot_qk = warp_xor_reduce_sum(b, pqk, stages=6)
            dot_dov = warp_xor_reduce_sum(b, pdov, stages=6)
            s_log2 = apply_attention_mask(
                b,
                b.fmul(dot_qk, scale_log2),
                mask_mode=s.mask_mode,
                k_idx=k_idx,
                query_pos=q_idx,
                sliding_window=s.sliding_window,
            )
            p = b.fmul(b.exp2(b.fsub(s_log2, m_q)), inv_l)
            dp_scale = b.fmul(b.fmul(p, b.fsub(dot_dov, d_q)), scale_inv)
            for k in range(ept):
                dv[k] = b.fma(p, do_lane[k], dv[k])
                dk[k] = b.fma(dp_scale, q_lane[k], dk[k])
        b.scf_yield(*dk, *dv)
    dk_row = b.add(b.mul(k_idx, kb.scalar("stride_dk_token")), b.mul(kv_head, kb.stride_head("k")))
    dv_row = b.add(b.mul(k_idx, kb.scalar("stride_dv_token")), b.mul(kv_head, kb.stride_head("v")))
    for k in range(ept):
        d = b.add(lane_d_base, b.const_i32(k))
        b.global_store(kb.ptr("dK"), b.add(dk_row, d), loop.results[k], align=4)
        b.global_store(kb.ptr("dV"), b.add(dv_row, d), loop.results[ept + k], align=4)
    b.ret()
    return kb.kernel


def build_fmha_bwd_split(spec: FmhaBwdSpec, arch: str = "gfx950"):
    """Atomic-free backward: ``[(kernel, signature, grid), ...]`` in launch order.

    Both kernels must run on the same stream (``_dkdv`` reads the ``Dsum``
    workspace written by ``_dq``).
    """
    ok, why = is_valid_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid fmha_bwd spec: {why}")
    s = spec.common
    kdq = _build_split_dq(spec)
    kb = FmhaKernelBuilder("rocke_fmha_bwd_split_dq_sig_probe", s)
    _declare_split_dq_params(kb)
    sig_dq = kb.signature()
    kdkdv = _build_split_dkdv(spec)
    kb = FmhaKernelBuilder("rocke_fmha_bwd_split_dkdv_sig_probe", s)
    _declare_split_dkdv_params(kb)
    sig_dkdv = kb.signature()
    return [
        (kdq, sig_dq, (spec.seqlen_q, s.shape.num_query_heads, 1)),
        (kdkdv, sig_dkdv, (spec.seqlen_k, s.shape.num_kv_heads, 1)),
    ]


# ---------------------------------------------------------------------
# MFMA backward (default-off lever, ``bwd_mode="mfma"``).
#
# FA2-style three-kernel backward on the K-packed ``16x16x32`` MFMA atom
# (gfx950) with ``ds_read_b64_tr_b16`` transposed LDS reads:
#
#   1. ``_mfma_pre`` (grid ``(seqlen_q / 16, HQ)``): ``Dsum = rowsum(dO * O)``
#      and ``Lse2 = M + log2(L)`` (log2-domain log-sum-exp), both f32
#      ``[HQ, seqlen_q]`` workspaces.
#   2. ``_mfma_dkdv`` (KV-major): each wave keeps ``ktiles`` 16-key K / V
#      tiles in registers (the B operands of ``S = Q.K^T`` and
#      ``dP = dO.V^T``) and walks 32-row Q / dO tiles that the CTA stages in
#      double-buffered LDS. The two 16-row Q sub-tiles are row-permuted so
#      a lane's C fragment holds the 8 contiguous query rows
#      ``8 * (lane / 16) + 0..7``: P and dS feed ``dV^T += dO^T.P`` and
#      ``dK^T += Q^T.dS`` straight from registers, with dO^T / Q^T read by
#      ``ds_read_b64_tr_b16`` from the same LDS tiles.
#   3. ``_mfma_dq`` (Q-major): each wave keeps ``qtiles`` 16-row Q / dO
#      tiles in registers and walks 32-key K / V tiles staged in LDS,
#      computing ``S^T = K.Q^T``, ``dP^T = V.dO^T`` and
#      ``dQ^T += K^T.dS^T`` (K^T by transpose read).
#
# dQ / dK / dV are written once (no atomics, no zero-init). Causal masks
# bound the walks (CTA-uniform) and zero the masked P entries.
# ---------------------------------------------------------------------

_MFMA_BWD_STEP = 32  # rows per inner step: two permuted 16-row sub-tiles


def _mfma_bwd_setup(spec: FmhaBwdSpec, arch: str):
    from rocke.core.arch import ArchTarget
    from rocke.helpers.atoms import MfmaAtom
    from rocke.helpers.io import io_ir_type

    target = ArchTarget.from_gfx(arch)
    if not target.memory.has_ds_read_tr:
        raise ValueError(f"fmha_bwd mfma mode needs ds_read_*_tr_b16 (absent on {arch})")
    s = spec.common
    atom = MfmaAtom.bf16_16x16x32() if s.dtype == "bf16" else MfmaAtom.f16_16x16x32()
    return target.mma.by_op_id(atom.name), io_ir_type(s.dtype)


def _mfma_bwd_lane(b, nw: int):
    """``(tid, lane, wave, m16, g)``; ``wave`` is pinned to an SGPR."""
    tid = b.thread_id_x()
    lane = b.mod(tid, b.const_i32(64)) if nw > 1 else tid
    wave = b.to_sgpr_u32(b.div(tid, b.const_i32(64))) if nw > 1 else b.const_i32(0)
    return tid, lane, wave, b.mod(lane, b.const_i32(16)), b.div(lane, b.const_i32(16))


def _mfma_bwd_perm_row(b, m16, t: int):
    """A-operand row of 16-row sub-tile ``t`` so C row ``4g + r`` is ``8g + r + 4t``."""
    return b.add(
        b.add(b.mul(b.div(m16, b.const_i32(4)), b.const_i32(8)), b.mod(m16, b.const_i32(4))),
        b.const_i32(4 * t),
    )


class _MfmaBwdStager:
    """Cooperative double-buffered staging of ``_MFMA_BWD_STEP``-row tiles."""

    def __init__(self, b, tid, nw: int, head_size: int, dt, pad: int, srcs):
        lanes_per_row = head_size // 8
        rows_per_pass = 64 * nw // lanes_per_row
        if rows_per_pass > _MFMA_BWD_STEP or _MFMA_BWD_STEP % rows_per_pass:
            raise ValueError(f"mfma bwd staging: {nw} waves do not tile {_MFMA_BWD_STEP} rows")
        self.b, self.dt = b, dt
        self.srcs = srcs  # [(tensor, row_base_fn(row) -> elem offset)]
        self.rows = [
            b.add(b.div(tid, b.const_i32(lanes_per_row)), b.const_i32(p * rows_per_pass))
            for p in range(_MFMA_BWD_STEP // rows_per_pass)
        ]
        self.col = b.mul(b.mod(tid, b.const_i32(lanes_per_row)), b.const_i32(8))
        self.lds = [
            b.smem_alloc(dt, [2, _MFMA_BWD_STEP, head_size + pad], name_hint=f"bwst{i}")
            for i in range(len(srcs))
        ]

    def load(self, base, *ctx):
        b = self.b
        return [
            [
                b.global_load_vN(src, b.add(fn(b.add(base, r), *ctx), self.col), self.dt, 8, align=16)
                for r in self.rows
            ]
            for src, fn in self.srcs
        ]

    def store(self, vals, buf):
        for lds, vs in zip(self.lds, vals):
            for r, v in zip(self.rows, vs):
                self.b.smem_store_vN(lds, [buf, r, self.col], v, 8)


def _mfma_bwd_tr8(b, lds, buf, g, m16, nd: int, dt):
    """Lane ``(g, m16)`` gets ``lds[buf, 8g + j, 16 nd + m16]`` for ``j = 0..7``."""
    col = b.add(b.const_i32(16 * nd), b.mul(b.mod(m16, b.const_i32(4)), b.const_i32(4)))
    r0 = b.add(b.mul(g, b.const_i32(8)), b.div(m16, b.const_i32(4)))
    lo = b.ds_read_tr16_b64(lds, buf, r0, col, dtype=dt)
    hi = b.ds_read_tr16_b64(lds, buf, b.add(r0, b.const_i32(4)), col, dtype=dt)
    return b.vec_concat(lo, hi)


def _mfma_bwd_pack(b, vals, dt):
    v = b.zero_vec(dt, 8)
    for j, x in enumerate(vals):
        v = b.vec_insert(v, b.cast_f32_to(x, dt), j)
    return v


def _declare_mfma_pre_params(kb: FmhaKernelBuilder) -> None:
    kb.add_tensor("O", readonly=True)
    kb.add_tensor("dO", readonly=True)
    kb.add_ptr("M_saved", dtype="f32", readonly=True)
    kb.add_ptr("L_saved", dtype="f32", readonly=True)
    kb.add_ptr("Dsum", dtype="f32", readonly=False)
    kb.add_ptr("Lse2", dtype="f32", readonly=False)
    kb.add_scalar("seqlen_q", "i32")
    kb.add_strides("o", "do")


def _declare_mfma_dkdv_params(kb: FmhaKernelBuilder) -> None:
    kb.add_tensor("Q", readonly=True)
    kb.add_tensor("K", readonly=True)
    kb.add_tensor("V", readonly=True)
    kb.add_tensor("dO", readonly=True)
    kb.add_ptr("Lse2", dtype="f32", readonly=True)
    kb.add_ptr("Dsum", dtype="f32", readonly=True)
    kb.add_ptr("dK", dtype="f32", readonly=False)
    kb.add_ptr("dV", dtype="f32", readonly=False)
    kb.add_scalar("scale_log2", "f32")
    kb.add_scalar("scale_inv", "f32")
    kb.add_scalar("seqlen_q", "i32")
    kb.add_strides("q", "k", "v")
    kb.add_strides("do")
    kb.add_scalar("stride_dk_token", "i32")
    kb.add_scalar("stride_dv_token", "i32")


def _declare_mfma_dq_params(kb: FmhaKernelBuilder) -> None:
    kb.add_tensor("Q", readonly=True)
    kb.add_tensor("K", readonly=True)
    kb.add_tensor("V", readonly=True)
    kb.add_tensor("dO", readonly=True)
    kb.add_ptr("Lse2", dtype="f32", readonly=True)
    kb.add_ptr("Dsum", dtype="f32", readonly=True)
    kb.add_ptr("dQ", dtype="f32", readonly=False)
    kb.add_scalar("scale_log2", "f32")
    kb.add_scalar("scale_inv", "f32")
    kb.add_scalar("seqlen_q", "i32")
    kb.add_scalar("seqlen_k", "i32")
    kb.add_strides("q", "k", "v")
    kb.add_strides("do")
    kb.add_scalar("stride_dq_token", "i32")


_MFMA_PRE_THREADS = 256


def _build_mfma_pre(spec: FmhaBwdSpec, arch: str) -> KernelDef:
    s = spec.common
    _, dt = _mfma_bwd_setup(spec, arch)
    HQ = s.shape.num_query_heads
    lanes_per_row = s.shape.head_size // 8
    kb = FmhaKernelBuilder(spec.kernel_name() + "_mfma_pre", s)
    kb.block_size(_MFMA_PRE_THREADS)
    _declare_mfma_pre_params(kb)
    b = kb.builder
    rows = _MFMA_PRE_THREADS // lanes_per_row
    q_base = b.to_sgpr_u32(b.mul(b.block_id_x(), b.const_i32(rows)))
    h = b.to_sgpr_u32(b.block_id_y())
    tid = b.thread_id_x()
    q = b.add(q_base, b.div(tid, b.const_i32(lanes_per_row)))
    sub = b.mod(tid, b.const_i32(lanes_per_row))
    col = b.mul(sub, b.const_i32(8))
    o8 = b.global_load_vN(kb.tensor("O"), b.add(kb.row_base("o", q, h), col), dt, 8, align=16)
    do8 = b.global_load_vN(kb.tensor("dO"), b.add(kb.row_base("do", q, h), col), dt, 8, align=16)
    acc = b.const_f32(0.0)
    for j in range(8):
        acc = b.fma(b.cast_to_f32(b.vec_extract(o8, j)), b.cast_to_f32(b.vec_extract(do8, j)), acc)
    m = 1
    while m < lanes_per_row:
        acc = b.fadd(acc, b.warp_shuffle_xor(acc, m))
        m *= 2
    with b.scf_if(b.cmp_eq(sub, b.const_i32(0))):
        ws = b.add(b.mul(h, kb.scalar("seqlen_q")), q)
        ml = b.add(b.mul(q, b.const_i32(HQ)), h)
        lse2 = b.fadd(
            b.global_load_f32(kb.ptr("M_saved"), ml),
            b.log2(b.global_load_f32(kb.ptr("L_saved"), ml)),
        )
        b.global_store(kb.ptr("Dsum"), ws, acc, align=4)
        b.global_store(kb.ptr("Lse2"), ws, lse2, align=4)
    b.ret()
    return kb.kernel


def _mfma_bwd_tag(spec: FmhaBwdSpec, which: str) -> str:
    if which == "dkdv":
        return f"_mfma_dkdv_w{spec.mfma_dkdv_waves}k{spec.mfma_dkdv_ktiles}p{spec.mfma_lds_pad}"
    return f"_mfma_dq_w{spec.mfma_dq_waves}q{spec.mfma_dq_qtiles}p{spec.mfma_lds_pad}"


def _build_mfma_dkdv(spec: FmhaBwdSpec, arch: str) -> KernelDef:
    s = spec.common
    op, dt = _mfma_bwd_setup(spec, arch)
    D = s.shape.head_size
    G = s.shape.num_queries_per_kv
    nw, kpw = spec.mfma_dkdv_waves, spec.mfma_dkdv_ktiles
    causal = s.mask_mode == "causal"
    BM = _MFMA_BWD_STEP
    n_a, n_d = D // 32, D // 16
    kb = FmhaKernelBuilder(spec.kernel_name() + _mfma_bwd_tag(spec, "dkdv"), s)
    kb.block_size(64 * nw)
    _declare_mfma_dkdv_params(kb)
    b = kb.builder
    Q, K, V, dO = kb.tensor("Q"), kb.tensor("K"), kb.tensor("V"), kb.tensor("dO")
    Lse2, Dsum = kb.ptr("Lse2"), kb.ptr("Dsum")
    scale_log2, seqlen_q = kb.scalar("scale_log2"), kb.scalar("seqlen_q")
    n_cta = b.to_sgpr_u32(b.mul(b.block_id_x(), b.const_i32(16 * kpw * nw)))
    kvh = b.to_sgpr_u32(b.block_id_y())
    tid, _, wave, m16, g = _mfma_bwd_lane(b, nw)
    n_wave = b.add(n_cta, b.mul(wave, b.const_i32(16 * kpw)))

    # Register-resident B operands: lane (g, m16) holds K / V[n + m16, 32a + 8g + 0..7].
    def _kv_b(src, name, i):
        row = kb.row_base(name, b.add(n_wave, b.add(b.const_i32(16 * i), m16)), kvh)
        return [
            b.global_load_vN(src, b.add(row, b.add(b.const_i32(32 * a), b.mul(g, b.const_i32(8)))), dt, 8, align=16)
            for a in range(n_a)
        ]

    k_b = [_kv_b(K, "k", i) for i in range(kpw)]
    v_b = [_kv_b(V, "v", i) for i in range(kpw)]

    stg = _MfmaBwdStager(
        b, tid, nw, D, dt, spec.mfma_lds_pad,
        [(Q, lambda r, h: kb.row_base("q", r, h)), (dO, lambda r, h: kb.row_base("do", r, h))],
    )
    q_lds, do_lds = stg.lds
    nqt = b.div(seqlen_q, b.const_i32(BM))
    qt_lo = b.div(n_cta, b.const_i32(BM)) if causal else b.const_i32(0)
    nq_eff = b.sub(nqt, qt_lo)
    total = b.mul(nq_eff, b.const_i32(G))

    def _coords(it):
        hg = b.div(it, nq_eff) if G > 1 else None
        h = b.add(b.mul(kvh, b.const_i32(G)), hg) if G > 1 else kvh
        qt = b.add(qt_lo, b.mod(it, nq_eff)) if G > 1 else b.add(qt_lo, it)
        return h, b.mul(qt, b.const_i32(BM))

    h0, q00 = _coords(b.const_i32(0))
    stg.store(stg.load(q00, h0), b.const_i32(0))
    b.sync()

    iter_args = []
    for i in range(kpw):
        iter_args += [(f"dk{i}_{n}", b.zero_vec_f32(4)) for n in range(n_d)]
        iter_args += [(f"dv{i}_{n}", b.zero_vec_f32(4)) for n in range(n_d)]
    last = b.sub(total, b.const_i32(1))
    loop = b.scf_for_iter(
        b.const_i32(0), total, b.const_i32(1), iter_args=iter_args, iv_name="it",
        elide_trailing_barrier=False,
    )
    with loop as (it, state):
        dk = [list(state[2 * n_d * i : 2 * n_d * i + n_d]) for i in range(kpw)]
        dv = [list(state[2 * n_d * i + n_d : 2 * n_d * (i + 1)]) for i in range(kpw)]
        buf = b.mod(it, b.const_i32(2))
        nxt = b.add(it, b.const_i32(1))
        nxt = b.select(b.cmp_lt(nxt, total), nxt, last)
        hn, qn = _coords(nxt)
        pre = stg.load(qn, hn)
        h, q0 = _coords(it)

        q_a = [
            [b.smem_load_vN(q_lds, buf, _mfma_bwd_perm_row(b, m16, t),
                            b.add(b.const_i32(32 * a), b.mul(g, b.const_i32(8))), dtype=dt, n=8)
             for a in range(n_a)]
            for t in range(2)
        ]
        do_a = [
            [b.smem_load_vN(do_lds, buf, _mfma_bwd_perm_row(b, m16, t),
                            b.add(b.const_i32(32 * a), b.mul(g, b.const_i32(8))), dtype=dt, n=8)
             for a in range(n_a)]
            for t in range(2)
        ]
        q_lane = b.add(q0, b.mul(g, b.const_i32(8)))  # slot (t, r) -> query q_lane + 4t + r
        ws = b.add(b.mul(h, seqlen_q), q_lane)
        lse = [b.global_load_vN(Lse2, b.add(ws, b.const_i32(4 * t)), F32, 4, align=16) for t in range(2)]
        dsum = [b.global_load_vN(Dsum, b.add(ws, b.const_i32(4 * t)), F32, 4, align=16) for t in range(2)]
        p_vec, ds_vec = [], []
        for i in range(kpw):
            key = b.add(n_wave, b.add(b.const_i32(16 * i), m16))
            ps, dss = [], []
            for t in range(2):
                s_acc = b.zero_vec_f32(4)
                dp_acc = b.zero_vec_f32(4)
                for a in range(n_a):
                    s_acc = b.mma(op, q_a[t][a], k_b[i][a], s_acc)
                for a in range(n_a):
                    dp_acc = b.mma(op, do_a[t][a], v_b[i][a], dp_acc)
                for r in range(4):
                    p = b.exp2(b.fsub(b.fmul(b.vec_extract(s_acc, r), scale_log2), b.vec_extract(lse[t], r)))
                    if causal:
                        qpos = b.add(q_lane, b.const_i32(4 * t + r))
                        p = b.select(b.cmp_le(key, qpos), p, b.const_f32(0.0))
                    ps.append(p)
                    dss.append(b.fmul(p, b.fsub(b.vec_extract(dp_acc, r), b.vec_extract(dsum[t], r))))
            p_vec.append(_mfma_bwd_pack(b, ps, dt))
            ds_vec.append(_mfma_bwd_pack(b, dss, dt))
        for n in range(n_d):
            do_t = _mfma_bwd_tr8(b, do_lds, buf, g, m16, n, dt)
            for i in range(kpw):
                dv[i][n] = b.mma(op, do_t, p_vec[i], dv[i][n])
            q_t = _mfma_bwd_tr8(b, q_lds, buf, g, m16, n, dt)
            for i in range(kpw):
                dk[i][n] = b.mma(op, q_t, ds_vec[i], dk[i][n])
        stg.store(pre, b.sub(b.const_i32(1), buf))
        b.sync()
        b.scf_yield(*[x for i in range(kpw) for x in dk[i] + dv[i]])

    # C fragment of dV^T / dK^T: lane (g, m16) slot r = [d = 16n + 4g + r, key = m16].
    scale_inv = kb.scalar("scale_inv")
    res = loop.results
    for i in range(kpw):
        key = b.add(n_wave, b.add(b.const_i32(16 * i), m16))
        dk_row = b.add(b.mul(key, kb.scalar("stride_dk_token")), b.mul(kvh, kb.stride_head("k")))
        dv_row = b.add(b.mul(key, kb.scalar("stride_dv_token")), b.mul(kvh, kb.stride_head("v")))
        for n in range(n_d):
            d = b.add(b.const_i32(16 * n), b.mul(g, b.const_i32(4)))
            dk_acc = res[2 * n_d * i + n]
            dk_out = b.zero_vec_f32(4)
            for r in range(4):
                dk_out = b.vec_insert(dk_out, b.fmul(b.vec_extract(dk_acc, r), scale_inv), r)
            b.global_store_vN(kb.ptr("dK"), b.add(dk_row, d), dk_out, 4, align=16)
            b.global_store_vN(kb.ptr("dV"), b.add(dv_row, d), res[2 * n_d * i + n_d + n], 4, align=16)
    b.ret()
    return kb.kernel


def _build_mfma_dq(spec: FmhaBwdSpec, arch: str) -> KernelDef:
    s = spec.common
    op, dt = _mfma_bwd_setup(spec, arch)
    D = s.shape.head_size
    G = s.shape.num_queries_per_kv
    nw, qpw = spec.mfma_dq_waves, spec.mfma_dq_qtiles
    causal = s.mask_mode == "causal"
    BK = _MFMA_BWD_STEP
    BQ = 16 * qpw * nw
    n_a, n_d = D // 32, D // 16
    kb = FmhaKernelBuilder(spec.kernel_name() + _mfma_bwd_tag(spec, "dq"), s)
    kb.block_size(64 * nw)
    _declare_mfma_dq_params(kb)
    b = kb.builder
    Q, K, V, dO = kb.tensor("Q"), kb.tensor("K"), kb.tensor("V"), kb.tensor("dO")
    scale_log2, seqlen_q = kb.scalar("scale_log2"), kb.scalar("seqlen_q")
    q_cta = b.to_sgpr_u32(b.mul(b.block_id_x(), b.const_i32(BQ)))
    h = b.to_sgpr_u32(b.block_id_y())
    kvh = b.to_sgpr_u32(b.div(h, b.const_i32(G))) if G > 1 else h
    tid, _, wave, m16, g = _mfma_bwd_lane(b, nw)
    q_wave = b.add(q_cta, b.mul(wave, b.const_i32(16 * qpw)))
    d_lane = b.mul(g, b.const_i32(8))

    # Register-resident B operands: lane (g, m16) holds Q / dO[q + m16, 32a + 8g + 0..7].
    q_b, do_b, lse, dsum, q_pos = [], [], [], [], []
    for i in range(qpw):
        q = b.add(q_wave, b.add(b.const_i32(16 * i), m16))
        q_pos.append(q)
        qr, dr = kb.row_base("q", q, h), kb.row_base("do", q, h)
        q_b.append([b.global_load_vN(Q, b.add(qr, b.add(b.const_i32(32 * a), d_lane)), dt, 8, align=16)
                    for a in range(n_a)])
        do_b.append([b.global_load_vN(dO, b.add(dr, b.add(b.const_i32(32 * a), d_lane)), dt, 8, align=16)
                     for a in range(n_a)])
        ws = b.add(b.mul(h, seqlen_q), q)
        lse.append(b.global_load_f32(kb.ptr("Lse2"), ws))
        dsum.append(b.global_load_f32(kb.ptr("Dsum"), ws))

    stg = _MfmaBwdStager(
        b, tid, nw, D, dt, spec.mfma_lds_pad,
        [(K, lambda r: kb.row_base("k", r, kvh)), (V, lambda r: kb.row_base("v", r, kvh))],
    )
    k_lds, v_lds = stg.lds
    nkt = b.div(kb.scalar("seqlen_k"), b.const_i32(BK))
    if causal:
        need = b.add(b.div(b.add(q_cta, b.const_i32(BQ - 1)), b.const_i32(BK)), b.const_i32(1))
        nkt = b.select(b.cmp_lt(need, nkt), need, nkt)
    stg.store(stg.load(b.const_i32(0)), b.const_i32(0))
    b.sync()
    last = b.sub(nkt, b.const_i32(1))
    iter_args = [(f"dq{i}_{n}", b.zero_vec_f32(4)) for i in range(qpw) for n in range(n_d)]
    loop = b.scf_for_iter(
        b.const_i32(0), nkt, b.const_i32(1), iter_args=iter_args, iv_name="kt",
        elide_trailing_barrier=False,
    )
    with loop as (kt, state):
        dq = [list(state[n_d * i : n_d * (i + 1)]) for i in range(qpw)]
        buf = b.mod(kt, b.const_i32(2))
        nxt = b.add(kt, b.const_i32(1))
        nxt = b.select(b.cmp_lt(nxt, nkt), nxt, last)
        pre = stg.load(b.mul(nxt, b.const_i32(BK)))
        k0 = b.mul(kt, b.const_i32(BK))
        k_a = [
            [b.smem_load_vN(k_lds, buf, _mfma_bwd_perm_row(b, m16, t),
                            b.add(b.const_i32(32 * a), d_lane), dtype=dt, n=8)
             for a in range(n_a)]
            for t in range(2)
        ]
        v_a = [
            [b.smem_load_vN(v_lds, buf, _mfma_bwd_perm_row(b, m16, t),
                            b.add(b.const_i32(32 * a), d_lane), dtype=dt, n=8)
             for a in range(n_a)]
            for t in range(2)
        ]
        k_lane = b.add(k0, d_lane)  # slot (t, r) -> key k_lane + 4t + r
        ds_vec = []
        for i in range(qpw):
            dss = []
            for t in range(2):
                s_acc = b.zero_vec_f32(4)
                dp_acc = b.zero_vec_f32(4)
                for a in range(n_a):
                    s_acc = b.mma(op, k_a[t][a], q_b[i][a], s_acc)
                for a in range(n_a):
                    dp_acc = b.mma(op, v_a[t][a], do_b[i][a], dp_acc)
                for r in range(4):
                    p = b.exp2(b.fsub(b.fmul(b.vec_extract(s_acc, r), scale_log2), lse[i]))
                    if causal:
                        key = b.add(k_lane, b.const_i32(4 * t + r))
                        p = b.select(b.cmp_le(key, q_pos[i]), p, b.const_f32(0.0))
                    dss.append(b.fmul(p, b.fsub(b.vec_extract(dp_acc, r), dsum[i])))
            ds_vec.append(_mfma_bwd_pack(b, dss, dt))
        for n in range(n_d):
            k_t = _mfma_bwd_tr8(b, k_lds, buf, g, m16, n, dt)
            for i in range(qpw):
                dq[i][n] = b.mma(op, k_t, ds_vec[i], dq[i][n])
        stg.store(pre, b.sub(b.const_i32(1), buf))
        b.sync()
        b.scf_yield(*[x for i in range(qpw) for x in dq[i]])

    # C fragment of dQ^T: lane (g, m16) slot r = [d = 16n + 4g + r, query = m16].
    scale_inv = kb.scalar("scale_inv")
    for i in range(qpw):
        row = b.add(b.mul(q_pos[i], kb.scalar("stride_dq_token")), b.mul(h, kb.stride_head("q")))
        for n in range(n_d):
            acc = loop.results[n_d * i + n]
            out = b.zero_vec_f32(4)
            for r in range(4):
                out = b.vec_insert(out, b.fmul(b.vec_extract(acc, r), scale_inv), r)
            b.global_store_vN(
                kb.ptr("dQ"), b.add(row, b.add(b.const_i32(16 * n), b.mul(g, b.const_i32(4)))), out, 4, align=16
            )
    b.ret()
    return kb.kernel


def build_fmha_bwd_mfma(spec: FmhaBwdSpec, arch: str = "gfx950"):
    """MFMA backward: ``[(kernel, signature, grid, block), ...]`` in launch order.

    Pre-pass, then dK/dV, then dQ, all on one stream. Extra ABI vs the
    split mode: the forward output ``O`` (pre-pass) and the f32
    ``Lse2`` / ``Dsum`` ``[HQ, seqlen_q]`` workspaces.
    """
    ok, why = is_valid_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid fmha_bwd spec: {why}")
    s = spec.common
    D = s.shape.head_size
    HQ, HK = s.shape.num_query_heads, s.shape.num_kv_heads
    bn = 16 * spec.mfma_dkdv_waves * spec.mfma_dkdv_ktiles
    bq = 16 * spec.mfma_dq_waves * spec.mfma_dq_qtiles
    rows_pre = _MFMA_PRE_THREADS // (D // 8)
    if D % 32 or spec.seqlen_k % bn or spec.seqlen_q % bq or spec.seqlen_q % _MFMA_BWD_STEP:
        raise ValueError(
            f"fmha_bwd mfma: head_size % 32, seqlen_k % {bn}, seqlen_q % {max(bq, 32)} must be 0"
        )
    if s.mask_mode not in ("none", "causal") or spec.seqlen_q != spec.seqlen_k and s.mask_mode == "causal":
        raise ValueError("fmha_bwd mfma: mask_mode none, or causal with seqlen_q == seqlen_k")
    out = []
    for which, build, declare, grid, block in (
        ("pre", _build_mfma_pre, _declare_mfma_pre_params, (spec.seqlen_q // rows_pre, HQ, 1), _MFMA_PRE_THREADS),
        ("dkdv", _build_mfma_dkdv, _declare_mfma_dkdv_params, (spec.seqlen_k // bn, HK, 1),
         64 * spec.mfma_dkdv_waves),
        ("dq", _build_mfma_dq, _declare_mfma_dq_params, (spec.seqlen_q // bq, HQ, 1), 64 * spec.mfma_dq_waves),
    ):
        k = build(spec, arch)
        kb = FmhaKernelBuilder(f"rocke_fmha_bwd_mfma_{which}_sig_probe", s)
        declare(kb)
        out.append((k, kb.signature(), grid, (block, 1, 1)))
    return out
