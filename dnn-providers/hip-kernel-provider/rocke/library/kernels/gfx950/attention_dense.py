"""Dense flash-attention prefill kernel for gfx950 (MI355X).

Productized from the ``flash_dense_dualwave_swp`` experiment
(``kernels/gfx950/experiments/flash_dense_dualwave_swp/``). This is the shippable
step-1 pipeline with every WINNING lever baked in as always-on (no env gates):

  * **CK-1 transposed PV** — P feeds the PV MFMA in its native QK-output layout via a
    half-local V load (``pv32_v_load_paired``); the cross-half P-relayout shuffle is
    gone (~96 ``ds_bpermute`` removed). +35% over the pre-CK-1 winner.
  * **LDS bank-conflict padding on K** (``[NBUF, BN, D+8]``) — kills the 8-way conflict
    on the QK K-reads. The dominant base win (+80% over the naive baseline).
  * **native exp2_fast** (``v_exp_f32``, no overflow guard — the softmax argument is
    bounded: at most the lazy-rescale threshold plus a small fma rounding residue) — one
    instruction per exp.
  * **full-population ``sched_group_barrier`` template** naming DS_READ/MFMA/VALU/TRANS
    per PV step.
  * **diagonal-only causal masking** — a mask-free body loop over below-diagonal KV
    tiles (~94% at Sq=8192) plus a masked diagonal tail.
  * **depth-1 cluster split** fusing exp2 into the PV MFMA loop for MFMA/VALU co-exec.
  * **partial-vmcnt software prefetch** — the per-tile K/V DMA drain is a partial
    `vmcnt` (keeps the freshest V prefetch in flight across the barrier) instead of a
    full `vmcnt(0)` serialize; raises MfmaUtil, bit-identical.
  * **PV-only `s_setprio`** — the PV MFMA cluster is bracketed at raised priority so
    it wins issue slots; paired with the prefetch this is a measured ~+3.5%.
  * **vectorized O store**.
  * **gfx950 wide LDS DMA** (qualified D128/BN64 persistent path) — two
    ``buffer_load_dwordx4 ... lds`` operations per operand/wave feed 520/544-half
    slab-padded K/V layouts. IGLP-1 owns the wide-path loop schedule and
    K-major PV traversal keeps it at zero spill.

Measured on MI355X (bf16, D=128, causal, 128/8 GQA, Sq=8192, 0 spill, err ~1.46e-3
vs SDPA). Absolute TFLOPS swing +/-25-30% with auto-clock, so only SAME-SESSION
ratios are load-bearing; one representative session, each number pinned to its
config (grid / decode / V-pad / lazy):

    default grid            (one-CTA/q-block, V-pad 32, lazy on) : ~543 TFLOPS
    persistent baseline     (qb-major,        V-pad 0,  lazy off): ~877 TFLOPS
    persistent + V-pad      (qb-major,        V-pad 32, lazy off): ~912 TFLOPS
    persistent (SHIPPED)    (hkv-major,       V-pad 32, lazy on) : ~948 TFLOPS

Clock-invariant deltas (the load-bearing part): hkv/qb ~1.04x, V-pad 0->32 ~+5%,
lazy ~+2%. Shape (batch/seqlen/heads/head_dim) is baked at build time (dense,
compile-time-sized ABI); query/KV tile geometry, LDS V-row padding, occupancy hint,
and persistent knobs live on ``Gfx950AttentionDenseSpec``. Lazy online-softmax rescale
(skip the O/l rescale when every lane's tile-max is within 8 log2 of the running max)
is ALWAYS-ON by default
(``lazy_rescale=True``): parity-identical (1.46e-3) and ~+2%.

Head-size / seqlen coverage:
  * ``head_size`` is 64 or 128 (bf16/fp16, MHA + GQA incl. non-power-of-2 NQK).
    D=128 uses the per-row padded LDS fast path (1 K/V row per async-DMA instr).
    D=64 packs 2 rows per instr (64 lanes x 2 bf16 = 128 elems = 2 D=64 rows),
    which rules out a per-row pad because a padded row is not contiguous with
    the next -- so K instead pads between DMA row-GROUPS (``lds_k_group_pad``):
    the group stays contiguous for the DMA while the group pitch restores the
    QK read's bank spread. V still uses the unpadded packed pitch, so a
    conflict-free transposed-V layout remains an open D=64 lever. D=128 codegen
    is byte-identical across this change (same IR hash -> same TFLOPS).
  * ``seqlen_q``/``seqlen_kv`` must be a multiple of 256 / ``block_n`` on the
    default (aligned) kernel. Non-multiple lengths are handled by a SEPARATE
    ``ragged=True`` kernel path (its own kernel_name) that pads the boundary
    tiles ON-CHIP: OOB query rows load as 0 via a bounds-checked buffer load
    (register pad), OOB keys load as 0 into LDS (LDS pad), the grid/work-item
    count is ceil'd to cover the partial last query block, and the partial O rows
    are dropped by a guarded store. Causal needs no key mask (padded ktok >=
    seqlen_kv > every real query, so causal drops them); non-causal adds a
    ktok<seqlen_kv key mask. Self-attention only. The aligned path is emitted
    byte-identically when ``ragged=False`` (no TFLOPS impact).

Experimental/negative levers from the sweep (step-2 8-cluster, K-staging, per-nsub
staging, score truncation, PV V-prefetch) are intentionally NOT carried over — see
the experiment's ``plan.md`` for their measured results.
"""

import hashlib
import math
from contextlib import nullcontext as _nullcontext
from dataclasses import dataclass, field, fields as _dataclass_fields
from types import MappingProxyType
from typing import Optional, Tuple

from rocke.core.ir import IRBuilder, KernelDef, PtrType, F32, I32, I64
from rocke.helpers.attention import mfma_32x32x16_for_dtype, pv32_v_load_paired
from rocke.helpers.schedule import MFMA, VALU, TRANS, DS_READ
from kernels.common.attention_dense_spec import (
    AttentionDenseSpec as _AttentionDenseSpecBase,
    DENSE_TILE_GEOMETRIES,
    attention_dense_cache_key,
    check_dense_spec_preflight,
)
from kernels.gfx950.attention_tiled_2d import _mfma_32x32_c_row, _mfma_32x32_c_col
from kernels.common._lse_store import (
    dense_row_has_key,
    lse_store,
    validate_dense_lse,
    zero_keyless_row,
)

LOG2E = 1.4426950408889634

# gfx950 LDS layout policy is separate from shared query/KV tile geometry.
GFX950_DENSE_LAYOUTS = MappingProxyType(
    {"default": MappingProxyType({"lds_v_row_pad": 32})}
)
_DEFAULT_GFX950_LAYOUT = GFX950_DENSE_LAYOUTS["default"]

# Shipped defaults of the performance knobs on Gfx950AttentionDenseSpec.
#   lds_num_buffers=2: K/V double buffer. The prologue primes exactly two tiles,
#     and NBUF=3 is a measured dead end (256 VGPR + 58 spills).
#   lds_k_row_pad=8: D128 K-row padding, the +80% bank-conflict fix. Separate
#     from the inherited D<128 lds_k_group_pad so neither re-sweep moves the other.
_DEFAULT_LDS_NUM_BUFFERS = 2
_DEFAULT_LDS_K_ROW_PAD = 8
# lds_v_row_pad: bf16 elements of V-row padding for the transposed PV read
#   (ds_read_b64_tr_b16). The transpose read has a stricter bank pattern than
#   K's ds_read_b128, so it needs a LARGER pad than the K pad (8): a measured
#   sweep @ GQA-8 S=8192 gives conflicts {VPAD0: 30, VPAD8: 29, VPAD16: 11,
#   VPAD32: 0} and TFLOPS {906, 901, 944, 953} -- i.e. +8 is useless here and
#   only +32 fully clears the V-read conflicts (matches flyDSL's SMEM_V_PAD).
# Lazy-rescale re-anchor threshold in the log2 domain: skip the O/l rescale when
# every lane's (tile_max - running_max) <= this. exp2(8)=256 bounds P (times the
# ordinary kernel's small fma rounding residue; see _MAX_SCALE), so the spec
# rejects anything larger.
_DEFAULT_LAZY_RESCALE_THRESHOLD = 8.0
# sched_barrier mask: 11 instruction-class bits (__builtin_amdgcn_sched_barrier).
_SCHED_BARRIER_MASK_MAX = 0x7FF
# sched_barrier mask letting only VALU (0x2), SALU (0x4) and TRANS (0x400)
# cross: MFMAs and LDS reads keep their program order around it.
_SCHED_ALU_ONLY = 0x406
_PV_LOOP_ORDERS = frozenset({"d_major", "k_major"})
_IGLP_OFF = -1
# Accepted softmax-scale range, a chosen safety margin (not a hardware limit).
# The ordinary kernel takes the row max on unscaled scores (valid only for
# scale > 0) and computes exp2(fma(s, qk_scale, -m)) with m = fl(max * qk_scale).
# For the row-max element that argument is the product's rounding residue, at
# most half an ulp of max * qk_scale, so a large scale or score lets it grow
# until P overflows (first in the fp16 cast before the PV MFMA). At scale 2**4
# and |raw score| <= 1e6 the residue is at most 1, so P stays within about
# 2**9 even on top of the lazy-rescale threshold. The bounds also keep the
# -2**99 mask sentinel exact, finite and far below real scores after the scale.
# Scales outside [2**-64, 2**4], including NaN, +-inf and scale <= 0, are
# rejected rather than mis-computed. The hipDNN matcher mirrors this range as
# the literals 0x1p-64F / 0x1p4F in Gfx950AttentionDenseNative.cpp
# (gfx950AttentionDenseGraphMatches) and its gtest; change them together.
# TestScaleValidation pins both bounds to those literals.
#
# KNOWN LIMITATION (ordinary grid only): nothing bounds the raw score, so the
# residue above is bounded only by the size of the log2-domain row max m, not by
# the scale. Once |m| >= 2**28 (pre-softmax logit |q.k * scale| >~ 1.9e8) the
# residue can exceed 8 and fp16 P overflows; once |m| >= 2**31 (logit >~ 1.5e9)
# it can exceed 120 and exp2 itself overflows, for any dtype. The output is then
# inf/NaN. Real logits are many orders of magnitude smaller, so this is accepted.
# The persistent grid (which scales each fp32 score before the max), and this
# kernel's earlier form (Q pre-scaled before the MFMA), use the same rounded score
# for the max and every exponent, so the row max's exp2 argument is exactly 0 and
# the output stays finite at any magnitude. If this limit is ever hit, apply the
# persistent grid's form here: multiply each fp32 score by qk_scale after the QK
# MFMA and use exp2(s - m). That costs the ordinary path a little speed.
_MIN_SCALE = 2.0**-64
_MAX_SCALE = 2.0**4


@dataclass(frozen=True)
class Gfx950AttentionDenseSpec(_AttentionDenseSpecBase):
    """gfx950 dense-attention spec and architecture-specific codegen policy."""

    lds_v_row_pad: int = _DEFAULT_GFX950_LAYOUT["lds_v_row_pad"]
    wide_lds_dma: bool = False
    emit_lse: bool = field(default=False, kw_only=True)

    # Performance-only codegen knobs. Every legal value computes the same
    # attention output; defaults reproduce the shipped kernel byte-for-byte.
    # ``None`` resolves through the shipped per-path policy (``resolved_*``).
    lds_num_buffers: int = field(default=_DEFAULT_LDS_NUM_BUFFERS, kw_only=True)
    # D128 K-row pad (one row per DMA instr); inert at D64, which uses
    # lds_k_group_pad between packed row-groups instead.
    lds_k_row_pad: int = field(default=_DEFAULT_LDS_K_ROW_PAD, kw_only=True)
    lazy_rescale_threshold: float = field(
        default=_DEFAULT_LAZY_RESCALE_THRESHOLD, kw_only=True
    )
    use_exp2_fast: bool = field(default=True, kw_only=True)
    # Softmax exp2 ops emitted per PV MFMA step. None: ceil(N_SUB*16 / PV steps)
    # on the grid body, 1 on the persistent body (keeps it within 256 VGPR).
    exp_per_pv_step: Optional[int] = field(default=None, kw_only=True)
    # Loop-top wait keeps the newest V DMA in flight (the count stays derived:
    # any other value reads stale LDS). False drains to vmcnt(0).
    partial_vmcnt_prefetch: bool = field(default=True, kw_only=True)
    # s_setprio level around the PV cluster; 0 emits no priority change.
    pv_priority: int = field(default=1, kw_only=True)
    # Depth-1 sched_barrier before PV. None: on unless wide_lds_dma (IGLP owns it).
    pv_sched_fence: Optional[bool] = field(default=None, kw_only=True)
    pv_sched_fence_mask: int = field(default=0, kw_only=True)
    # Per-PV-step sched_group_barrier template. None: on for the persistent
    # body without wide_lds_dma, off for the grid body and wide_lds_dma.
    # MFMA/VALU/TRANS populations follow what each step emits.
    pv_sched_group_template: Optional[bool] = field(default=None, kw_only=True)
    pv_sched_group_ds_read: int = field(default=2, kw_only=True)
    # llvm.amdgcn.iglp.opt at the top of the KV loop body. None: 1 with
    # wide_lds_dma, else off. -1 is off; exclusive with the fence and template.
    iglp_mode: Optional[int] = field(default=None, kw_only=True)
    # PV MFMA traversal: "d_major" (output tile outer) or "k_major" (key step
    # outer). Each output tile accumulates in the same order either way.
    # None: k_major with wide_lds_dma (zero spill), else d_major.
    pv_loop_order: Optional[str] = field(default=None, kw_only=True)
    # Causal only: unmasked body over below-diagonal KV tiles plus a masked
    # diagonal tail, versus masking every tile. Sliding window keeps its phases.
    causal_diag_split: bool = field(default=True, kw_only=True)
    # Output elements per global store; alignment follows the width. bf16 only
    # away from 4: narrower fp16 stores change which f32->f16 conversion the
    # backend selects for some elements, moving them by one ulp, so fp16 is
    # fixed at 4. 8 pairs the two half-waves with v_permlane32_swap so each lane
    # writes one 16-byte store.
    o_store_width: int = field(default=4, kw_only=True)
    # Grid body only (not read by the persistent body, varlen, or paged): map
    # the launch grid's linear workgroup id to work items.
    # "default": block (x, y, z) = (query block, query head, batch).
    # "lpt": query heads fastest, then batch, then query block; causal visits
    # the query blocks from the last (most KV tiles) to the first, so the
    # longest work items are dispatched first. The grid shape is unchanged.
    grid_order: str = field(default="lpt", kw_only=True)
    # Keep the softmax row sum as a per-half-wave partial across the KV loop
    # and combine the two halves once before the epilogue, instead of a
    # cross-half exchange per KV tile.
    defer_row_sum: bool = field(default=True, kw_only=True)
    # Grid body only: one KV tile of software skew inside each wave. QK(j+1)
    # runs while the exp2 of tile j is computed, and PV(j) runs while the row
    # max of tile j+1 is computed, so both MFMA runs have independent VALU
    # work. One barrier per KV tile; every K/V refill gets a whole iteration.
    qk_lookahead: bool = field(default=False, kw_only=True)
    # Combine the two 32-lane halves of the row max (and, under qk_lookahead,
    # the row sum) with v_permlane32_swap_b32 instead of a ds_bpermute shuffle.
    permlane32_reduce: bool = field(default=False, kw_only=True)
    # Grid body only: tag LDS-DMA writes and LDS reads with disjoint alias
    # scopes so the backend's waitcnt pass no longer drains every in-flight
    # K/V DMA (vmcnt(0)) before the first LDS read after it. The body's own
    # s_waitcnt + s_barrier already order each DMA before its reads.
    lds_dma_manual_wait: bool = field(default=False, kw_only=True)
    # qk_lookahead only: split the waves into two halves that run one barrier
    # apart (two barriers per KV tile), so while one half is in its QK phase
    # the other is in its PV phase; the lagging half runs at s_setprio 1.
    wave_group_pingpong: bool = field(default=False, kw_only=True)
    # qk_lookahead only: KV tiles per loop trip (1 or 2). Two alternates the
    # carried score registers instead of copying them at every back edge.
    qk_lookahead_unroll: int = field(default=1, kw_only=True)
    # qk_lookahead only: bit 1 reads every K operand of a tile from LDS before
    # its first QK MFMA, bit 2 every V operand before the first PV MFMA, each
    # behind a scheduling fence, so the MFMA run carries no LDS waits. Costs
    # the operand registers of a whole tile.
    bulk_operand_reads: int = field(default=0, kw_only=True)
    # wide_lds_dma only: address every K/V operand read of a tile as one
    # per-lane base (lane offset + buffer offset, formed once per tile) plus a
    # compile-time immediate, so the ds_read offset field absorbs the constant
    # part instead of the backend keeping one address register per read.
    lds_read_base: bool = field(default=False, kw_only=True)
    # Grid body only: lift the wave index and qk_scale into SGPRs with
    # readfirstlane, so wave-uniform LDS-DMA destinations and the exp2 scale
    # do not each hold a vector register across the KV loop.
    scalar_uniforms: bool = field(default=False, kw_only=True)
    # qk_lookahead + lds_read_base only: issue each K (QK) and V (PV) operand
    # read this many MFMAs before its use (0 = leave placement to the backend
    # scheduler). Holds that many operands in registers; a fence per MFMA
    # keeps the distance.
    operand_read_ahead: int = field(default=0, kw_only=True)
    # wave_group_pingpong + lds_read_base only: barrier windows per KV tile
    # that separate operand reads (all K, then all V operands of a tile) from
    # MFMA runs (QK, then PV). The lagging wave group runs one window behind,
    # so on each SIMD one wave reads operands while the other issues MFMAs.
    # 4: K reads | QK | V reads | PV (the only mode; K(j+3) refills the buffer
    # the lagging group reads in window 1, so the K reads cannot share the QK
    # window). K is prefetched two tiles ahead. 0 = off.
    mfma_windows: int = field(default=0, kw_only=True)
    # mfma_windows only: 0 = the lagging group runs at s_setprio 1; 1 = every
    # wave raises its priority for its MFMA windows; 2 = for its operand
    # windows; 3 = no s_setprio.
    window_priority: int = field(default=0, kw_only=True)
    # mfma_windows only: P packs (of block_n // 16, the last ones in PV order)
    # whose exp2 runs in the PV MFMA window instead of the QK window. The QK
    # window otherwise carries the whole tile's exp2 work and outlasts the
    # other group's operand window; this moves part of it next to the PV
    # MFMAs that do not consume those packs. Needs pv_loop_order k_major.
    pv_window_exp: int = field(default=0, kw_only=True)

    def resolved_exp_per_pv_step(self) -> int:
        if self.exp_per_pv_step is not None:
            return int(self.exp_per_pv_step)
        if self.persistent:
            return 1
        n_sub = self.block_n // 32
        pv_steps = (self.head_size // 32) * (self.block_n // 16)
        return -(-(n_sub * 16) // pv_steps)

    def resolved_pv_sched_fence(self) -> bool:
        if self.pv_sched_fence is None:
            return not self.wide_lds_dma
        return bool(self.pv_sched_fence)

    def resolved_pv_sched_group_template(self) -> bool:
        if self.pv_sched_group_template is None:
            return self.persistent and not self.wide_lds_dma
        return bool(self.pv_sched_group_template)

    def resolved_grid_order(self) -> str:
        """The order the grid body emits: varlen and paged keep the block-id map."""
        if self.persistent or self.varlen or self.paged:
            return "default"
        return self.grid_order

    def resolved_iglp_mode(self) -> int:
        if self.iglp_mode is None:
            return 1 if self.wide_lds_dma else _IGLP_OFF
        return int(self.iglp_mode)

    def resolved_pv_loop_order(self) -> str:
        if self.pv_loop_order is None:
            return "k_major" if self.wide_lds_dma else "d_major"
        return self.pv_loop_order

    def supported_persist_decodes(self) -> frozenset[str]:
        return super().supported_persist_decodes() | {
            "gqa_pair",
            "gqa_pair_2phase",
        }

    def __post_init__(self) -> None:
        super().__post_init__()
        if type(self.emit_lse) is not bool:
            raise ValueError(
                f"emit_lse must be a bool, got {type(self.emit_lse).__name__}"
            )
        if self.lds_v_row_pad < 0 or self.lds_v_row_pad % 8 != 0:
            raise ValueError(
                "lds_v_row_pad must be a non-negative multiple of 8 bf16 "
                f"elements (16 bytes), got {self.lds_v_row_pad}"
            )
        self._validate_codegen_knobs()
        if self.causal_bottom_right:
            # The non-persistent contiguous builder is the only gfx950 path
            # that implements the compile-time shifted diagonal.
            if self.paged:
                raise ValueError("causal_bottom_right is not supported with paged=True")
            if self.persistent:
                raise ValueError(
                    "causal_bottom_right is not supported with persistent=True"
                )
            if self.sliding_window > 0:
                raise ValueError(
                    "causal_bottom_right is not supported with sliding_window>0"
                )
            if self.varlen:
                raise ValueError(
                    "causal_bottom_right is not supported with varlen=True"
                )
        if self.wide_lds_dma:
            if self.head_size != 128 or self.block_n != 64:
                raise ValueError("wide_lds_dma requires head_size=128 and block_n=64")
            if self.ragged or self.varlen or self.paged:
                raise ValueError(
                    "wide_lds_dma is validated only for aligned contiguous K/V"
                )
            if (
                self.lds_k_group_pad != 8
                or self.lds_k_row_pad != _DEFAULT_LDS_K_ROW_PAD
                or self.lds_v_row_pad != _DEFAULT_GFX950_LAYOUT["lds_v_row_pad"]
            ):
                raise ValueError(
                    "wide_lds_dma requires K/V slab padding of 8/32 elements"
                )

        if self.persist_decode == "gqa_pair":
            gqa = self.num_queries_per_kv
            nqb = (self.seqlen_q + self.block_m - 1) // self.block_m
            expected_np = nqb * self.num_kv_heads * self.batch
            if not self.persistent or not self.causal:
                raise ValueError("gqa_pair requires persistent causal attention")
            if self.ragged or self.varlen or self.paged:
                raise ValueError(
                    "gqa_pair is validated only for aligned dense attention"
                )
            if nqb % 2 or gqa % 2:
                raise ValueError("gqa_pair requires even NQB and even GQA ratio")
            if self.num_persistent != expected_np:
                raise ValueError(
                    "gqa_pair requires num_persistent == NQB*Hkv*B "
                    f"({expected_np}), got {self.num_persistent}"
                )
        if self.persist_decode == "gqa_pair_2phase":
            gqa = self.num_queries_per_kv
            nqb = (self.seqlen_q + self.block_m - 1) // self.block_m
            expected_np = nqb * self.num_kv_heads * self.batch * gqa // 2
            if not self.persistent or not self.causal:
                raise ValueError("gqa_pair_2phase requires persistent causal attention")
            if self.ragged or self.varlen or self.paged:
                raise ValueError(
                    "gqa_pair_2phase is validated only for aligned dense attention"
                )
            if nqb % 2 or gqa < 2:
                raise ValueError("gqa_pair_2phase requires even NQB and GQA ratio >= 2")
            if self.num_persistent != expected_np:
                raise ValueError(
                    "gqa_pair_2phase requires num_persistent == W/2 "
                    f"({expected_np}), got {self.num_persistent}"
                )

    def _validate_codegen_knobs(self) -> None:
        if self.lds_k_row_pad < 0 or self.lds_k_row_pad % 8 != 0:
            raise ValueError(
                "lds_k_row_pad must be a non-negative multiple of 8 elements "
                f"(16 bytes, keeps ds_read_b128 aligned), got {self.lds_k_row_pad}"
            )
        if self.lds_num_buffers != _DEFAULT_LDS_NUM_BUFFERS:
            raise ValueError(
                "lds_num_buffers: only 2 is implemented (the prologue primes two "
                f"tiles; 3 spills), got {self.lds_num_buffers}"
            )
        if (
            not 0.0
            < float(self.lazy_rescale_threshold)
            <= (_DEFAULT_LAZY_RESCALE_THRESHOLD)
        ):
            raise ValueError(
                "lazy_rescale_threshold must be in (0, 8] log2 units (8 bounds P "
                f"at 256), got {self.lazy_rescale_threshold}"
            )
        if self.exp_per_pv_step is not None and int(self.exp_per_pv_step) < 1:
            raise ValueError(
                f"exp_per_pv_step must be >= 1 or None, got {self.exp_per_pv_step}"
            )
        if not 0 <= int(self.pv_priority) <= 3:
            raise ValueError(f"pv_priority must be in 0..3, got {self.pv_priority}")
        if not 0 <= int(self.pv_sched_fence_mask) <= _SCHED_BARRIER_MASK_MAX:
            raise ValueError(
                f"pv_sched_fence_mask must be in 0..{_SCHED_BARRIER_MASK_MAX:#x}, "
                f"got {self.pv_sched_fence_mask}"
            )
        if int(self.pv_sched_group_ds_read) < 1:
            raise ValueError(
                "pv_sched_group_ds_read must be >= 1, got "
                f"{self.pv_sched_group_ds_read}"
            )
        if self.iglp_mode not in (None, _IGLP_OFF, 0, 1):
            raise ValueError(
                f"iglp_mode must be None, -1 (off), 0 or 1, got {self.iglp_mode}"
            )
        if self.pv_loop_order is not None and self.pv_loop_order not in (
            _PV_LOOP_ORDERS
        ):
            raise ValueError(
                f"pv_loop_order must be one of {sorted(_PV_LOOP_ORDERS)} or None, "
                f"got {self.pv_loop_order!r}"
            )
        if self.o_store_width not in (1, 2, 4, 8):
            raise ValueError(
                f"o_store_width must be 1, 2, 4 or 8, got {self.o_store_width}"
            )
        if self.o_store_width != 4 and self.dtype == "fp16":
            raise ValueError(
                f"o_store_width={self.o_store_width} is bf16-only: narrower fp16 "
                "stores are not bit-identical to the width-4 output"
            )
        if self.resolved_iglp_mode() != _IGLP_OFF and (
            self.resolved_pv_sched_fence() or self.resolved_pv_sched_group_template()
        ):
            raise ValueError(
                "iglp_mode owns loop scheduling; disable pv_sched_fence and "
                "pv_sched_group_template when enabling it"
            )
        if self.grid_order not in ("default", "lpt"):
            raise ValueError(
                f"grid_order must be 'default' or 'lpt', got {self.grid_order!r}"
            )
        if self.qk_lookahead:
            if self.persistent:
                raise ValueError("qk_lookahead is implemented on the grid body only")
            if self.sliding_window > 0 or self.paged or self.ragged:
                raise ValueError(
                    "qk_lookahead does not support sliding_window, paged or ragged"
                )
            if self.resolved_pv_sched_group_template():
                raise ValueError(
                    "qk_lookahead owns the softmax schedule; disable "
                    "pv_sched_group_template"
                )
        if self.wave_group_pingpong and (not self.qk_lookahead or self.num_waves % 2):
            raise ValueError(
                "wave_group_pingpong requires qk_lookahead and an even wave count"
            )
        if self.qk_lookahead_unroll not in (1, 2):
            raise ValueError(
                f"qk_lookahead_unroll must be 1 or 2, got {self.qk_lookahead_unroll}"
            )
        if self.bulk_operand_reads not in (0, 1, 2, 3):
            raise ValueError(
                f"bulk_operand_reads must be 0..3, got {self.bulk_operand_reads}"
            )
        if self.bulk_operand_reads and (
            not self.qk_lookahead or (self.head_size != 128 and not self.wide_lds_dma)
        ):
            raise ValueError(
                "bulk_operand_reads requires qk_lookahead (D128 or wide_lds_dma)"
            )
        if self.lds_read_base and (not self.wide_lds_dma or self.persistent):
            raise ValueError("lds_read_base requires wide_lds_dma on the grid body")
        if self.scalar_uniforms and self.persistent:
            raise ValueError("scalar_uniforms is implemented on the grid body only")
        if not 0 <= self.operand_read_ahead <= 16 or (
            self.operand_read_ahead
            and (
                not (self.qk_lookahead and self.lds_read_base)
                or self.bulk_operand_reads
            )
        ):
            raise ValueError(
                "operand_read_ahead must be 0..16, needs qk_lookahead and "
                "lds_read_base, and excludes bulk_operand_reads; got "
                f"{self.operand_read_ahead}"
            )
        if self.mfma_windows not in (0, 4) or self.window_priority not in (
            0,
            1,
            2,
            3,
        ):
            raise ValueError(
                "mfma_windows must be 0 or 4 and window_priority 0..3, got "
                f"{self.mfma_windows}, {self.window_priority}"
            )
        if self.window_priority and not self.mfma_windows:
            raise ValueError("window_priority needs mfma_windows")
        if self.mfma_windows and (
            not (
                self.wave_group_pingpong
                and self.lds_read_base
                and self.lds_dma_manual_wait
            )
            or self.bulk_operand_reads
            or self.operand_read_ahead
        ):
            raise ValueError(
                "mfma_windows needs wave_group_pingpong, lds_read_base and "
                "lds_dma_manual_wait, and excludes bulk_operand_reads and "
                "operand_read_ahead"
            )
        if self.pv_window_exp and (
            not self.mfma_windows
            or not 0 < self.pv_window_exp < self.block_n // 16
            or self.resolved_pv_loop_order() != "k_major"
        ):
            raise ValueError(
                "pv_window_exp needs mfma_windows and k_major PV order, and must "
                f"be below block_n // 16; got {self.pv_window_exp}"
            )

    @property
    def resolved_persist_decode(self) -> str:
        if self.persist_decode != "auto":
            return self.persist_decode
        gqa = self.num_queries_per_kv
        nqb = (self.seqlen_q + self.block_m - 1) // self.block_m
        aligned_causal = (
            self.persistent
            and self.causal
            and not self.ragged
            and not self.varlen
            and not self.paged
            and self.sliding_window == 0
        )
        if aligned_causal and nqb % 2 == 0 and gqa % 2 == 0:
            pair_np = nqb * self.num_kv_heads * self.batch
            if self.num_persistent == pair_np:
                return "gqa_pair"
        if aligned_causal and nqb % 2 == 0 and gqa >= 2:
            two_phase_np = nqb * self.num_kv_heads * self.batch * gqa // 2
            if self.num_persistent == two_phase_np:
                return "gqa_pair_2phase"
        return super().resolved_persist_decode

    @property
    def runtime_shape(self) -> bool:
        """Whether the body reads the problem shape *only* from its kernel params,
        baking it nowhere -- so ONE compiled kernel serves every shape and the three
        fields drop out of the cache key. This is what collapses the AOT
        batch x seqlen instance explosion.

        Not the same as having the params: every non-persistent body takes them (see
        ``_has_shape_params``). The sub-modes excluded here take them too, and use
        them for the q/k/o base offsets, but still bake seqlen somewhere else -- the
        paged K/V bound, the ragged k-tail mask and OOB store predicate, the
        non-runtime k-tile trip count -- so they must keep per-shape identity.
        A moving bottom-right diagonal also bakes the sequence-length difference;
        equal-length bottom-right stays in the unshifted runtime-shape cohort.
        ``persistent`` is excluded for a stronger reason: it is a separate body that
        declares no shape params at all."""
        return not (
            self.persistent
            or self.ragged
            or self.varlen
            or self.paged
            or self.sliding_window > 0
            or (self.causal_bottom_right and self.seqlen_q != self.seqlen_kv)
        )

    @property
    def runtime_param_fields(self) -> tuple[str, ...]:
        """The shape fields this body reads *only* from its kernel params, and so
        the fields ``attention_dense_cache_key`` may drop.

        Gated on ``runtime_shape``, not on ``_has_shape_params``: taking a field as
        a kernarg is necessary but not sufficient: a sub-mode that also bakes the
        field elsewhere would be a cache collision if it declared it here.

        Kept in sync with the body by hand. A future per-batch specialization would
        narrow this (bake ``batch``, keep the seqlens runtime) -- the cache key
        follows automatically, but the symbol name would have to derive from this
        tuple first. It does not today.
        """
        return ("batch", "seqlen_q", "seqlen_kv") if self.runtime_shape else ()

    def _layout_name_parts(self) -> tuple[str, ...]:
        if self.head_size != 128:
            return ()
        parts = []
        # The base names the D<128 group pad; the D128 row pads are tagged only
        # when they differ from the shipped values, so default symbols are stable.
        if self.lds_k_row_pad != _DEFAULT_LDS_K_ROW_PAD:
            parts.append(f"krpad{self.lds_k_row_pad}")
        if self.lds_v_row_pad != _DEFAULT_GFX950_LAYOUT["lds_v_row_pad"]:
            parts.append(f"vpad{self.lds_v_row_pad}")
        return tuple(parts)

    def _shape_name_parts(self) -> tuple[str, ...]:
        return () if self.runtime_shape else super()._shape_name_parts()

    def _algorithm_name_parts(self) -> tuple[str, ...]:
        parts = list(super()._algorithm_name_parts())
        if self.wide_lds_dma:
            parts.append("wdma")
        # Preserve shipped symbols at the default while keeping explicitly
        # swept WPE binaries distinct for AOT packaging and name-based tools.
        if self.waves_per_eu != 2:
            parts.append(f"wpe{self.waves_per_eu}")
        codegen = self._codegen_name_part()
        if codegen:
            parts.append(codegen)
        if self.emit_lse:
            parts.append("lse")
        return tuple(parts)

    def _codegen_name_part(self) -> str:
        """One token for every codegen knob set away from its default, so a
        name-keyed AOT catalog cannot serve a default binary for a tuned spec."""
        changed = sorted(
            (name, getattr(self, name))
            for name, default in _CODEGEN_KNOB_DEFAULTS.items()
            if getattr(self, name) != default
        )
        if not changed:
            return ""
        return "cg" + hashlib.sha256(repr(changed).encode()).hexdigest()[:8]

    def _persist_decode_name_part(self) -> str:
        return {
            "hkv_major": "hkvmaj",
            "gqa_pair": "gqapair",
            "gqa_pair_2phase": "gqapair2",
        }.get(self.resolved_persist_decode, "")


_CODEGEN_KNOB_DEFAULTS = MappingProxyType(
    {
        f.name: f.default
        for f in _dataclass_fields(Gfx950AttentionDenseSpec)
        if f.name
        in (
            "lds_num_buffers",
            "lazy_rescale_threshold",
            "use_exp2_fast",
            "exp_per_pv_step",
            "partial_vmcnt_prefetch",
            "pv_priority",
            "pv_sched_fence",
            "pv_sched_fence_mask",
            "pv_sched_group_template",
            "pv_sched_group_ds_read",
            "iglp_mode",
            "pv_loop_order",
            "causal_diag_split",
            "o_store_width",
            "grid_order",
            "defer_row_sum",
            "qk_lookahead",
            "permlane32_reduce",
            "lds_dma_manual_wait",
            "wave_group_pingpong",
            "qk_lookahead_unroll",
            "bulk_operand_reads",
            "lds_read_base",
            "scalar_uniforms",
            "operand_read_ahead",
            "mfma_windows",
            "window_priority",
            "pv_window_exp",
        )
    }
)

# Compatibility name for existing gfx950 callers. Cross-architecture code must
# import the neutral base from kernels.common.attention_dense_spec instead.
AttentionDenseSpec = Gfx950AttentionDenseSpec


def supports_attention_dense(
    spec: AttentionDenseSpec, *, arch: str = "gfx950"
) -> Tuple[bool, str]:
    """Return (ok, reason). The kernel is gfx950-only and dense (no paging/bias)."""
    if arch != "gfx950":
        return False, f"attention_dense is gfx950-only (got {arch})"
    if not isinstance(spec, Gfx950AttentionDenseSpec):
        return False, (
            "gfx950 attention_dense requires Gfx950AttentionDenseSpec, got "
            f"{type(spec).__name__}"
        )
    # Shared preflight: dataclass re-validation (which reconstructs THIS subclass, so
    # the gfx950-private knob validators run too), positive extents, block_n dividing
    # the query tile, and the 32-bit extent bounds. All four are base-spec properties
    # with the same verdict for every dense body, so they live in kernels.common
    # rather than being replicated here and in gfx942. The 32-bit check is what the
    # runtime-shape path leans on: the extent is a device-side mul of two kernargs
    # there, and batch/seqlen no longer split the launcher-cache key, so this is the
    # only place an oversized launch can be caught -- and it runs per launch, since
    # run_attention_dense calls supports() before the cache lookup.
    ok, why = check_dense_spec_preflight(spec)
    if not ok:
        return False, why
    supported_block_m = {
        int(geometry["block_m"]) for geometry in DENSE_TILE_GEOMETRIES.values()
    }
    if spec.block_m not in supported_block_m:
        return False, (
            f"gfx950 block_m must be one of {sorted(supported_block_m)}, "
            f"got {spec.block_m}"
        )
    if spec.block_n % spec.num_waves != 0:
        return False, (
            f"block_n={spec.block_n} must be divisible by num_waves="
            f"{spec.num_waves} so K/V DMA rows distribute evenly"
        )
    return True, ""


def _pv_steps(order: str, d_tiles: int, kk_steps: int) -> Tuple[Tuple[int, int], ...]:
    """PV MFMA ``(dt, kk_step)`` sequence for ``pv_loop_order``."""
    if order == "k_major":
        return tuple((dt, kk) for kk in range(kk_steps) for dt in range(d_tiles))
    return tuple((dt, kk) for dt in range(d_tiles) for kk in range(kk_steps))


def _dense_lse_output(
    b, dst, index, m, l, row_exists, row_has_key, lane_half, *, sinks, head, dtype
):
    """Emit stats from one half-wave while both halves retain their O stores."""
    owner = b.land(row_exists, b.cmp_eq(lane_half, b.const_i32(0)))
    with b.scf_if(owner):
        with b.scf_if(row_has_key):
            lse_store(b, dst, index, m, l, row_has_mass=row_has_key)
        with b.scf_if(b.lnot(row_has_key)):
            if sinks is not None:
                sink = b.cast_to_f32(b.global_load(sinks, head, dtype, align=2))
                sink_m = b.fmul(sink, b.const_f32(LOG2E))
                mass = b.cmp_eq(b.const_i32(1), b.const_i32(1))
                lse_store(b, dst, index, sink_m, b.const_f32(1.0), row_has_mass=mass)
            else:
                lse_store(
                    b,
                    dst,
                    index,
                    b.const_f32(0.0),
                    b.const_f32(1.0),
                    row_has_mass=row_has_key,
                )


def build_attention_dense(
    spec: AttentionDenseSpec, *, arch: str = "gfx950"
) -> KernelDef:
    """Emit the dense flash-attention prefill kernel described by ``spec``."""
    if arch != "gfx950":
        raise NotImplementedError(f"attention_dense is gfx950-only (got {arch})")
    ok, reason = supports_attention_dense(spec, arch=arch)
    if not ok:
        raise ValueError(f"unsupported gfx950 attention_dense spec: {reason}")

    if spec.persistent:
        return _build_attention_dense_persistent(spec)

    B = spec.batch
    Sq = spec.seqlen_q
    Skv = spec.seqlen_kv
    Hq = spec.num_query_heads
    Hkv = spec.num_kv_heads
    D = spec.head_size
    causal = spec.causal
    dtype = spec.dtype_ir

    BLOCK_M = spec.block_m
    WAVES = spec.num_waves
    BN = spec.block_n
    NBUF = spec.lds_num_buffers
    PAD = spec.lds_k_row_pad
    W = spec.sliding_window
    Wt = W // BN  # window length in KV tiles (0 when disabled)
    # Compile-time bottom-right diagonal shift. The persistent builder returns
    # above and deliberately remains unchanged.
    DIAG_OFF = (Skv - Sq) if spec.causal_bottom_right else 0
    DIAG_TILES = DIAG_OFF // BN
    varlen = spec.varlen
    RAGGED = spec.ragged
    LAZY_RESCALE = spec.lazy_rescale
    LAZY_THRESHOLD = float(spec.lazy_rescale_threshold)
    use_sinks = spec.use_sinks

    K_STEPS = D // 16
    D_TILES = D // 32
    N_SUB = BN // 32
    KK_STEPS = BN // 16
    PV_STEPS = _pv_steps(spec.resolved_pv_loop_order(), D_TILES, KK_STEPS)
    EXP_PER = spec.resolved_exp_per_pv_step()
    SCHED_GROUP = spec.resolved_pv_sched_group_template()
    SCHED_GROUP_DS_READ = int(spec.pv_sched_group_ds_read)
    PV_FENCE = spec.resolved_pv_sched_fence()
    PV_FENCE_MASK = int(spec.pv_sched_fence_mask)
    PV_PRIORITY = int(spec.pv_priority)
    PINGPONG = spec.wave_group_pingpong
    UNROLL = int(spec.qk_lookahead_unroll)
    BULK_K = bool(spec.bulk_operand_reads & 1)
    BULK_V = bool(spec.bulk_operand_reads & 2)
    LDS_READ_BASE = spec.lds_read_base
    READ_AHEAD = int(spec.operand_read_ahead)
    WINDOWS = int(spec.mfma_windows)
    WIN_PRIO = int(spec.window_priority)
    PV_EXP = int(spec.pv_window_exp)
    IGLP = spec.resolved_iglp_mode()
    gqa = Hq // Hkv
    stride_q_tok = Hq * D
    stride_k_tok = Hkv * D
    # DMA row packing: one async_buffer_load_lds instr moves 64 lanes x 2 bf16 =
    # 128 elems. D==128 => 1 row/instr (the padded fast path, byte-identical).
    # D<128 => pack 128//D rows/instr into one contiguous LDS group; K places
    # those groups at a padded pitch, V at the plain packed pitch. Reads get the
    # correct pitch from their LDS tensor shape either way.
    ROWS_PER_INSTR = 128 // D

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = WAVES * 64
    b.kernel.attrs["waves_per_eu"] = int(spec.waves_per_eu)
    if spec.lds_dma_manual_wait:
        b.kernel.attrs["lds_dma_alias_scopes"] = True

    q = b.param(
        "q_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
    )
    k = b.param(
        "k_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
    )
    v = b.param(
        "v_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
    )
    o = b.param(
        "o_ptr", PtrType(dtype, "global"), noalias=True, writeonly=True, align=16
    )
    scale = b.param("scale", F32)
    # Problem shape as kernel params. Unconditional here: every spec reaching this
    # body takes them, so the base offsets below read params with no baked
    # alternative. Declared right after scale to fix their ABI position (mirrored in
    # attention_dense_signature, gated there by _has_shape_params because that
    # function also serves the persistent body, which returned above and declares
    # none of these).
    #
    # Taking the shape as params is an ABI fact and is NOT the same as
    # ``spec.runtime_shape``, which is the narrower claim that the body bakes the
    # shape *nowhere* and so can drop it from the cache key. Sub-modes still bake it
    # elsewhere -- the paged K/V bound below, the ragged k-tail mask, the non-runtime
    # k-tile trip count -- and keep per-shape identity via runtime_param_fields.
    batch_p = b.param("batch", I32)
    seqlen_q_p = b.param("seqlen_q", I32)
    seqlen_kv_p = b.param("seqlen_kv", I32)
    if use_sinks:
        sinks = b.param(
            "sink_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
        )
    if varlen:
        cu_q = b.param(
            "cu_seqlens_q", PtrType(I32, "global"), noalias=True, readonly=True, align=4
        )
        cu_kv = b.param(
            "cu_seqlens_kv",
            PtrType(I32, "global"),
            noalias=True,
            readonly=True,
            align=4,
        )
    block_tables = kv_lens = bt_stride = None
    if spec.paged:
        block_tables = b.param(
            "block_tables", PtrType(I32, "global"), noalias=True, readonly=True, align=4
        )
        kv_lens = b.param(
            "kv_lens", PtrType(I32, "global"), noalias=True, readonly=True, align=4
        )
        bt_stride = b.param("block_table_stride", I32)
    if spec.emit_lse:
        lse = b.param(
            "lse_ptr",
            PtrType(F32, "global"),
            noalias=True,
            writeonly=True,
            align=4,
        )
    qk_scale = b.fmul(scale, b.const_f32(LOG2E))
    if spec.scalar_uniforms:
        # The product of two uniforms is computed on the VALU (no scalar float
        # ALU), so lift it back to an SGPR instead of holding a VGPR.
        qk_scale = b.bitcast(b.readfirstlane(b.bitcast(qk_scale, I32)), F32)

    # native v_exp_f32 (softmax arg bounded; see _MAX_SCALE) or the guarded exp2
    _exp2 = b.exp2_fast if spec.use_exp2_fast else b.exp2

    tid = b.thread_id_x()
    wave = b.div(tid, b.const_i32(64))
    if spec.scalar_uniforms:
        # Wave-uniform: every per-wave LDS-DMA destination then stays scalar.
        wave = b.readfirstlane(wave)
    lane = b.mod(tid, b.const_i32(64))
    lane_m = b.mod(lane, b.const_i32(32))
    lane_h = b.div(lane, b.const_i32(32))
    d_base = b.mul(lane_h, b.const_i32(8))
    # Mask sentinel, written into RAW (unscaled) scores; it also seeds the running
    # max m (log2 units) when there are no sinks. It is a power of two so
    # sentinel * qk_scale is exact. A row fully masked in its first visited tile
    # then gets, exactly: P = exp2(0) = 1 when qk_scale <= 1 (m = sentinel *
    # qk_scale), or P = exp2(sentinel * (qk_scale - 1)) = 0 when qk_scale > 1
    # (m stays at the sentinel). Either way the next tile with a real key has
    # alpha = exp2(m_old - m_new) = 0, which clears it. A non-power-of-two
    # sentinel would instead leave a huge rounding residue whose sign depends on
    # the scale: exp2 of it is 0 when negative and inf (then NaN) when positive.
    neg_inf = b.const_f32(-(2.0**99))
    if use_sinks:
        rcp_ln2 = b.const_f32(LOG2E)
        one_f = b.const_f32(1.0)

    if spec.resolved_grid_order() == "lpt":
        # Same grid (nqb, Hq, B); re-rank the linear workgroup id so query
        # heads vary fastest and, under causal, the longest query blocks go
        # first (the hardware dispatches workgroups in linear-id order).
        nqb_v = b.div(b.add(seqlen_q_p, b.const_i32(BLOCK_M - 1)), b.const_i32(BLOCK_M))
        lin = b.add(
            b.block_id_x(),
            b.mul(
                nqb_v,
                b.add(b.block_id_y(), b.mul(b.block_id_z(), b.const_i32(Hq))),
            ),
        )
        hq = b.mod(lin, b.const_i32(Hq))
        lin_r = b.div(lin, b.const_i32(Hq))
        bt = b.mod(lin_r, batch_p)
        qb_rank = b.div(lin_r, batch_p)
        qb = b.sub(b.sub(nqb_v, b.const_i32(1)), qb_rank) if causal else qb_rank
    else:
        qb = b.block_id_x()
        hq = b.block_id_y()
        bt = b.block_id_z()
    hkv = b.div(hq, b.const_i32(gqa))
    q_tok0 = b.add(b.mul(qb, b.const_i32(BLOCK_M)), b.mul(wave, b.const_i32(32)))

    if varlen:
        # Per-sequence token bases from cu_seqlens; early-exit q-blocks that fall
        # past this sequence's length (packed [total_tok, H, D] layout).
        q_seq0 = b.global_load_i32(cu_q, bt)
        q_seq1 = b.global_load_i32(cu_q, b.add(bt, b.const_i32(1)))
        seqlen_q_b = b.sub(q_seq1, q_seq0)
        kv_seq0 = b.global_load_i32(cu_kv, bt)
        kv_seq1 = b.global_load_i32(cu_kv, b.add(bt, b.const_i32(1)))
        seqlen_kv_b = b.sub(kv_seq1, kv_seq0)
        with b.scf_if(b.cmp_ge(b.mul(qb, b.const_i32(BLOCK_M)), seqlen_q_b)):
            b.ret()
        q_base = b.add(
            b.mul(q_seq0, b.const_i32(stride_q_tok)), b.mul(hq, b.const_i32(D))
        )
        k_base = b.add(
            b.mul(kv_seq0, b.const_i32(stride_k_tok)), b.mul(hkv, b.const_i32(D))
        )
    else:
        q_base = b.add(
            b.mul(b.mul(bt, seqlen_q_p), b.const_i32(stride_q_tok)),
            b.mul(hq, b.const_i32(D)),
        )
        k_base = b.add(
            b.mul(b.mul(bt, seqlen_kv_p), b.const_i32(stride_k_tok)),
            b.mul(hkv, b.const_i32(D)),
        )

    # --- LDS allocation: PAD on K (bank-conflict fix), +PAD_V on V (transposed
    #     PV read bank-conflict pad). A per-ROW pad requires 1 row/instr (a
    #     padded row is not contiguous with the next), so the packed D<128 K
    #     loader pads between DMA row-GROUPS instead: the group stays contiguous
    #     for the multi-row DMA while the group pitch still breaks the QK read's
    #     bank pattern (see the ``lds_k_group_pad`` spec field). V keeps the
    #     unpadded packed pitch. ---
    if ROWS_PER_INSTR == 1:
        K_GROUP = 1
        K_ROWS_LDS = BN
        LDROW = D + PAD
    else:
        K_GROUP = ROWS_PER_INSTR
        K_ROWS_LDS = BN // K_GROUP
        LDROW = K_GROUP * D + spec.lds_k_group_pad
    VROW = D + spec.lds_v_row_pad if ROWS_PER_INSTR == 1 else D
    WIDE_DMA = spec.wide_lds_dma
    if WIDE_DMA:
        # Same slab layout as the persistent wide path: 8 rows x 64 columns per
        # padded line, two dwordx4 DMA instructions per operand per line pass.
        K_D_RPT = D // 64
        K_N_RPT = BN // 8
        K_LINE_STRIDE = 64 * 8 + PAD
        V_D_RPT = D // 64
        V_N_RPT = BN // 8
        V_LINE_STRIDE = 64 * 8 + spec.lds_v_row_pad
        WIDE_LINE_PASSES = 8 // WAVES
        assert 8 % WAVES == 0
        # mfma_windows: one K or V tile is this many DMA instructions per wave;
        # each window wait leaves the two newest tiles in flight.
        WIN_VMCNT = 2 * WIDE_LINE_PASSES * (D // 64)
        K_lds = b.smem_alloc(
            dtype, [NBUF, K_D_RPT, K_N_RPT, K_LINE_STRIDE], name_hint="Klds"
        )
        V_lds = b.smem_alloc(
            dtype, [NBUF, V_D_RPT, V_N_RPT, V_LINE_STRIDE], name_hint="Vlds"
        )
    else:
        K_lds = b.smem_alloc(dtype, [NBUF, K_ROWS_LDS, LDROW], name_hint="Klds")
        V_lds = b.smem_alloc(dtype, [NBUF, BN, VROW], name_hint="Vlds")
    # Packed-K read decode, hoisted out of the KV loop: krow = nsub*32 + lane_m
    # with nsub*32 even, so the group/sub-row split is a shift+and on lane_m
    # plus a compile-time nsub scale.
    if K_GROUP > 1:
        k_lane_grp = b.div(lane_m, b.const_i32(K_GROUP))
        k_sub_col = b.mul(b.mod(lane_m, b.const_i32(K_GROUP)), b.const_i32(D))
    else:
        k_lane_grp = None
        k_sub_col = None

    # Q packs (B operand), loaded unscaled. qk_scale = softmax_scale * log2(e) is
    # applied in fp32 after the QK MFMA (softmax_max + the exp2 FMA below):
    # rounding Q * qk_scale back to bf16 costs up to 2^-8 relative error per
    # element, and the score error it causes grows with the scale.
    # ragged: a bounds-checked buffer load returns 0 for OOB query rows (the
    # partial last block), so padded rows are register-zero (their output is
    # dropped by the guarded store). Aligned: direct global load (unchanged IR).
    q_rsrc = b.buffer_rsrc(q, b.const_i32(B * Sq * Hq * D * 2)) if RAGGED else None
    q_tok = b.add(q_tok0, lane_m)
    q_packs = []
    for ks in range(K_STEPS):
        col = b.add(b.const_i32(ks * 16), d_base)
        addr = b.add(b.add(q_base, b.mul(q_tok, b.const_i32(stride_q_tok))), col)
        if RAGGED:
            raw = b.buffer_load_vN(
                q_rsrc, b.mul(addr, b.const_i32(2)), b.const_i32(0), dtype, 8
            )
        else:
            raw = b.global_load_vN(q, addr, dtype, 8, align=16)
        q_packs.append(raw)

    # ragged: ceil so the partial last KV tile is visited (its OOB keys load 0
    # into LDS and are masked out); aligned: exact.
    n_ktiles = ((Skv + BN - 1) // BN) if RAGGED else (Skv // BN)
    n_per = BLOCK_M // BN

    K_BYTES_PER_BUF = K_ROWS_LDS * LDROW * 2
    # Bytes per DMA destination group. ROWS_PER_INSTR==1 => one row, so this is
    # the row pitch and the D=128 emission is unchanged.
    K_GROUP_BYTES = LDROW * 2
    V_BYTES_PER_BUF = BN * VROW * 2
    if WIDE_DMA:
        K_BYTES_PER_BUF = K_D_RPT * K_N_RPT * K_LINE_STRIDE * 2
        V_BYTES_PER_BUF = V_D_RPT * V_N_RPT * V_LINE_STRIDE * 2
    # NOT a padded pitch: the DMA writes ROWS_PER_INSTR*D CONTIGUOUS elements, so
    # a padded V would need its second row at +D, not +D+pad. Padding V requires
    # a pad-aware transposed read, not a wider stride here.
    V_GROUP_BYTES = ROWS_PER_INSTR * VROW * 2
    ROWS_PER_WAVE = BN // WAVES
    # Group indexing uses floor division; keep it exact so a future head size or
    # tile width fails loudly instead of silently mis-addressing LDS.
    assert BN % K_GROUP == 0 and ROWS_PER_WAVE % ROWS_PER_INSTR == 0, (
        f"K row-group split must divide evenly: BN={BN} K_GROUP={K_GROUP} "
        f"ROWS_PER_WAVE={ROWS_PER_WAVE} ROWS_PER_INSTR={ROWS_PER_INSTR}"
    )
    V_DMA_PASSES = (BN * D) // (WAVES * 64 * 8)
    LOOP_VMCNT = V_DMA_PASSES if spec.partial_vmcnt_prefetch else 0
    zero_soff = b.const_i32(0)
    K_lds_addr = b.smem_addr_of(K_lds)
    V_lds_addr = b.smem_addr_of(V_lds)
    # Emit a SEPARATE num_records IR node per buffer_rsrc (NOT one shared node):
    # develop emits two here, so sharing one silently breaks byte-identity (the
    # attention_dense representative-IR golden).
    #
    # Gated on ``spec.paged``, not on ``spec.runtime_shape``: the two branches bound
    # two DIFFERENT buffers, so this is not the removable bookkeeping split that the
    # batch/seqlen param declaration was.
    if spec.paged:
        # The paged bound is the WHOLE page cache. Pages are addressed through
        # block_tables at page_id*block_size, so any page id the table can name must
        # be in range -- batch*seqlen_kv is only the logical KV length of one
        # request and is <= (usually <<) the cache extent. Bounding by it would put
        # legitimately-high page ids past num_records, where buffer loads return 0
        # with no fault. ``num_kv_blocks`` is not a kernel param, so this stays baked.
        _kv_cache_elems = spec.num_kv_blocks * spec.block_size * Hkv * D * 2
        k_rsrc = b.buffer_rsrc(k, b.const_i32(_kv_cache_elems))
        v_rsrc = b.buffer_rsrc(v, b.const_i32(_kv_cache_elems))
    else:
        # Contiguous K/V: num_records must track THIS launch's real extent, since
        # under runtime_shape one kernel serves every seqlen and a baked bound would
        # drop valid reads for a larger one. For the baked sub-modes the product is
        # exactly the B*Skv this used to bake -- run_attention_dense_torch fills the
        # params from the same spec -- so only the emitted IR changes.
        _kv_elem_bytes = Hkv * D * 2
        k_rsrc = b.buffer_rsrc(
            k, b.mul(b.mul(batch_p, seqlen_kv_p), b.const_i32(_kv_elem_bytes))
        )
        v_rsrc = b.buffer_rsrc(
            v, b.mul(b.mul(batch_p, seqlen_kv_p), b.const_i32(_kv_elem_bytes))
        )
    # Unused, but the lowering keeps dead ops, so dropping it moves every golden.
    b.zext(b.to_sgpr_u32(b.mul(wave, b.const_i32(64 * 16))), I64)
    if spec.paged:
        # ROWS_PER_WAVE <= block_size and block_size % ROWS_PER_WAVE == 0 is
        # enforced at spec construction (__post_init__ paged validation), so every
        # wave's K/V rows fall within one page -- the per-wave block_tables hoist
        # below relies on that.
        # Single-seq: seq_base folds to 0; the bt*bt_stride form keeps bt_stride
        # live and generalizes to multi-seq. kv_lens[bt] bounds the page index.
        _pg_seq_base = b.mul(bt, bt_stride)
        _pg_kv_len = b.global_load_i32(kv_lens, bt)
        _pg_n_pages = b.div(
            b.add(_pg_kv_len, b.const_i32(spec.block_size - 1)),
            b.const_i32(spec.block_size),
        )

    def _async_load(rsrc, lds_base, buf_val, tile_key0, bytes_per_buf, group_bytes):
        """Async DMA one K/V tile into its LDS layout.

        ROWS_PER_INSTR==1 (D==128): one instr per padded row -- 64 lanes x 2 bf16
        fill exactly one D-wide row (the original byte-identical fast path), and
        a group IS a row.
        ROWS_PER_INSTR>1 (D<128): pack ROWS_PER_INSTR rows per instr into one
        contiguous group (lane l -> row l//(D/2), col 2*(l%(D/2))). The group is
        placed at group_bytes stride, which carries K's inter-group pad; V's
        group_bytes is the plain unpadded ROWS_PER_INSTR*D pitch, so its layout
        is unchanged."""
        buf_off = b.mul(b.zext(buf_val, I64), b.const_i64(bytes_per_buf))
        if ROWS_PER_INSTR == 1:
            if spec.paged:
                # All ROWS_PER_WAVE rows of this wave fall in ONE page (asserted at
                # setup), so the block_tables lookup is wave-uniform -- hoist it out
                # of the row loop (was per-row: ~ROWS_PER_WAVE x fewer indirection
                # loads + div/mask). Per-row cost is then just a mod + add.
                _wg0 = b.add(tile_key0, b.mul(wave, b.const_i32(ROWS_PER_WAVE)))
                _wpage = b.div(_wg0, b.const_i32(spec.block_size))
                _wphys = b.masked_global_load(
                    block_tables,
                    b.add(_pg_seq_base, _wpage),
                    b.cmp_lt(_wpage, _pg_n_pages),
                    b.const_i32(0),
                    dtype=I32,
                    align=4,
                )
                _wphys_base = b.mul(_wphys, b.const_i32(spec.block_size))
                # NOTE: _wphys (the physical block id) is used raw -- intentionally
                # NOT range-checked here. The K/V read goes through a bounds-checked
                # CDNA buffer SRD (buffer_rsrc word3 0x00027000, num_records = whole
                # cache = _kv_cache_elems), so an out-of-range id yields a voff beyond
                # num_records and raw.ptr.buffer.load.lds drops it / fills 0 rather
                # than reading OOB (contained, but SILENT wrong output on a malformed
                # table). This backstop holds ONLY while the whole offset stays in the
                # i32 voffset; the deferred i64 path folds physical_block into a
                # 64-bit base (bypassing num_records) and MUST add an explicit id
                # guard there.
            for r in range(ROWS_PER_WAVE):
                row = b.add(b.mul(wave, b.const_i32(ROWS_PER_WAVE)), b.const_i32(r))
                row_lds_off = b.add(
                    buf_off, b.zext(b.mul(row, b.const_i32(group_bytes)), I64)
                )
                row_base = b.smem_ptr_add(lds_base, row_lds_off)
                gkey = b.add(tile_key0, row)
                gcol = b.mul(lane, b.const_i32(2))
                if spec.paged:
                    kv_row = b.add(
                        _wphys_base, b.mod(gkey, b.const_i32(spec.block_size))
                    )
                else:
                    kv_row = gkey
                voff = b.add(
                    b.add(k_base, b.mul(kv_row, b.const_i32(stride_k_tok))), gcol
                )
                b.async_buffer_load_lds_addr(
                    rsrc, row_base, b.mul(voff, b.const_i32(2)), zero_soff, 1
                )
        else:
            lanes_per_row = D // 2
            sub_row = b.div(lane, b.const_i32(lanes_per_row))
            col = b.mul(b.mod(lane, b.const_i32(lanes_per_row)), b.const_i32(2))
            groups_per_wave = ROWS_PER_WAVE // ROWS_PER_INSTR
            for it in range(groups_per_wave):
                row0 = b.add(
                    b.mul(wave, b.const_i32(ROWS_PER_WAVE)),
                    b.const_i32(it * ROWS_PER_INSTR),
                )
                grp = b.add(b.mul(wave, b.const_i32(groups_per_wave)), b.const_i32(it))
                row_lds_off = b.add(
                    buf_off, b.zext(b.mul(grp, b.const_i32(group_bytes)), I64)
                )
                row_base = b.smem_ptr_add(lds_base, row_lds_off)
                gkey = b.add(b.add(tile_key0, row0), sub_row)
                voff = b.add(b.add(k_base, b.mul(gkey, b.const_i32(stride_k_tok))), col)
                b.async_buffer_load_lds_addr(
                    rsrc, row_base, b.mul(voff, b.const_i32(2)), zero_soff, 1
                )

    def _async_load_wide(
        rsrc, lds_base, buf_val, tile_key0, bytes_per_buf, line_stride
    ):
        """Two 128-bit-per-lane DMAs fill one slab-padded K or V tile per line pass
        (the persistent wide path's layout)."""
        buf_off = b.mul(b.zext(buf_val, I64), b.const_i64(bytes_per_buf))
        n_in_wave = b.div(lane, b.const_i32(8))
        d_bucket = b.mod(lane, b.const_i32(8))
        n_rpt = BN // 8
        for n_pass in range(WIDE_LINE_PASSES):
            line_id = b.add(wave, b.const_i32(n_pass * WAVES))
            row = b.add(tile_key0, b.add(b.mul(n_in_wave, b.const_i32(8)), line_id))
            src_base = b.add(
                b.add(k_base, b.mul(row, b.const_i32(stride_k_tok))),
                b.mul(d_bucket, b.const_i32(8)),
            )
            for d_rpt in range(D // 64):
                line = b.add(
                    b.mul(line_id, b.const_i32(line_stride * 2)),
                    b.const_i32(d_rpt * n_rpt * line_stride * 2),
                )
                row_base = b.smem_ptr_add(lds_base, b.add(buf_off, b.zext(line, I64)))
                voff = b.add(src_base, b.const_i32(d_rpt * 64))
                b.async_buffer_load_lds_addr(
                    rsrc, row_base, b.mul(voff, b.const_i32(2)), zero_soff, 4
                )

    def async_load_k(lds_base, buf_val, tile_key0):
        if WIDE_DMA:
            _async_load_wide(
                k_rsrc, lds_base, buf_val, tile_key0, K_BYTES_PER_BUF, K_LINE_STRIDE
            )
            return
        _async_load(
            k_rsrc, lds_base, buf_val, tile_key0, K_BYTES_PER_BUF, K_GROUP_BYTES
        )

    def async_load_v(lds_base, buf_val, tile_key0):
        if WIDE_DMA:
            _async_load_wide(
                v_rsrc, lds_base, buf_val, tile_key0, V_BYTES_PER_BUF, V_LINE_STRIDE
            )
            return
        _async_load(
            v_rsrc, lds_base, buf_val, tile_key0, V_BYTES_PER_BUF, V_GROUP_BYTES
        )

    def load_tile(buf_val, tile_idx):
        tk0 = b.mul(tile_idx, b.const_i32(BN))
        async_load_k(K_lds_addr, buf_val, tk0)
        async_load_v(V_lds_addr, buf_val, tk0)

    # ---- per-tile compute closures ----

    if LDS_READ_BASE:
        # lds_read_base: the per-lane element offset inside one K buffer. Each
        # tile adds its buffer offset once; every read then adds only a
        # compile-time constant, which lands in the ds_read offset field.
        _z = b.const_i32(0)
        k_lane_elem = b.add(
            b.mul(b.mod(lane_m, b.const_i32(8)), b.const_i32(K_LINE_STRIDE)),
            b.add(
                b.mul(b.div(lane_m, b.const_i32(8)), b.const_i32(64)),
                b.mul(lane_h, b.const_i32(8)),
            ),
        )

    def k_tile_base(kbuf):
        return b.add(k_lane_elem, b.mul(kbuf, b.const_i32(K_BYTES_PER_BUF // 2)))

    def read_k_at(kt, nsub, ks):
        off = (ks // 4) * K_N_RPT * K_LINE_STRIDE + nsub * 256 + (ks % 4) * 16
        return b.smem_load_vN(
            K_lds, _z, _z, _z, b.add(kt, b.const_i32(off)), dtype=dtype, n=8
        )

    def _do_qk_read_ahead(kbuf, after_mfma, accs, tail_read=None):
        """``operand_read_ahead``: each K operand read runs READ_AHEAD MFMAs
        ahead of its use, fenced like the PV run (see ``do_pv``). Once the K
        reads run out, ``tail_read`` (optional) fills the same slots, so the
        next phase's first operands are in flight before this run ends."""
        kt = k_tile_base(kbuf)
        steps = [(nsub, ks) for nsub in range(N_SUB) for ks in range(K_STEPS)]
        packs = [None] * len(steps)
        for k in range(min(READ_AHEAD, len(steps))):
            packs[k] = read_k_at(kt, *steps[k])
        s_reg = []
        acc = None
        for idx, (nsub, ks) in enumerate(steps):
            if ks == 0:
                acc = b.zero_vec_f32(16)
            acc = mfma_32x32x16_for_dtype(b, dtype, packs[idx], q_packs[ks], acc)
            if idx + READ_AHEAD < len(steps):
                packs[idx + READ_AHEAD] = read_k_at(kt, *steps[idx + READ_AHEAD])
            elif tail_read is not None:
                tail_read()
            b.sched_barrier(_SCHED_ALU_ONLY)
            if after_mfma is not None:
                after_mfma()
            if ks == K_STEPS - 1:
                if accs is not None:
                    accs.append(acc)
                s_reg.append([b.vec_extract(acc, i) for i in range(16)])
        return s_reg

    def do_qk(kbuf, after_mfma=None, accs=None, tail_read=None):
        """QK MFMA: S^T = K@Q^T. mfma(a=K, bv=Q) => key on the 16 per-lane accumulator
        regs (+lane^32), query on lane%32 -- the layout that keeps softmax a cheap
        in-lane reduce + one lane^32 exchange, and lets CK-1's transposed PV consume P
        with no relayout shuffle.

        ``after_mfma`` (optional) is called after every MFMA so a caller can place
        independent softmax work in program order between them; ``accs`` (optional
        list) receives each nsub's raw accumulator vector."""
        if BULK_K:
            return _do_qk_bulk(kbuf, after_mfma, accs)
        if READ_AHEAD:
            return _do_qk_read_ahead(kbuf, after_mfma, accs, tail_read)
        s_reg = []
        kt = k_tile_base(kbuf) if LDS_READ_BASE else None
        for nsub in range(N_SUB):
            acc = b.zero_vec_f32(16)
            if LDS_READ_BASE:
                pass
            elif WIDE_DMA:
                kline = b.mod(lane_m, b.const_i32(8))
                kelem_base = b.add(
                    b.add(
                        b.mul(b.div(lane_m, b.const_i32(8)), b.const_i32(64)),
                        b.mul(lane_h, b.const_i32(8)),
                    ),
                    b.const_i32(nsub * 256),
                )
            elif K_GROUP == 1:
                krow = b.add(b.const_i32(nsub * 32), lane_m)
            else:
                krow = b.add(b.const_i32(nsub * (32 // K_GROUP)), k_lane_grp)
            for ks in range(K_STEPS):
                if LDS_READ_BASE:
                    k_pack = read_k_at(kt, nsub, ks)
                    acc = mfma_32x32x16_for_dtype(b, dtype, k_pack, q_packs[ks], acc)
                    if after_mfma is not None:
                        after_mfma()
                    continue
                if WIDE_DMA:
                    kelem = b.add(kelem_base, b.const_i32((ks % 4) * 16))
                    k_pack = b.smem_load_vN(
                        K_lds,
                        kbuf,
                        b.const_i32(ks // 4),
                        kline,
                        kelem,
                        dtype=dtype,
                        n=8,
                    )
                    acc = mfma_32x32x16_for_dtype(b, dtype, k_pack, q_packs[ks], acc)
                    if after_mfma is not None:
                        after_mfma()
                    continue
                col = b.add(b.const_i32(ks * 16), d_base)
                if K_GROUP > 1:
                    col = b.add(k_sub_col, col)
                k_pack = b.smem_load_vN(K_lds, kbuf, krow, col, dtype=dtype, n=8)
                acc = mfma_32x32x16_for_dtype(b, dtype, k_pack, q_packs[ks], acc)
                if after_mfma is not None:
                    after_mfma()
            # Raw (unscaled) scores. qk_scale is folded into softmax_max (one
            # multiply on the row max) and into the exp2 argument as one FMA,
            # so the scale adds no per-score instruction. Needs qk_scale > 0
            # (max commutes with the scale); the runner enforces the
            # [_MIN_SCALE, _MAX_SCALE] range.
            if accs is not None:
                accs.append(acc)
            s_reg.append([b.vec_extract(acc, i) for i in range(16)])
        return s_reg

    def _do_qk_bulk(kbuf, after_mfma, accs):
        """``bulk_operand_reads``: issue every K operand read of the tile before
        the first QK MFMA (one LDS wait), so the MFMA run has no LDS waits."""
        packs = []
        kt = k_tile_base(kbuf) if LDS_READ_BASE else None
        for nsub in range(N_SUB):
            if LDS_READ_BASE:
                pass
            elif WIDE_DMA:
                kline = b.mod(lane_m, b.const_i32(8))
                kelem_base = b.add(
                    b.add(
                        b.mul(b.div(lane_m, b.const_i32(8)), b.const_i32(64)),
                        b.mul(lane_h, b.const_i32(8)),
                    ),
                    b.const_i32(nsub * 256),
                )
            else:
                krow = b.add(b.const_i32(nsub * 32), lane_m)
            for ks in range(K_STEPS):
                if LDS_READ_BASE:
                    packs.append(read_k_at(kt, nsub, ks))
                elif WIDE_DMA:
                    kelem = b.add(kelem_base, b.const_i32((ks % 4) * 16))
                    packs.append(
                        b.smem_load_vN(
                            K_lds,
                            kbuf,
                            b.const_i32(ks // 4),
                            kline,
                            kelem,
                            dtype=dtype,
                            n=8,
                        )
                    )
                else:
                    col = b.add(b.const_i32(ks * 16), d_base)
                    packs.append(
                        b.smem_load_vN(K_lds, kbuf, krow, col, dtype=dtype, n=8)
                    )
        b.sched_barrier(0)
        return _qk_mfma_run(packs, after_mfma, accs)

    def _qk_mfma_run(packs, after_mfma, accs):
        """The QK MFMA run over K operands already in registers (tile order:
        nsub-major, then k-step)."""
        s_reg = []
        for nsub in range(N_SUB):
            acc = b.zero_vec_f32(16)
            for ks in range(K_STEPS):
                acc = mfma_32x32x16_for_dtype(
                    b, dtype, packs[nsub * K_STEPS + ks], q_packs[ks], acc
                )
                if after_mfma is not None:
                    after_mfma()
            if accs is not None:
                accs.append(acc)
            s_reg.append([b.vec_extract(acc, i) for i in range(16)])
        return s_reg

    def read_k_tile(kbuf):
        """Every K operand of one tile (lds_read_base addressing)."""
        kt = k_tile_base(kbuf)
        return [read_k_at(kt, n, ks) for n in range(N_SUB) for ks in range(K_STEPS)]

    def do_mask(s_reg, tile_idx, lower=False, upper=True):
        """Apply causal (upper: ktok<=q) and/or sliding-window (lower:
        ktok>q-W) masks in-place on the QK-output layout. W is compile-time so
        the lower threshold folds to an immediate. No relayout (reuses the same
        lane->ktok/query_tok maps as causal)."""
        if not causal:
            return
        tile_key0 = b.mul(tile_idx, b.const_i32(BN))
        query_tok = b.add(q_tok0, _mfma_32x32_c_col(b, lane, 0))
        if DIAG_OFF:
            query_tok = b.add(query_tok, b.const_i32(DIAG_OFF))
        # lower bound key: q - W + 1  (keep iff ktok > q - W)
        win_lo = b.sub(query_tok, b.const_i32(W)) if lower else None
        for nsub in range(N_SUB):
            sub_base = b.add(tile_key0, b.const_i32(nsub * 32))
            for i in range(16):
                ktok = b.add(sub_base, _mfma_32x32_c_row(b, lane, i))
                if upper:
                    s_reg[nsub][i] = b.select(
                        b.cmp_le(ktok, query_tok), s_reg[nsub][i], neg_inf
                    )
                if lower:
                    s_reg[nsub][i] = b.select(
                        b.cmp_gt(ktok, win_lo), s_reg[nsub][i], neg_inf
                    )

    def do_kbound_mask(s_reg, tile_idx):
        """ragged non-causal: force scores of padded keys (ktok >= seqlen_kv, the
        OOB rows of the partial last KV tile) to -inf. Causal doesn't need this
        (padded ktok >= seqlen_kv > every real query, so causal already drops
        them). seqlen_kv is compile-time -> the bound folds to an immediate."""
        tile_key0 = b.mul(tile_idx, b.const_i32(BN))
        for nsub in range(N_SUB):
            sub_base = b.add(tile_key0, b.const_i32(nsub * 32))
            for i in range(16):
                ktok = b.add(sub_base, _mfma_32x32_c_row(b, lane, i))
                s_reg[nsub][i] = b.select(
                    b.cmp_lt(ktok, b.const_i32(Skv)), s_reg[nsub][i], neg_inf
                )

    def softmax_max(s_reg, m_i):
        local_max = neg_inf
        for nsub in range(N_SUB):
            for i in range(16):
                local_max = b.fmax(local_max, s_reg[nsub][i])
        return _softmax_max_from_local(local_max, m_i)

    def half_combine(x, op):
        """op(x, x from lane ^ 32). ``permlane32_reduce``: one VALU
        v_permlane32_swap_b32 (both halves of the result carry the other half's
        value) instead of a ds_bpermute round trip through LDS."""
        if not spec.permlane32_reduce:
            return op(x, b.warp_shuffle_xor(x, 32))
        xi = b.bitcast(x, I32)
        lo, hi = b.permlane32_swap(xi, xi)
        return op(b.bitcast(lo, F32), b.bitcast(hi, F32))

    def _softmax_max_from_local(local_max, m_i):
        """Finish the running max from this lane's partial row max."""
        tile_max_raw = half_combine(local_max, b.fmax)
        # Into the log2 domain the running max, threshold, and sinks use.
        tile_max = b.fmul(tile_max_raw, qk_scale)
        if LAZY_RESCALE:
            m_diff = b.fsub(tile_max, m_i)
            below_i32 = b.select(
                b.fcmp("ole", m_diff, b.const_f32(LAZY_THRESHOLD)),
                b.const_i32(1),
                b.const_i32(0),
            )
            skip = b.cmp_ne(b.wave_all(below_i32), b.const_i32(0))
            m_new = b.select(skip, m_i, b.fmax(m_i, tile_max))
        else:
            skip = None
            m_new = b.fmax(m_i, tile_max)
        alpha = _exp2(b.fsub(m_i, m_new))
        return m_new, alpha, skip

    def relayout_p(p):
        """CK-1 half-local P feed: assemble the PV B-operand from lane-local P regs
        only (a bf16 cast + pack, NO cross-half warp_shuffle_xor/select). Pairs with
        the half-local V load in ``read_v`` so the K-axis stays aligned."""
        packs = []
        for kk_step in range(KK_STEPS):
            elems = []
            for kk in range(8):
                local_in_group = kk % 4
                band = kk // 4
                key_idx = kk_step * 16 + band * 8 + local_in_group
                p_tile = key_idx // 32
                row_static = key_idx % 32
                preg = (row_static // 8) * 4 + (row_static % 4)
                elems.append(b.cast_f32_to(p[p_tile][preg], dtype))
            packs.append(b.vec_pack(elems, dtype))
        return packs

    if WIDE_DMA:
        vline = b.add(
            b.mul(lane_h, b.const_i32(4)),
            b.div(b.mod(lane, b.const_i32(16)), b.const_i32(4)),
        )
        velem_lane = b.add(
            b.mul(
                b.mod(b.div(lane, b.const_i32(16)), b.const_i32(2)),
                b.const_i32(16),
            ),
            b.mul(b.mod(lane, b.const_i32(4)), b.const_i32(4)),
        )
        if LDS_READ_BASE:
            v_lane_elem = b.add(b.mul(vline, b.const_i32(V_LINE_STRIDE)), velem_lane)

    def v_operand_base(vbuf):
        """lds_read_base: the tile's per-lane V element base (lane offset plus
        buffer offset), formed once per PV run; otherwise ``vbuf`` itself."""
        if not LDS_READ_BASE:
            return vbuf
        return b.add(v_lane_elem, b.mul(vbuf, b.const_i32(V_BYTES_PER_BUF // 2)))

    def read_v(dt, kk_step, vbuf):
        """CK-1 half-local transposed V A-operand (matches ``relayout_p``).
        Under lds_read_base ``vbuf`` is the tile base from ``v_operand_base``."""
        if LDS_READ_BASE:
            off = (dt // 2) * V_N_RPT * V_LINE_STRIDE + (dt % 2) * 32 + kk_step * 128
            a0 = b.ds_read_tr16_b64(
                V_lds, _z, _z, _z, b.add(vbuf, b.const_i32(off)), dtype=dtype
            )
            a1 = b.ds_read_tr16_b64(
                V_lds, _z, _z, _z, b.add(vbuf, b.const_i32(off + 64)), dtype=dtype
            )
            return b.vec_concat(a0, a1)
        if WIDE_DMA:
            velem = b.add(velem_lane, b.const_i32((dt % 2) * 32 + kk_step * 128))
            a0 = b.ds_read_tr16_b64(
                V_lds, vbuf, b.const_i32(dt // 2), vline, velem, dtype=dtype
            )
            a1 = b.ds_read_tr16_b64(
                V_lds,
                vbuf,
                b.const_i32(dt // 2),
                vline,
                b.add(velem, b.const_i32(64)),
                dtype=dtype,
            )
            return b.vec_concat(a0, a1)
        return pv32_v_load_paired(
            b,
            V_lds=V_lds,
            v_buf=vbuf,
            n=dt,
            k=kk_step,
            lane_half32=lane_h,
            lane_col32=lane_m,
            dtype=dtype,
        )

    def v_prefetch(vbuf):
        """operand_read_ahead: lets an earlier MFMA run (do_qk's ``tail_read``)
        start this tile's first PV operand reads. Returns (read_one, state);
        pass ``state`` to do_pv as ``pre``."""
        st = {"base": v_operand_base(vbuf), "ops": []}

        def read_one():
            k = len(st["ops"])
            if k < min(READ_AHEAD, len(PV_STEPS)):
                st["ops"].append(read_v(*PV_STEPS[k], st["base"]))

        return read_one, st

    def do_pv(o_acc_in, p_packs, vbuf, after_mfma=None, bulk=False, pre=None):
        out = list(o_acc_in)
        vbuf = pre["base"] if pre is not None else v_operand_base(vbuf)
        n_steps = len(PV_STEPS)
        if bulk:
            # bulk_operand_reads: all V operand reads ahead of the PV MFMA run.
            vops = [read_v(dt, kk_step, vbuf) for dt, kk_step in PV_STEPS]
            b.sched_barrier(0)
        elif READ_AHEAD:
            # operand_read_ahead: V operand reads run READ_AHEAD MFMAs ahead of
            # their use; each fence keeps MFMAs and LDS reads in this order
            # while VALU, SALU and transcendental work may still move across.
            vops = [None] * n_steps
            done = list(pre["ops"]) if pre is not None else []
            vops[: len(done)] = done
            for k in range(len(done), min(READ_AHEAD, n_steps)):
                vops[k] = read_v(*PV_STEPS[k], vbuf)
        for idx, (dt, kk_step) in enumerate(PV_STEPS):
            if bulk or READ_AHEAD:
                vop = vops[idx]
            else:
                vop = read_v(dt, kk_step, vbuf)
            out[dt] = mfma_32x32x16_for_dtype(b, dtype, vop, p_packs[kk_step], out[dt])
            if READ_AHEAD and not bulk:
                if idx + READ_AHEAD < n_steps:
                    vops[idx + READ_AHEAD] = read_v(*PV_STEPS[idx + READ_AHEAD], vbuf)
                b.sched_barrier(_SCHED_ALU_ONLY)
            if after_mfma is not None:
                after_mfma()
        return out

    def read_v_tile(vbuf):
        """Every V operand of one tile, in PV_STEPS order."""
        base = v_operand_base(vbuf)
        return [read_v(dt, kk_step, base) for dt, kk_step in PV_STEPS]

    def pv_mfma_run(o_acc_in, p_packs, vops, after_mfma=None):
        """The PV MFMA run over V operands already in registers."""
        out = list(o_acc_in)
        for idx, (dt, kk_step) in enumerate(PV_STEPS):
            out[dt] = mfma_32x32x16_for_dtype(
                b, dtype, vops[idx], p_packs[kk_step], out[dt]
            )
            if after_mfma is not None:
                after_mfma()
        return out

    def rescale_o(o_acc, alpha):
        return [
            b.vec_pack(
                [b.fmul(b.vec_extract(o_acc[dt], i), alpha) for i in range(16)], F32
            )
            for dt in range(D_TILES)
        ]

    def pv_fused_exp(o_acc_in, p_packs, vbuf, s_reg, m_new):
        """Depth-1 cluster: interleave exp2(s * qk_scale - m_new) into the PV MFMA loop so the
        softmax VALU/TRANS co-executes in the MFMA shadow. The full per-step
        instruction population (DS_READ/MFMA/VALU/TRANS) is named to sched_group_barrier
        so the IGLP grouping matches the real stream. The VALU count is an upper
        bound: the compiler may pair two exp-argument FMAs into one v_pk_fma_f32, so a
        step can issue fewer VALU ops than named."""
        # fma(s, qk_scale, -m) is not bounded for huge scores: see the KNOWN
        # LIMITATION note at _MAX_SCALE.
        neg_m = b.fneg(m_new)
        slots = [(nsub, i) for nsub in range(N_SUB) for i in range(16)]
        p_vals = [[None] * 16 for _ in range(N_SUB)]
        it = iter(slots)
        out = list(o_acc_in)
        vbuf = v_operand_base(vbuf)
        for dt, kk_step in PV_STEPS:
            out[dt] = mfma_32x32x16_for_dtype(
                b, dtype, read_v(dt, kk_step, vbuf), p_packs[kk_step], out[dt]
            )
            n_emit = 0
            for _ in range(EXP_PER):
                slot = next(it, None)
                if slot is None:
                    break
                nsub, i = slot
                p_vals[nsub][i] = _exp2(b.fma(s_reg[nsub][i], qk_scale, neg_m))
                n_emit += 1
            if SCHED_GROUP:
                b.sched_group_barrier(DS_READ, SCHED_GROUP_DS_READ, 0)
                b.sched_group_barrier(MFMA, 1, 0)
                b.sched_group_barrier(VALU, max(1, n_emit), 0)
                b.sched_group_barrier(TRANS, max(1, n_emit), 0)
        for slot in it:
            nsub, i = slot
            p_vals[nsub][i] = _exp2(b.fma(s_reg[nsub][i], qk_scale, neg_m))
        l_local = b.const_f32(0.0)
        for nsub in range(N_SUB):
            for i in range(16):
                l_local = b.fadd(l_local, p_vals[nsub][i])
        if spec.defer_row_sum:
            l_tile = l_local
        else:
            l_tile = half_combine(l_local, b.fadd)
        return out, p_vals, l_tile

    if varlen:
        n_ktiles_val = b.div(seqlen_kv_b, b.const_i32(BN))
    elif spec.runtime_shape:
        n_ktiles_val = b.div(seqlen_kv_p, b.const_i32(BN))
    else:
        n_ktiles_val = b.const_i32(n_ktiles)
    if causal:
        # Ceil the final reachable key to a KV-tile count. block_m is a
        # per-spec geometry choice; supports_attention_dense enforces that BN
        # divides it, so the qb term can stay outside the ceil.
        n_upper = b.add(
            b.mul(qb, b.const_i32(n_per)),
            b.const_i32((spec.block_m - 1 + DIAG_OFF) // BN + 1),
        )
        n_upper = b.select(b.cmp_lt(n_upper, n_ktiles_val), n_upper, n_ktiles_val)
    else:
        n_upper = n_ktiles_val

    # Sliding-window: first KV tile any row in this block attends to. Valid band
    # is [start_tile, n_upper); tiles < start_tile are fully outside the window
    # (all -inf) so they are never visited (the KV-loop prune). W==0 keeps
    # start_tile=0 -> full causal, byte-identical to the always-on path.
    if causal and W > 0:
        _diag0 = b.mul(qb, b.const_i32(n_per))
        _lo_raw = b.sub(_diag0, b.const_i32(Wt))
        start_tile = b.select(
            b.cmp_gt(_lo_raw, b.const_i32(0)), _lo_raw, b.const_i32(0)
        )
    else:
        start_tile = b.const_i32(0)
    start_buf = b.mod(start_tile, b.const_i32(NBUF))
    start_buf1 = b.mod(b.add(start_tile, b.const_i32(1)), b.const_i32(NBUF))

    def window_barrier(lgkmcnt=-1, vmcnt=-1, opens=None):
        """mfma_windows: one window boundary. The fences keep every MFMA and
        LDS access of a window on its side of the barrier. ``opens`` ("mfma"
        or "read") names the next window for window_priority 1 and 2."""
        b.sched_barrier(0)
        if lgkmcnt >= 0 or vmcnt >= 0:
            b.s_waitcnt(vmcnt=vmcnt, lgkmcnt=lgkmcnt)
        b.s_barrier_bare()
        if opens is not None and WIN_PRIO in (1, 2):
            b.s_setprio(int((opens == "mfma") == (WIN_PRIO == 1)))
        b.sched_barrier(0)

    def _emit_lookahead_main():
        """QK-lookahead KV loop (``qk_lookahead``): one tile of software skew
        inside the wave so every MFMA run has independent softmax VALU work.

        Iteration j holds the raw scores S_j and the running max m_j (which
        already includes tile j), and O / l scaled to m_j:
          phase A: S_{j+1} = QK(j+1)  ||  P_j = exp2(S_j*qk_scale - m_j), l += sum(P_j)
          phase B: O += P_j V_j        ||  mask + row max of S_{j+1} -> m_{j+1}
          then O, l *= exp2(m_j - m_{j+1}) (lazy: skipped when the max stays).
        LDS: tile t lives in buffer t % 2. One barrier per iteration: at the top
        of iteration j every wave has finished QK(j) and PV(j-1), so K(j+2)
        refills K buffer j % 2 and V(j+1) refills V buffer (j+1) % 2; both have a
        whole iteration to land before the next top-of-loop wait.
        Returns (m, l, o_acc) for the shared epilogue."""
        sb = start_buf
        sb1 = start_buf1
        tk_s = b.mul(start_tile, b.const_i32(BN))
        async_load_k(K_lds_addr, sb, tk_s)
        async_load_v(V_lds_addr, sb, tk_s)
        async_load_k(
            K_lds_addr, sb1, b.mul(b.add(start_tile, b.const_i32(1)), b.const_i32(BN))
        )
        b.s_waitcnt(vmcnt=0)
        b.s_barrier_bare()
        acc0 = []
        s0 = do_qk(sb, accs=acc0)
        do_mask(s0, start_tile)
        if WINDOWS:
            # mfma_windows prefetches K two tiles ahead: once every wave is
            # done reading K(start) it refills that buffer with K(start + 2).
            window_barrier(lgkmcnt=0)
            async_load_k(
                K_lds_addr,
                sb,
                b.mul(b.add(start_tile, b.const_i32(2)), b.const_i32(BN)),
            )
        if use_sinks:
            sink_h = b.global_load(sinks, hq, dtype, align=2)
            m_init = b.fmul(b.cast_to_f32(sink_h), rcp_ln2)
        else:
            m_init = neg_inf
        m0, alpha0, _skip0 = softmax_max(s0, m_init)
        if use_sinks:
            # The sink's exp(sink - m_init) = 1 contribution, rescaled to m0.
            l0 = b.fmul(one_f, alpha0)
            if spec.defer_row_sum:
                l0 = b.select(b.cmp_eq(lane_h, b.const_i32(0)), l0, b.const_f32(0.0))
        else:
            l0 = b.const_f32(0.0)
        o0 = [b.zero_vec_f32(16) for _ in range(D_TILES)]
        if causal:
            s0v = [b.vec_pack(s0[n], F32) for n in range(N_SUB)]
        else:
            s0v = acc0
        la_args = (
            [("m", m0), ("l", l0)]
            + [(f"o{dt}", o0[dt]) for dt in range(D_TILES)]
            + [(f"s{n}", s0v[n]) for n in range(N_SUB)]
        )
        exp_per_qk = -(-((KK_STEPS - PV_EXP) * 8) // (N_SUB * K_STEPS))
        # pv_window_exp: deferred exps spread over the PV MFMAs that do not
        # consume a deferred pack (k_major: the first D_TILES * kept packs).
        exp_per_pv = -(-(PV_EXP * 8) // (D_TILES * (KK_STEPS - PV_EXP)))
        max_per_pv = -(-(N_SUB * 16) // len(PV_STEPS))
        la_rs_ctr = [0]
        # (nsub, reg) feeding each PV B-operand pack, in relayout_p's order.
        _P_ORDER = []
        for kk_step in range(KK_STEPS):
            row = []
            for kk in range(8):
                key_idx = kk_step * 16 + (kk // 4) * 8 + kk % 4
                rs = key_idx % 32
                row.append((key_idx // 32, (rs // 8) * 4 + rs % 4))
            _P_ORDER.append(row)

        def la_body(j, carry, mask_next):
            if IGLP != _IGLP_OFF:
                b.iglp_opt(IGLP)
            m_j = carry[0]
            l_i = carry[1]
            o_acc = list(carry[2 : 2 + D_TILES])
            s_j = [
                [b.vec_extract(carry[2 + D_TILES + n], i) for i in range(16)]
                for n in range(N_SUB)
            ]
            j1 = b.add(j, b.const_i32(1))
            buf_j = b.mod(j, b.const_i32(NBUF))
            buf_j1 = b.mod(j1, b.const_i32(NBUF))
            if WINDOWS:
                # Window 1 (operands; opened by the previous tile's last
                # boundary, or the prologue): refill V buffer (j+1) % 2, whose
                # V(j-1) both groups read before that boundary, and read every
                # K(j+1) operand. The previous tile's softmax finish and O
                # rescale also run here. The closing wait makes V(j) land one
                # boundary ahead of its reads in window 3.
                async_load_v(V_lds_addr, buf_j1, b.mul(j1, b.const_i32(BN)))
                kops = read_k_tile(buf_j1)
                window_barrier(lgkmcnt=0, vmcnt=WIN_VMCNT, opens="mfma")
            else:
                kops = None
                b.s_waitcnt(vmcnt=0)
                b.s_barrier_bare()
                async_load_k(
                    K_lds_addr, buf_j, b.mul(b.add(j, b.const_i32(2)), b.const_i32(BN))
                )
            if not PINGPONG:
                async_load_v(V_lds_addr, buf_j1, b.mul(j1, b.const_i32(BN)))

            # Phase A: QK(j+1) with the exp2 of tile j in its shadow. The exps
            # run in PV-operand order so each bf16 P pack, and the two partial
            # row sums, are emitted as soon as their inputs exist instead of as
            # a serial tail after the last MFMA.
            neg_m = b.fneg(m_j)
            all_slots = [
                (kk, e, n, i)
                for kk, row in enumerate(_P_ORDER)
                for e, (n, i) in enumerate(row)
            ]
            n_qk_slots = (KK_STEPS - PV_EXP) * 8
            slots = iter(all_slots[:n_qk_slots])
            pv_slots = iter(all_slots[n_qk_slots:])
            pk_elems = [[None] * 8 for _ in range(KK_STEPS)]
            p_packs = [None] * KK_STEPS
            lsum = [None, None]

            def emit_exp(kk, e, n, i):
                p = _exp2(b.fma(s_j[n][i], qk_scale, neg_m))
                h = e % 2
                lsum[h] = p if lsum[h] is None else b.fadd(lsum[h], p)
                pk_elems[kk][e] = b.cast_f32_to(p, dtype)
                if e == 7:
                    p_packs[kk] = b.vec_pack(pk_elems[kk], dtype)

            def exp_hook():
                for _ in range(exp_per_qk):
                    slot = next(slots, None)
                    if slot is None:
                        return
                    emit_exp(*slot)

            # V(j) landed before this iteration's top barrier, so the first PV
            # operand reads may already run in the QK MFMA shadow.
            pv_read, pv_pre = v_prefetch(buf_j) if READ_AHEAD else (None, None)
            if WINDOWS:
                # Window 2 (MFMA): QK(j+1) from the K operands of window 1.
                s_next = _qk_mfma_run(kops, exp_hook, None)
            else:
                s_next = do_qk(buf_j1, after_mfma=exp_hook, tail_read=pv_read)
            for slot in slots:
                emit_exp(*slot)

            def finish_row_sum():
                l_local = b.fadd(lsum[0], lsum[1])
                if not spec.defer_row_sum:
                    l_local = half_combine(l_local, b.fadd)
                return b.fadd(l_i, l_local)

            if not PV_EXP:
                l_acc = finish_row_sum()

            # Phase B: PV(j) with the mask + row max of tile j+1 in its shadow.
            vops = None
            if WINDOWS:
                # Window 3 (operands): K(j+3) refills K buffer (j+1) % 2, which
                # both groups finished reading in window 1, then every V(j)
                # operand is read; K(j+2) must land one boundary early.
                window_barrier(opens="read")
                async_load_k(
                    K_lds_addr, buf_j1, b.mul(b.add(j, b.const_i32(3)), b.const_i32(BN))
                )
                vops = read_v_tile(buf_j)
                window_barrier(lgkmcnt=0, vmcnt=WIN_VMCNT, opens="mfma")
            elif PINGPONG:
                # Second barrier: the lagging group runs phase A of this tile
                # while the leading group runs phase B. V(j+1) refills V buffer
                # (j+1) % 2 only after both groups are past PV(j-1).
                b.s_waitcnt(vmcnt=0)
                b.s_barrier_bare()
                async_load_v(V_lds_addr, buf_j1, b.mul(j1, b.const_i32(BN)))
            if mask_next:
                do_mask(s_next, j1)
            maxes = [neg_inf]
            mslots = iter([(n, i) for n in range(N_SUB) for i in range(16)])

            def max_hook():
                for _ in range(max_per_pv):
                    slot = next(mslots, None)
                    if slot is None:
                        return
                    n, i = slot
                    maxes[0] = b.fmax(maxes[0], s_next[n][i])

            if PV_PRIORITY and not PINGPONG:
                b.s_setprio(PV_PRIORITY)
            if WINDOWS:
                # Window 4 (MFMA): PV(j). Its closing boundary opens the next
                # tile's window 1, which also carries the softmax finish and
                # the O rescale below.
                if PV_EXP:
                    o_acc = list(o_acc)
                    for idx, (dt, kk_step) in enumerate(PV_STEPS):
                        if p_packs[kk_step] is None:
                            for slot in pv_slots:
                                emit_exp(*slot)
                        o_acc[dt] = mfma_32x32x16_for_dtype(
                            b, dtype, vops[idx], p_packs[kk_step], o_acc[dt]
                        )
                        for _ in range(exp_per_pv):
                            slot = next(pv_slots, None)
                            if slot is not None:
                                emit_exp(*slot)
                        max_hook()
                    l_acc = finish_row_sum()
                else:
                    o_acc = pv_mfma_run(o_acc, p_packs, vops, after_mfma=max_hook)
                window_barrier(opens="read")
            else:
                o_acc = do_pv(
                    o_acc, p_packs, buf_j, after_mfma=max_hook, bulk=BULK_V, pre=pv_pre
                )
            if PV_PRIORITY and not PINGPONG:
                b.s_setprio(0)
            for n, i in mslots:
                maxes[0] = b.fmax(maxes[0], s_next[n][i])
            m_new, alpha, skip = _softmax_max_from_local(maxes[0], m_j)
            if LAZY_RESCALE:
                la_rs_ctr[0] += 1
                tg = la_rs_ctr[0]
                trips = b.select(skip, b.const_i32(0), b.const_i32(1))
                rs_args = [(f"lo{dt}_{tg}", o_acc[dt]) for dt in range(D_TILES)]
                rs_args.append((f"ll_{tg}", l_acc))
                rs = b.scf_for_iter(
                    b.const_i32(0), trips, b.const_i32(1), rs_args, iv_name=f"lrs{tg}"
                )
                with rs as (_iv, rc):
                    o_sc = rescale_o(list(rc[:D_TILES]), alpha)
                    b.scf_yield(*o_sc, b.fmul(rc[D_TILES], alpha))
                o_acc = list(rs.results[:D_TILES])
                l_new = rs.results[D_TILES]
            else:
                o_acc = rescale_o(o_acc, alpha)
                l_new = b.fmul(l_acc, alpha)
            s_next_v = [b.vec_pack(s_next[n], F32) for n in range(N_SUB)]
            return [m_new, l_new, *o_acc, *s_next_v]

        def la_loop(lo, hi, args, mask_next, tag):
            """KV loop over [lo, hi). ``qk_lookahead_unroll=2`` emits two tiles per
            trip, so the carried scores alternate between two register sets
            instead of being copied back at every back edge, plus a 0/1-trip
            remainder loop."""
            if UNROLL == 1:
                lp = b.scf_for_iter(lo, hi, b.const_i32(1), args, iv_name=tag)
                with lp as (j, carry):
                    b.scf_yield(*la_body(j, carry, mask_next))
                return lp
            span = b.sub(hi, lo)
            span = b.select(b.cmp_lt(span, b.const_i32(0)), b.const_i32(0), span)
            hi2 = b.add(lo, b.mul(b.div(span, b.const_i32(2)), b.const_i32(2)))
            lp2 = b.scf_for_iter(lo, hi2, b.const_i32(2), args, iv_name=tag + "2")
            with lp2 as (j, carry):
                c1 = la_body(j, carry, mask_next)
                b.scf_yield(*la_body(b.add(j, b.const_i32(1)), c1, mask_next))
            rem_args = [(name + "_r", val) for (name, _), val in zip(args, lp2.results)]
            lp1 = b.scf_for_iter(hi2, hi, b.const_i32(1), rem_args, iv_name=tag + "1")
            with lp1 as (j, carry):
                b.scf_yield(*la_body(j, carry, mask_next))
            return lp1

        last = b.sub(n_upper, b.const_i32(1))
        if PINGPONG:
            # Waves [WAVES/2, WAVES) lag the leading half by one barrier and
            # win issue arbitration; the leading half pays the extra barrier
            # back after the loop so both halves meet the epilogue barrier.
            lag = b.cmp_ge(b.readfirstlane(wave), b.const_i32(WAVES // 2))
            with b.scf_if(lag):
                if WIN_PRIO == 0:
                    b.s_setprio(1)
                b.s_barrier_bare()
        if causal and spec.causal_diag_split:
            # Tile j+1 needs the causal mask once it reaches the diagonal.
            diag_start = b.mul(qb, b.const_i32(n_per))
            if DIAG_TILES:
                diag_start = b.add(diag_start, b.const_i32(DIAG_TILES))
            body_hi = b.sub(diag_start, b.const_i32(1))
            body_hi = b.select(b.cmp_lt(body_hi, last), body_hi, last)
            body_hi = b.select(b.cmp_lt(body_hi, start_tile), start_tile, body_hi)
            body = la_loop(start_tile, body_hi, la_args, False, "lab")
            tail_args = [
                (name + "_t", val) for (name, _), val in zip(la_args, body.results)
            ]
            loop = la_loop(body_hi, last, tail_args, True, "lat")
        else:
            loop = la_loop(start_tile, last, la_args, causal, "lak")
        res = loop.results
        m_fin = res[0]
        l_i = res[1]
        o_acc = list(res[2 : 2 + D_TILES])
        s_last = [
            [b.vec_extract(res[2 + D_TILES + n], i) for i in range(16)]
            for n in range(N_SUB)
        ]
        neg_m = b.fneg(m_fin)
        p_last = [
            [_exp2(b.fma(s_last[n][i], qk_scale, neg_m)) for i in range(16)]
            for n in range(N_SUB)
        ]
        l_local = b.const_f32(0.0)
        for n in range(N_SUB):
            for i in range(16):
                l_local = b.fadd(l_local, p_last[n][i])
        if not spec.defer_row_sum:
            l_local = half_combine(l_local, b.fadd)
        l_i = b.fadd(l_i, l_local)
        if spec.defer_row_sum:
            l_i = half_combine(l_i, b.fadd)
        if PINGPONG:
            with b.scf_if(b.lnot(lag)):
                b.s_barrier_bare()
            if WIN_PRIO == 0:
                with b.scf_if(lag):
                    b.s_setprio(0)
            elif WIN_PRIO in (1, 2):
                b.s_setprio(0)
        # V(last) was issued at the top of the final iteration (or the prologue).
        b.s_waitcnt(vmcnt=0)
        b.s_barrier_bare()
        o_acc = do_pv(o_acc, relayout_p(p_last), b.mod(last, b.const_i32(NBUF)))
        return m_fin, l_i, o_acc

    def _emit_epilogue(m_fin, l_i, o_acc):
        """O = (P@V) / l, vectorized store (plus LSE when requested)."""
        if spec.emit_lse:
            positive_l = b.fcmp("ogt", l_i, b.const_f32(0.0))
            rcp_l = b.rcp(b.select(positive_l, l_i, b.const_f32(1.0)))
        else:
            rcp_l = b.rcp(l_i)
        if varlen:
            o_base = b.add(
                b.mul(q_seq0, b.const_i32(stride_q_tok)), b.mul(hq, b.const_i32(D))
            )
        else:
            o_base = b.add(
                b.mul(b.mul(bt, seqlen_q_p), b.const_i32(stride_q_tok)),
                b.mul(hq, b.const_i32(D)),
            )
        qtok = b.add(q_tok0, _mfma_32x32_c_col(b, lane, 0))
        q_row_byte = b.add(o_base, b.mul(qtok, b.const_i32(stride_q_tok)))
        d_half = b.mul(lane_h, b.const_i32(4))
        if spec.emit_lse:
            row_has_key = dense_row_has_key(
                b,
                qtok,
                seqlen_kv_b if varlen else seqlen_kv_p,
                causal=causal,
                diagonal=DIAG_OFF,
                window=W,
            )
            if varlen:
                lse_index = b.add(b.mul(b.add(q_seq0, qtok), b.const_i32(Hq)), hq)
                row_exists = b.cmp_lt(qtok, seqlen_q_b)
            else:
                lse_index = b.add(
                    b.mul(b.add(b.mul(bt, b.const_i32(Hq)), hq), seqlen_q_p),
                    qtok,
                )
                row_exists = b.cmp_lt(qtok, seqlen_q_p)
            _dense_lse_output(
                b,
                lse,
                lse_index,
                m_fin,
                l_i,
                row_exists,
                row_has_key,
                lane_h,
                sinks=sinks if use_sinks else None,
                head=hq,
                dtype=dtype,
            )
        # ragged: drop padded query rows (qtok >= seqlen_q) via a per-lane guard so
        # they never write (and never clobber a neighbouring batch's real rows). A
        # buffer store's OOB-drop only protects the last batch's overflow, so use an
        # explicit predicate that is correct for any batch.
        o_store_ctx = (
            b.scf_if(b.cmp_lt(qtok, b.const_i32(Sq))) if RAGGED else _nullcontext()
        )
        if spec.emit_lse and varlen:
            o_store_ctx = b.scf_if(row_exists)
        with o_store_ctx:
            _emit_o_store(
                b,
                o,
                q_row_byte,
                d_half,
                o_acc,
                rcp_l,
                dtype,
                D_TILES,
                spec.o_store_width,
                row_has_key=row_has_key if spec.emit_lse else None,
            )
        b.ret()

    if spec.qk_lookahead:
        _emit_epilogue(*_emit_lookahead_main())
        return b.kernel

    # Prologue: prime the K/V double buffer and compute the first (start) tile.
    load_tile(start_buf, start_tile)
    load_tile(start_buf1, b.add(start_tile, b.const_i32(1)))
    b.s_waitcnt(vmcnt=0)
    b.s_barrier_bare()
    # ragged non-causal needs the key-pad mask (ktok<seqlen_kv) on any tile that
    # can hold padded keys; causal drops them for free (see do_kbound_mask).
    RAG_KBOUND = RAGGED and (not causal) and (Skv % BN != 0)
    s0 = do_qk(start_buf)
    if causal and W > 0:
        do_mask(s0, start_tile, lower=True, upper=True)
    else:
        do_mask(s0, start_tile)
    if RAG_KBOUND:
        do_kbound_mask(s0, start_tile)

    if use_sinks:
        # Load sink value for this query head and convert to log2 domain
        sink_h = b.global_load(sinks, hq, dtype, align=2)
        sink_f = b.fmul(b.cast_to_f32(sink_h), rcp_ln2)
        m_init = sink_f
        l_init = one_f
    else:
        m_init = neg_inf

    m0, alpha0, _skip0 = softmax_max(s0, m_init)
    # tile-0 softmax exp + relayout only; PV lags by one tile (fused into the loop).
    neg_m0 = b.fneg(m0)  # same unbounded-score limitation; see _MAX_SCALE
    p0_vals = [
        [_exp2(b.fma(s0[nsub][i], qk_scale, neg_m0)) for i in range(16)]
        for nsub in range(N_SUB)
    ]
    l0_local = b.const_f32(0.0)
    for nsub in range(N_SUB):
        for i in range(16):
            l0_local = b.fadd(l0_local, p0_vals[nsub][i])
    if spec.defer_row_sum:
        l0 = l0_local
    else:
        l0 = half_combine(l0_local, b.fadd)
    if use_sinks:
        # Rescale l_init by alpha0: when m0 > m_init (sink), multiply by alpha0 to change
        # the sink's contribution from exp(sink - m_init) = 1.0 to exp(sink - m0).
        sink_l = b.fmul(l_init, alpha0)
        if spec.defer_row_sum:
            # Per-half partials: only half 0 carries the sink term.
            sink_l = b.select(
                b.cmp_eq(lane_h, b.const_i32(0)), sink_l, b.const_f32(0.0)
            )
        l0 = b.fadd(l0, sink_l)
    o0 = [b.zero_vec_f32(16) for _ in range(D_TILES)]
    pk0 = relayout_p(p0_vals)

    iter_args = (
        [("m", m0), ("l", l0)]
        + [(f"o{dt}", o0[dt]) for dt in range(D_TILES)]
        + [(f"pk{kk}", pk0[kk]) for kk in range(KK_STEPS)]
    )

    _rs_ctr = [0]

    def emit_loop_body(j, carry, mask_lower=False, mask_upper=False, mask_kbound=False):
        if IGLP != _IGLP_OFF:
            b.iglp_opt(IGLP)
        m_i = carry[0]
        l_i = carry[1]
        o_acc = list(carry[2 : 2 + D_TILES])
        p_prev = list(carry[2 + D_TILES : 2 + D_TILES + KK_STEPS])
        kbuf = b.mod(j, b.const_i32(NBUF))
        vbuf_prev = b.mod(b.add(j, b.const_i32(NBUF - 1)), b.const_i32(NBUF))
        pbuf = b.mod(b.add(j, b.const_i32(1)), b.const_i32(NBUF))

        # PF (partial-vmcnt prefetch): keep the freshest V(j) DMA in flight so it
        # overlaps compute instead of a full vmcnt(0) serialize (bit-identical).
        b.s_waitcnt(vmcnt=LOOP_VMCNT)
        b.s_barrier_bare()
        s = do_qk(kbuf)
        if mask_lower or mask_upper:
            do_mask(s, j, lower=mask_lower, upper=mask_upper)
        if mask_kbound:
            do_kbound_mask(s, j)
        m_new, alpha, skip = softmax_max(s, m_i)
        if PV_FENCE:
            b.sched_barrier(PV_FENCE_MASK)  # depth-1 fence: m_new region-live-in
        if PV_PRIORITY:
            b.s_setprio(PV_PRIORITY)  # PV-only s_setprio (paired with PF ~+3.5%)
        o_acc, p_vals, l_tile = pv_fused_exp(o_acc, p_prev, vbuf_prev, s, m_new)
        if PV_PRIORITY:
            b.s_setprio(0)
        if LAZY_RESCALE:
            _rs_ctr[0] += 1
            tg = _rs_ctr[0]
            trips = b.select(skip, b.const_i32(0), b.const_i32(1))
            rs_args = [(f"ro{dt}_{tg}", o_acc[dt]) for dt in range(D_TILES)]
            rs_args.append((f"rl_{tg}", l_i))
            rs = b.scf_for_iter(
                b.const_i32(0), trips, b.const_i32(1), rs_args, iv_name=f"rs{tg}"
            )
            with rs as (_iv, rc):
                o_sc = rescale_o(list(rc[:D_TILES]), alpha)
                b.scf_yield(*o_sc, b.fmul(rc[D_TILES], alpha))
            o_acc = list(rs.results[:D_TILES])
            l_new = b.fadd(rs.results[D_TILES], l_tile)
        else:
            l_new = b.fadd(b.fmul(l_i, alpha), l_tile)
            o_acc = rescale_o(o_acc, alpha)
        p_packs = relayout_p(p_vals)
        b.s_barrier_bare()
        load_tile(pbuf, b.add(j, b.const_i32(1)))
        b.scf_yield(m_new, l_new, *o_acc, *p_packs)

    if causal and W > 0:
        # Sliding-window three-phase band loop (prologue already did start_tile):
        #   L: [start+1, mid_lo)  window-edge tiles (masked)
        #   M: [mid_lo, mid_hi)   interior (mask-free: both bounds hold for all rows)
        #   R: [mid_hi, n_upper)  causal-edge tiles (masked)
        # Boundary phases apply BOTH bounds (robust for W<block_m overlap); the
        # redundant bound is a no-op compare. M is provably mask-free by geometry.
        diag_start = b.mul(qb, b.const_i32(n_per))
        a = b.add(start_tile, b.const_i32(1))
        left_end = b.add(diag_start, b.const_i32(n_per - Wt))  # start of mask-free M

        def _clamp(x, lo, hi):
            x = b.select(b.cmp_lt(x, lo), lo, x)  # max(x, lo)
            x = b.select(b.cmp_lt(x, hi), x, hi)  # min(x, hi)
            return x

        mid_lo = _clamp(left_end, a, n_upper)
        mid_hi = _clamp(diag_start, mid_lo, n_upper)

        phL = b.scf_for_iter(a, mid_lo, b.const_i32(1), iter_args, iv_name="swl")
        with phL as (j, carry):
            emit_loop_body(j, carry, mask_lower=True, mask_upper=True)
        mid_args = [
            (name + "_m", val) for (name, _), val in zip(iter_args, phL.results)
        ]
        phM = b.scf_for_iter(mid_lo, mid_hi, b.const_i32(1), mid_args, iv_name="swm")
        with phM as (j, carry):
            emit_loop_body(j, carry)
        rgt_args = [
            (name + "_r", val) for (name, _), val in zip(iter_args, phM.results)
        ]
        loop = b.scf_for_iter(mid_hi, n_upper, b.const_i32(1), rgt_args, iv_name="swr")
        with loop as (j, carry):
            emit_loop_body(j, carry, mask_lower=True, mask_upper=True)
    elif causal and spec.causal_diag_split:
        # Diagonal-only masking: below-diagonal tiles need no mask (~94% at Sq=8192).
        diag_start = b.mul(qb, b.const_i32(n_per))
        if DIAG_TILES:
            diag_start = b.add(diag_start, b.const_i32(DIAG_TILES))
        body_upper = b.select(b.cmp_lt(diag_start, n_upper), diag_start, n_upper)
        body = b.scf_for_iter(
            b.const_i32(1), body_upper, b.const_i32(1), iter_args, iv_name="nb"
        )
        with body as (j, carry):
            emit_loop_body(j, carry)
        tail_args = [
            (name + "_t", val) for (name, _), val in zip(iter_args, body.results)
        ]
        tail_lo = b.select(
            b.cmp_lt(diag_start, b.const_i32(1)), b.const_i32(1), diag_start
        )
        loop = b.scf_for_iter(tail_lo, n_upper, b.const_i32(1), tail_args, iv_name="nt")
        with loop as (j, carry):
            emit_loop_body(j, carry, mask_upper=True)
    else:
        loop = b.scf_for_iter(
            b.const_i32(1), n_upper, b.const_i32(1), iter_args, iv_name="nkt"
        )
        with loop as (j, carry):
            emit_loop_body(j, carry, mask_upper=causal, mask_kbound=RAG_KBOUND)

    res = loop.results
    l_i = res[1]
    if spec.defer_row_sum:
        l_i = half_combine(l_i, b.fadd)
    o_acc = list(res[2 : 2 + D_TILES])
    p_prev = list(res[2 + D_TILES : 2 + D_TILES + KK_STEPS])

    # PF: drain the last iter's in-flight V prefetch before the epilogue do_pv.
    b.s_waitcnt(vmcnt=0)
    b.s_barrier_bare()
    last_vbuf = b.mod(b.add(n_upper, b.const_i32(NBUF - 1)), b.const_i32(NBUF))
    o_acc = do_pv(o_acc, p_prev, last_vbuf)

    _emit_epilogue(res[0], l_i, o_acc)
    return b.kernel


def _emit_o_store(
    b, o, q_row_byte, d_half, o_acc, rcp_l, dtype, d_tiles, width, row_has_key=None
):
    """O = acc / l, ``width`` contiguous head-dim elements per global store.

    ``row_has_key`` (LSE path only) zeroes rows that see no key; ``None`` emits the
    unchanged store sequence.
    """
    if width == 8:
        # Each lane holds 4 contiguous head-dim values per 8-wide group; the
        # other 4 sit in lane ^ 32 (same query row). One v_permlane32_swap per
        # register pair hands the low half-wave all of group g and the high
        # half-wave all of group g+1, so every lane stores 8 contiguous values.
        lane_h = b.div(d_half, b.const_i32(4))
        for dt in range(d_tiles):
            for g in range(0, 4, 2):
                own_lo, own_hi, got_lo, got_hi = [], [], [], []
                for kk in range(4):
                    a = b.fmul(b.vec_extract(o_acc[dt], g * 4 + kk), rcp_l)
                    c = b.fmul(b.vec_extract(o_acc[dt], (g + 1) * 4 + kk), rcp_l)
                    lo, hi = b.permlane32_swap(b.bitcast(a, I32), b.bitcast(c, I32))
                    own_lo.append(a)
                    own_hi.append(c)
                    got_lo.append(b.bitcast(lo, F32))
                    got_hi.append(b.bitcast(hi, F32))
                first = b.cmp_eq(lane_h, b.const_i32(0))
                vals = [
                    b.cast_f32_to(b.select(first, x, y), dtype)
                    for x, y in zip(own_lo + got_hi, got_lo + own_hi)
                ]
                packed = b.vec_pack(vals, dtype)
                if row_has_key is not None:
                    packed = zero_keyless_row(b, packed, row_has_key, dtype, 8)
                addr = b.add(
                    q_row_byte,
                    b.add(b.const_i32(dt * 32 + g * 8), b.mul(lane_h, b.const_i32(8))),
                )
                b.global_store_vN(o, addr, packed, 8, align=16)
        return
    for dt in range(d_tiles):
        for g in range(4):
            for c in range(0, 4, width):
                d0 = b.add(b.const_i32(dt * 32 + g * 8 + c), d_half)
                addr = b.add(q_row_byte, d0)
                vals = [
                    b.cast_f32_to(
                        b.fmul(b.vec_extract(o_acc[dt], g * 4 + kk), rcp_l), dtype
                    )
                    for kk in range(c, c + width)
                ]
                packed = b.vec_pack(vals, dtype)
                if row_has_key is not None:
                    packed = zero_keyless_row(b, packed, row_has_key, dtype, width)
                b.global_store_vN(o, addr, packed, width, align=2 * width)


def _build_attention_dense_persistent(spec: AttentionDenseSpec) -> KernelDef:
    """Persistent (grid-stride) variant of the dense flash-attention kernel.

    Launches a 1-D grid of ``spec.num_persistent`` long-lived CTAs; each CTA
    grid-strides over the flattened work-item space ``W = (Sq//BLOCK_M)*Hq*B`` and
    runs the byte-identical inner step-1 CK-1 pipeline per work item, so the per-CTA
    launch/dispatch + scalar setup + K/V-prime cold-start is amortized once per CU
    instead of once per query-block (see the ``persistent`` spec field). Every
    algorithmic lever is the same always-on set as the default build; the only
    differences are the outer work loop, the qb-major work decode (load-balances the
    causal triangle), the per-work-item state reset, and a default
    ``exp_per_pv_step`` of 1 (keeps the extra loop-carried index math within 256
    VGPR at 0 spill; numerically identical to the grid body's 2 — pure emission
    ordering)."""
    B = spec.batch
    Sq = spec.seqlen_q
    Skv = spec.seqlen_kv
    Hq = spec.num_query_heads
    Hkv = spec.num_kv_heads
    D = spec.head_size
    causal = spec.causal
    dtype = spec.dtype_ir

    BLOCK_M = spec.block_m
    WAVES = spec.num_waves
    BN = spec.block_n
    NBUF = spec.lds_num_buffers
    PAD = spec.lds_k_row_pad
    NP = spec.num_persistent
    INTERLEAVE = spec.interleave

    K_STEPS = D // 16
    D_TILES = D // 32
    N_SUB = BN // 32
    KK_STEPS = BN // 16
    PV_STEPS = _pv_steps(spec.resolved_pv_loop_order(), D_TILES, KK_STEPS)
    EXP_PER = spec.resolved_exp_per_pv_step()
    SCHED_GROUP = spec.resolved_pv_sched_group_template()
    SCHED_GROUP_DS_READ = int(spec.pv_sched_group_ds_read)
    PV_FENCE = spec.resolved_pv_sched_fence()
    PV_FENCE_MASK = int(spec.pv_sched_fence_mask)
    PV_PRIORITY = int(spec.pv_priority)
    IGLP = spec.resolved_iglp_mode()
    LAZY_THRESHOLD = float(spec.lazy_rescale_threshold)
    gqa = Hq // Hkv
    stride_q_tok = Hq * D
    stride_k_tok = Hkv * D
    # DMA row packing (see default builder): 1 row/instr for D==128 (padded fast
    # path), else pack 128//D rows/instr into one contiguous LDS group, with K's
    # groups at a padded pitch and V's at the plain packed pitch.
    ROWS_PER_INSTR = 128 // D
    RAGGED = spec.ragged
    # ragged: ceil both the KV tiles and the query-block count so the partial
    # last block/tile is covered (padded rows/keys are handled on-chip).
    n_ktiles = ((Skv + BN - 1) // BN) if RAGGED else (Skv // BN)
    n_per = BLOCK_M // BN
    NQB = ((Sq + BLOCK_M - 1) // BLOCK_M) if RAGGED else (Sq // BLOCK_M)
    W = NQB * Hq * B  # total work items
    SW = spec.sliding_window  # sliding-window length (0 = disabled)
    SWt = SW // BN  # window length in KV tiles
    LAZY_RESCALE = spec.lazy_rescale
    use_sinks = spec.use_sinks
    WIDE_DMA = spec.wide_lds_dma

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = WAVES * 64
    b.kernel.attrs["waves_per_eu"] = int(spec.waves_per_eu)

    q = b.param(
        "q_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
    )
    k = b.param(
        "k_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
    )
    v = b.param(
        "v_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
    )
    o = b.param(
        "o_ptr", PtrType(dtype, "global"), noalias=True, writeonly=True, align=16
    )
    scale = b.param("scale", F32)
    if use_sinks:
        sinks = b.param(
            "sink_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
        )
    if spec.emit_lse:
        lse = b.param(
            "lse_ptr",
            PtrType(F32, "global"),
            noalias=True,
            writeonly=True,
            align=4,
        )
    qk_scale = b.fmul(scale, b.const_f32(LOG2E))
    _exp2 = b.exp2_fast if spec.use_exp2_fast else b.exp2

    # ----- CTA-invariant scalar setup (paid ONCE per persistent CTA) -----
    tid = b.thread_id_x()
    wave = b.div(tid, b.const_i32(64))
    lane = b.mod(tid, b.const_i32(64))
    lane_m = b.mod(lane, b.const_i32(32))
    lane_h = b.div(lane, b.const_i32(32))
    d_base = b.mul(lane_h, b.const_i32(8))
    neg_inf = b.const_f32(-1e30)
    if use_sinks:
        rcp_ln2 = b.const_f32(LOG2E)
        one_f = b.const_f32(1.0)

    # The wide gfx950 layout stores 8 rows x 64 columns in each 520-element
    # slab line. Two dwordx4 DMA instructions per wave fill the D=128 halves.
    if WIDE_DMA:
        K_GROUP = 1
        K_D_RPT = D // 64
        K_N_RPT = BN // 8
        K_LINE_STRIDE = 64 * 8 + PAD  # 520 half elements
        V_D_RPT = D // 64
        V_N_RPT = BN // 8
        V_LINE_STRIDE = 64 * 8 + spec.lds_v_row_pad
        WIDE_LINE_PASSES = 8 // WAVES
        assert 8 % WAVES == 0
        K_lds = b.smem_alloc(
            dtype,
            [NBUF, K_D_RPT, K_N_RPT, K_LINE_STRIDE],
            name_hint="Klds",
        )
        V_lds = b.smem_alloc(
            dtype,
            [NBUF, V_D_RPT, V_N_RPT, V_LINE_STRIDE],
            name_hint="Vlds",
        )
        k_lane_grp = None
        k_sub_col = None
    # 1 row/instr => per-row padded pitch (bank-conflict fix); packed D<128 =>
    # pad between DMA row-GROUPS on K, unpadded on V (see the default builder).
    elif ROWS_PER_INSTR == 1:
        K_GROUP = 1
        K_ROWS_LDS = BN
        LDROW = D + PAD
    else:
        K_GROUP = ROWS_PER_INSTR
        K_ROWS_LDS = BN // K_GROUP
        LDROW = K_GROUP * D + spec.lds_k_group_pad
    if not WIDE_DMA:
        K_lds = b.smem_alloc(dtype, [NBUF, K_ROWS_LDS, LDROW], name_hint="Klds")
        if K_GROUP > 1:
            k_lane_grp = b.div(lane_m, b.const_i32(K_GROUP))
            k_sub_col = b.mul(b.mod(lane_m, b.const_i32(K_GROUP)), b.const_i32(D))
        else:
            k_lane_grp = None
            k_sub_col = None
    if not WIDE_DMA:
        VROW = (D + spec.lds_v_row_pad) if ROWS_PER_INSTR == 1 else D
        V_lds = b.smem_alloc(dtype, [NBUF, BN, VROW], name_hint="Vlds")

    if WIDE_DMA:
        K_BYTES_PER_BUF = K_D_RPT * K_N_RPT * K_LINE_STRIDE * 2
        K_GROUP_BYTES = K_LINE_STRIDE * 2
    else:
        K_BYTES_PER_BUF = K_ROWS_LDS * LDROW * 2
        K_GROUP_BYTES = LDROW * 2
    if WIDE_DMA:
        V_BYTES_PER_BUF = V_D_RPT * V_N_RPT * V_LINE_STRIDE * 2
        V_GROUP_BYTES = V_LINE_STRIDE * 2
    else:
        V_BYTES_PER_BUF = BN * VROW * 2
        # Not a padded pitch -- see the default builder: the DMA writes contiguous
        # rows, so padding V needs a pad-aware transposed read, not a wider stride.
        V_GROUP_BYTES = ROWS_PER_INSTR * VROW * 2
    ROWS_PER_WAVE = BN // WAVES
    assert BN % K_GROUP == 0 and ROWS_PER_WAVE % ROWS_PER_INSTR == 0, (
        f"K row-group split must divide evenly: BN={BN} K_GROUP={K_GROUP} "
        f"ROWS_PER_WAVE={ROWS_PER_WAVE} ROWS_PER_INSTR={ROWS_PER_INSTR}"
    )
    V_DMA_PASSES = (BN * D) // (WAVES * 64 * 8)
    LOOP_VMCNT = V_DMA_PASSES if spec.partial_vmcnt_prefetch else 0
    zero_soff = b.const_i32(0)
    K_lds_addr = b.smem_addr_of(K_lds)
    V_lds_addr = b.smem_addr_of(V_lds)
    k_rsrc = b.buffer_rsrc(k, b.const_i32(B * Skv * Hkv * D * 2))
    v_rsrc = b.buffer_rsrc(v, b.const_i32(B * Skv * Hkv * D * 2))
    # ragged: bounds-checked Q load (OOB partial-block rows -> 0 register pad).
    q_rsrc = b.buffer_rsrc(q, b.const_i32(B * Sq * Hq * D * 2)) if RAGGED else None
    # Unused, but the lowering keeps dead ops, so dropping it moves every golden.
    b.zext(b.to_sgpr_u32(b.mul(wave, b.const_i32(64 * 16))), I64)

    # ----- persistent grid-stride loop over the flattened work-item space -----
    cta_id = b.block_id_x()
    outer = b.scf_for(cta_id, b.const_i32(W), b.const_i32(NP), iv_name="wi")
    with outer as wi:
        # Cross-work-item LDS reuse safety: drain the previous item's trailing DMA
        # and barrier so all waves finished the previous epilogue reads before we
        # reissue into the shared K/V buffers.
        b.s_waitcnt(vmcnt=0)
        b.s_barrier_bare()

        if spec.resolved_persist_decode == "gqa_pair_2phase":
            # NP=W/2 CTAs. gqa neighboring CTAs cover all local query heads
            # for one (qb_pair,hkv,bt); phase 0/1 selects complementary qbs.
            cta = b.mod(wi, b.const_i32(NP))
            phase = b.div(wi, b.const_i32(NP))
            hql = b.mod(cta, b.const_i32(gqa))
            rem = b.div(cta, b.const_i32(gqa))
            bt = b.mod(rem, b.const_i32(B))
            rem = b.div(rem, b.const_i32(B))
            hkv = b.mod(rem, b.const_i32(Hkv))
            qb_pair = b.div(rem, b.const_i32(Hkv))
            hq = b.add(b.mul(hkv, b.const_i32(gqa)), hql)
            qb = b.select(
                b.cmp_ne(phase, b.const_i32(0)),
                b.sub(b.const_i32(NQB - 1), qb_pair),
                qb_pair,
            )
        elif spec.resolved_persist_decode == "gqa_pair":
            # NP=NQB*Hkv*B CTAs. Two neighboring CTAs cover one
            # (qb_pair,hkv,bt) group; each handles half the local query heads
            # at both complementary qbs. The low/high costs sum to a constant.
            cta = b.mod(wi, b.const_i32(NP))
            phase = b.div(wi, b.const_i32(NP))
            pair_lane = b.mod(cta, b.const_i32(2))
            rem = b.div(cta, b.const_i32(2))
            bt = b.mod(rem, b.const_i32(B))
            rem = b.div(rem, b.const_i32(B))
            hkv = b.mod(rem, b.const_i32(Hkv))
            qb_pair = b.div(rem, b.const_i32(Hkv))
            half_gqa = gqa // 2
            high = b.cmp_ge(phase, b.const_i32(half_gqa))
            phase_half = b.mod(phase, b.const_i32(half_gqa))
            hql = b.add(b.mul(pair_lane, b.const_i32(half_gqa)), phase_half)
            hq = b.add(b.mul(hkv, b.const_i32(gqa)), hql)
            qb = b.select(
                high,
                b.sub(b.const_i32(NQB - 1), qb_pair),
                qb_pair,
            )
        elif spec.resolved_persist_decode == "hkv_major":
            # hkv-MAJOR + causal-balanced decode:
            #   wi = hkv*(NQB*gqa*B) + blk*(gqa*B) + hql*B + bt
            # * hkv in the MSB -> each grid-stride phase (NP consecutive wi) stays
            #   within ~1 kv-head, so the shared GQA K/V is L2-resident across its
            #   gqa query heads (recovers the non-persistent grid's locality:
            #   measured L2 hit 57% -> ~90%+ vs qb_major).
            # * `blk` (0..NQB-1) is folded to a query-block index that PAIRS a low
            #   and a high qb per CTA: blk<half -> qb=blk (cheap), blk>=half ->
            #   qb=NQB-1-(blk-half) (expensive), so a CTA that grid-strides over
            #   both halves of a kv-head does qb=X and qb=NQB-1-X (constant causal
            #   cost) -> keeps qb_major's load balance while gaining L2 locality.
            half = NQB // 2
            bt = b.mod(wi, b.const_i32(B))
            rem = b.div(wi, b.const_i32(B))  # hkv*(NQB*gqa) + blk*gqa + hql
            hql = b.mod(rem, b.const_i32(gqa))
            r2 = b.div(rem, b.const_i32(gqa))  # hkv*NQB + blk
            blk = b.mod(r2, b.const_i32(NQB))
            hkv = b.div(r2, b.const_i32(NQB))
            hq = b.add(b.mul(hkv, b.const_i32(gqa)), hql)
            # qb = blk<half ? blk : (NQB-1 - (blk-half))
            qb_hi = b.sub(b.const_i32(NQB - 1 + half), blk)  # NQB-1-(blk-half)
            qb = b.select(b.cmp_lt(blk, b.const_i32(half)), blk, qb_hi)
        else:
            # qb-MAJOR decode: wi = qb*(Hq*B) + hq*B + bt. Putting qb (the
            # triangular causal cost index) in the MSB spreads cheap+expensive
            # query blocks across each CTA under grid-stride; a qb-fast decode
            # would alias qb to a constant per CTA when NP is a multiple of NQB
            # (32x imbalance).
            bt = b.mod(wi, b.const_i32(B))
            rem = b.div(wi, b.const_i32(B))
            hq = b.mod(rem, b.const_i32(Hq))
            qb0 = b.div(rem, b.const_i32(Hq))
            if INTERLEAVE and causal and NQB > 1:
                # Reverse qb for ODD hq. Keying on rem (= qb0*Hq + hq) instead is
                # not a bijection for odd Hq: two qb0 alias onto one qb.
                odd = b.cmp_eq(b.mod(hq, b.const_i32(2)), b.const_i32(1))
                qb = b.select(odd, b.sub(b.const_i32(NQB - 1), qb0), qb0)
            else:
                qb = qb0
            hkv = b.div(hq, b.const_i32(gqa))

        q_tok0 = b.add(b.mul(qb, b.const_i32(BLOCK_M)), b.mul(wave, b.const_i32(32)))
        q_base = b.add(
            b.mul(b.mul(bt, b.const_i32(Sq)), b.const_i32(stride_q_tok)),
            b.mul(hq, b.const_i32(D)),
        )
        k_base = b.add(
            b.mul(b.mul(bt, b.const_i32(Skv)), b.const_i32(stride_k_tok)),
            b.mul(hkv, b.const_i32(D)),
        )

        q_tok = b.add(q_tok0, lane_m)
        q_packs = []
        for ks in range(K_STEPS):
            col = b.add(b.const_i32(ks * 16), d_base)
            addr = b.add(b.add(q_base, b.mul(q_tok, b.const_i32(stride_q_tok))), col)
            if RAGGED:
                raw = b.buffer_load_vN(
                    q_rsrc, b.mul(addr, b.const_i32(2)), b.const_i32(0), dtype, 8
                )
            else:
                raw = b.global_load_vN(q, addr, dtype, 8, align=16)
            q_packs.append(raw)

        def _async_load(rsrc, lds_base, buf_val, tile_key0, bytes_per_buf, group_bytes):
            """Async DMA one K/V tile (see default builder ``_async_load``).
            ROWS_PER_INSTR==1 (D==128): incremental one-instr-per-padded-row fast
            path (byte-identical), where a group IS a row.
            ROWS_PER_INSTR>1 (D<128): pack rows per instr into one contiguous
            group placed at ``group_bytes`` stride (K carries the inter-group pad,
            V's stride is the plain unpadded ROWS_PER_INSTR*D pitch)."""
            buf_off = b.mul(b.zext(buf_val, I64), b.const_i64(bytes_per_buf))
            if ROWS_PER_INSTR == 1:
                row0 = b.mul(wave, b.const_i32(ROWS_PER_WAVE))
                row_lds_off = b.add(
                    buf_off, b.zext(b.mul(row0, b.const_i32(group_bytes)), I64)
                )
                gcol = b.mul(lane, b.const_i32(2))
                voff = b.add(
                    b.add(
                        k_base,
                        b.mul(b.add(tile_key0, row0), b.const_i32(stride_k_tok)),
                    ),
                    gcol,
                )
                for r in range(ROWS_PER_WAVE):
                    row_base = b.smem_ptr_add(lds_base, row_lds_off)
                    b.async_buffer_load_lds_addr(
                        rsrc, row_base, b.mul(voff, b.const_i32(2)), zero_soff, 1
                    )
                    if r + 1 < ROWS_PER_WAVE:
                        row_lds_off = b.add(row_lds_off, b.const_i64(group_bytes))
                        voff = b.add(voff, b.const_i32(stride_k_tok))
            else:
                lanes_per_row = D // 2
                sub_row = b.div(lane, b.const_i32(lanes_per_row))
                col = b.mul(b.mod(lane, b.const_i32(lanes_per_row)), b.const_i32(2))
                groups_per_wave = ROWS_PER_WAVE // ROWS_PER_INSTR
                for it in range(groups_per_wave):
                    row0 = b.add(
                        b.mul(wave, b.const_i32(ROWS_PER_WAVE)),
                        b.const_i32(it * ROWS_PER_INSTR),
                    )
                    grp = b.add(
                        b.mul(wave, b.const_i32(groups_per_wave)), b.const_i32(it)
                    )
                    row_lds_off = b.add(
                        buf_off, b.zext(b.mul(grp, b.const_i32(group_bytes)), I64)
                    )
                    row_base = b.smem_ptr_add(lds_base, row_lds_off)
                    gkey = b.add(b.add(tile_key0, row0), sub_row)
                    voff = b.add(
                        b.add(k_base, b.mul(gkey, b.const_i32(stride_k_tok))), col
                    )
                    b.async_buffer_load_lds_addr(
                        rsrc, row_base, b.mul(voff, b.const_i32(2)), zero_soff, 1
                    )

        def _async_load_k_wide(lds_base, buf_val, tile_key0):
            """Two 128-bit-per-lane DMAs fill one slab-padded K tile per wave."""
            buf_off = b.mul(
                b.zext(buf_val, I64),
                b.const_i64(K_BYTES_PER_BUF),
            )
            n_in_wave = b.div(lane, b.const_i32(8))
            d_bucket = b.mod(lane, b.const_i32(8))
            for n_pass in range(WIDE_LINE_PASSES):
                line_id = b.add(wave, b.const_i32(n_pass * WAVES))
                krow = b.add(
                    tile_key0,
                    b.add(b.mul(n_in_wave, b.const_i32(8)), line_id),
                )
                src_base = b.add(
                    b.add(k_base, b.mul(krow, b.const_i32(stride_k_tok))),
                    b.mul(d_bucket, b.const_i32(8)),
                )
                for d_rpt in range(K_D_RPT):
                    line = b.add(
                        b.mul(line_id, b.const_i32(K_LINE_STRIDE * 2)),
                        b.const_i32(d_rpt * K_N_RPT * K_LINE_STRIDE * 2),
                    )
                    row_base = b.smem_ptr_add(
                        lds_base,
                        b.add(buf_off, b.zext(line, I64)),
                    )
                    voff = b.add(src_base, b.const_i32(d_rpt * 64))
                    b.async_buffer_load_lds_addr(
                        k_rsrc,
                        row_base,
                        b.mul(voff, b.const_i32(2)),
                        zero_soff,
                        4,
                    )

        def _async_load_v_wide(lds_base, buf_val, tile_key0):
            """Two 128-bit-per-lane DMAs fill one slab-padded V tile per wave."""
            buf_off = b.mul(
                b.zext(buf_val, I64),
                b.const_i64(V_BYTES_PER_BUF),
            )
            n_in_wave = b.div(lane, b.const_i32(8))
            d_bucket = b.mod(lane, b.const_i32(8))
            for n_pass in range(WIDE_LINE_PASSES):
                line_id = b.add(wave, b.const_i32(n_pass * WAVES))
                vrow = b.add(
                    tile_key0,
                    b.add(b.mul(n_in_wave, b.const_i32(8)), line_id),
                )
                src_base = b.add(
                    b.add(k_base, b.mul(vrow, b.const_i32(stride_k_tok))),
                    b.mul(d_bucket, b.const_i32(8)),
                )
                for d_rpt in range(V_D_RPT):
                    line = b.add(
                        b.mul(line_id, b.const_i32(V_LINE_STRIDE * 2)),
                        b.const_i32(d_rpt * V_N_RPT * V_LINE_STRIDE * 2),
                    )
                    row_base = b.smem_ptr_add(
                        lds_base,
                        b.add(buf_off, b.zext(line, I64)),
                    )
                    voff = b.add(src_base, b.const_i32(d_rpt * 64))
                    b.async_buffer_load_lds_addr(
                        v_rsrc,
                        row_base,
                        b.mul(voff, b.const_i32(2)),
                        zero_soff,
                        4,
                    )

        def async_load_k(lds_base, buf_val, tile_key0):
            if WIDE_DMA:
                _async_load_k_wide(lds_base, buf_val, tile_key0)
            else:
                _async_load(
                    k_rsrc,
                    lds_base,
                    buf_val,
                    tile_key0,
                    K_BYTES_PER_BUF,
                    K_GROUP_BYTES,
                )

        def async_load_v(lds_base, buf_val, tile_key0):
            if WIDE_DMA:
                _async_load_v_wide(lds_base, buf_val, tile_key0)
            else:
                _async_load(
                    v_rsrc,
                    lds_base,
                    buf_val,
                    tile_key0,
                    V_BYTES_PER_BUF,
                    V_GROUP_BYTES,
                )

        def load_tile(buf_val, tile_idx):
            tk0 = b.mul(tile_idx, b.const_i32(BN))
            async_load_k(K_lds_addr, buf_val, tk0)
            async_load_v(V_lds_addr, buf_val, tk0)

        def do_qk(kbuf):
            s_reg = []
            for nsub in range(N_SUB):
                acc = b.zero_vec_f32(16)
                if WIDE_DMA:
                    kline = b.mod(lane_m, b.const_i32(8))
                    kelem_base = b.add(
                        b.add(
                            b.mul(b.div(lane_m, b.const_i32(8)), b.const_i32(64)),
                            b.mul(lane_h, b.const_i32(8)),
                        ),
                        b.const_i32(nsub * 256),
                    )
                elif K_GROUP == 1:
                    krow = b.add(b.const_i32(nsub * 32), lane_m)
                else:
                    krow = b.add(b.const_i32(nsub * (32 // K_GROUP)), k_lane_grp)
                for ks in range(K_STEPS):
                    if WIDE_DMA:
                        kelem = b.add(
                            kelem_base,
                            b.const_i32((ks % 4) * 16),
                        )
                        k_pack = b.smem_load_vN(
                            K_lds,
                            kbuf,
                            b.const_i32(ks // 4),
                            kline,
                            kelem,
                            dtype=dtype,
                            n=8,
                        )
                    else:
                        col = b.add(b.const_i32(ks * 16), d_base)
                        if K_GROUP > 1:
                            col = b.add(k_sub_col, col)
                        k_pack = b.smem_load_vN(
                            K_lds, kbuf, krow, col, dtype=dtype, n=8
                        )
                    acc = mfma_32x32x16_for_dtype(b, dtype, k_pack, q_packs[ks], acc)
                s_reg.append(
                    [b.fmul(b.vec_extract(acc, i), qk_scale) for i in range(16)]
                )
            return s_reg

        def do_mask(s_reg, tile_idx, lower=False, upper=True):
            if not causal:
                return
            tile_key0 = b.mul(tile_idx, b.const_i32(BN))
            query_tok = b.add(q_tok0, _mfma_32x32_c_col(b, lane, 0))
            win_lo = b.sub(query_tok, b.const_i32(SW)) if lower else None
            for nsub in range(N_SUB):
                sub_base = b.add(tile_key0, b.const_i32(nsub * 32))
                for i in range(16):
                    ktok = b.add(sub_base, _mfma_32x32_c_row(b, lane, i))
                    if upper:
                        s_reg[nsub][i] = b.select(
                            b.cmp_le(ktok, query_tok), s_reg[nsub][i], neg_inf
                        )
                    if lower:
                        s_reg[nsub][i] = b.select(
                            b.cmp_gt(ktok, win_lo), s_reg[nsub][i], neg_inf
                        )

        def do_kbound_mask(s_reg, tile_idx):
            """ragged non-causal: -inf the padded keys (ktok >= seqlen_kv) of the
            partial last KV tile. Causal drops them for free."""
            tile_key0 = b.mul(tile_idx, b.const_i32(BN))
            for nsub in range(N_SUB):
                sub_base = b.add(tile_key0, b.const_i32(nsub * 32))
                for i in range(16):
                    ktok = b.add(sub_base, _mfma_32x32_c_row(b, lane, i))
                    s_reg[nsub][i] = b.select(
                        b.cmp_lt(ktok, b.const_i32(Skv)), s_reg[nsub][i], neg_inf
                    )

        RAG_KBOUND = RAGGED and (not causal) and (Skv % BN != 0)

        def softmax_max(s_reg, m_i):
            local_max = neg_inf
            for nsub in range(N_SUB):
                for i in range(16):
                    local_max = b.fmax(local_max, s_reg[nsub][i])
            tile_max = b.fmax(local_max, b.warp_shuffle_xor(local_max, 32))
            if LAZY_RESCALE:
                # Lazy max: only re-anchor when some lane's tile_max exceeds the
                # running max by > threshold; else keep m_i (skip the rescale).
                m_diff = b.fsub(tile_max, m_i)
                below_i32 = b.select(
                    b.fcmp("ole", m_diff, b.const_f32(LAZY_THRESHOLD)),
                    b.const_i32(1),
                    b.const_i32(0),
                )
                skip = b.cmp_ne(b.wave_all(below_i32), b.const_i32(0))
                m_new = b.select(skip, m_i, b.fmax(m_i, tile_max))
            else:
                skip = None
                m_new = b.fmax(m_i, tile_max)
            alpha = _exp2(b.fsub(m_i, m_new))
            return m_new, alpha, skip

        def softmax_stats(s_reg, m_i, l_i=None):
            m_new, alpha, _skip = softmax_max(s_reg, m_i)
            p = [
                [_exp2(b.fsub(s_reg[nsub][i], m_new)) for i in range(16)]
                for nsub in range(N_SUB)
            ]
            l_local = b.const_f32(0.0)
            for nsub in range(N_SUB):
                for i in range(16):
                    l_local = b.fadd(l_local, p[nsub][i])
            if spec.defer_row_sum:
                l_tile = l_local
            else:
                l_tile = b.fadd(l_local, b.warp_shuffle_xor(l_local, 32))
            if use_sinks:
                # Rescale l_i by alpha: when m_new > m_i, multiply by alpha to change
                # the sink's contribution from exp(sink - m_i) to exp(sink - m_new).
                sink_l = b.fmul(l_i, alpha)
                if spec.defer_row_sum:
                    # Per-half partials: only half 0 carries the sink term.
                    sink_l = b.select(
                        b.cmp_eq(lane_h, b.const_i32(0)), sink_l, b.const_f32(0.0)
                    )
                l_tile = b.fadd(l_tile, sink_l)
            return m_new, alpha, p, l_tile

        def relayout_p(p):
            packs = []
            for kk_step in range(KK_STEPS):
                elems = []
                for kk in range(8):
                    local_in_group = kk % 4
                    band = kk // 4
                    key_idx = kk_step * 16 + band * 8 + local_in_group
                    p_tile = key_idx // 32
                    row_static = key_idx % 32
                    preg = (row_static // 8) * 4 + (row_static % 4)
                    elems.append(b.cast_f32_to(p[p_tile][preg], dtype))
                packs.append(b.vec_pack(elems, dtype))
            return packs

        if WIDE_DMA:
            vline = b.add(
                b.mul(lane_h, b.const_i32(4)),
                b.div(b.mod(lane, b.const_i32(16)), b.const_i32(4)),
            )
            velem_lane = b.add(
                b.mul(
                    b.mod(b.div(lane, b.const_i32(16)), b.const_i32(2)),
                    b.const_i32(16),
                ),
                b.mul(b.mod(lane, b.const_i32(4)), b.const_i32(4)),
            )

        def read_v(dt, kk_step, vbuf):
            if WIDE_DMA:
                velem = b.add(
                    velem_lane,
                    b.const_i32((dt % 2) * 32 + kk_step * 128),
                )
                a0 = b.ds_read_tr16_b64(
                    V_lds,
                    vbuf,
                    b.const_i32(dt // 2),
                    vline,
                    velem,
                    dtype=dtype,
                )
                a1 = b.ds_read_tr16_b64(
                    V_lds,
                    vbuf,
                    b.const_i32(dt // 2),
                    vline,
                    b.add(velem, b.const_i32(64)),
                    dtype=dtype,
                )
                return b.vec_concat(a0, a1)
            return pv32_v_load_paired(
                b,
                V_lds=V_lds,
                v_buf=vbuf,
                n=dt,
                k=kk_step,
                lane_half32=lane_h,
                lane_col32=lane_m,
                dtype=dtype,
            )

        def do_pv(o_acc_in, p_packs, vbuf):
            out = list(o_acc_in)
            for dt, kk_step in PV_STEPS:
                out[dt] = mfma_32x32x16_for_dtype(
                    b, dtype, read_v(dt, kk_step, vbuf), p_packs[kk_step], out[dt]
                )
            return out

        def rescale_o(o_acc, alpha):
            return [
                b.vec_pack(
                    [b.fmul(b.vec_extract(o_acc[dt], i), alpha) for i in range(16)],
                    F32,
                )
                for dt in range(D_TILES)
            ]

        def pv_fused_exp(o_acc_in, p_packs, vbuf, s_reg, m_new):
            slots = [(nsub, i) for nsub in range(N_SUB) for i in range(16)]
            p_vals = [[None] * 16 for _ in range(N_SUB)]
            it = iter(slots)
            out = list(o_acc_in)
            for dt, kk_step in PV_STEPS:
                out[dt] = mfma_32x32x16_for_dtype(
                    b, dtype, read_v(dt, kk_step, vbuf), p_packs[kk_step], out[dt]
                )
                n_emit = 0
                for _ in range(EXP_PER):
                    slot = next(it, None)
                    if slot is None:
                        break
                    nsub, i = slot
                    p_vals[nsub][i] = _exp2(b.fsub(s_reg[nsub][i], m_new))
                    n_emit += 1
                if SCHED_GROUP:
                    b.sched_group_barrier(DS_READ, SCHED_GROUP_DS_READ, 0)
                    b.sched_group_barrier(MFMA, 1, 0)
                    b.sched_group_barrier(VALU, max(1, n_emit), 0)
                    b.sched_group_barrier(TRANS, max(1, n_emit), 0)
            for slot in it:
                nsub, i = slot
                p_vals[nsub][i] = _exp2(b.fsub(s_reg[nsub][i], m_new))
            l_local = b.const_f32(0.0)
            for nsub in range(N_SUB):
                for i in range(16):
                    l_local = b.fadd(l_local, p_vals[nsub][i])
            if spec.defer_row_sum:
                l_tile = l_local
            else:
                l_tile = b.fadd(l_local, b.warp_shuffle_xor(l_local, 32))
            return out, p_vals, l_tile

        _rs_ctr = [0]

        def emit_loop_body(
            j, carry, mask_lower=False, mask_upper=False, mask_kbound=False
        ):
            if IGLP != _IGLP_OFF:
                # IGLP owns post-RA placement (the wide-DMA default); manual
                # scheduling barriers are mutually exclusive with it.
                b.iglp_opt(IGLP)
            m_i = carry[0]
            l_i = carry[1]
            o_acc = list(carry[2 : 2 + D_TILES])
            p_prev = list(carry[2 + D_TILES : 2 + D_TILES + KK_STEPS])
            pbuf = b.mod(b.add(j, b.const_i32(1)), b.const_i32(NBUF))
            kbuf = b.mod(j, b.const_i32(NBUF))
            vbuf_prev = b.mod(b.add(j, b.const_i32(NBUF - 1)), b.const_i32(NBUF))

            # PF (partial-vmcnt prefetch): keep the freshest V(j) DMA in flight
            # (drain only V(j-1)+K(j), both older) so DMA overlaps compute instead
            # of a full vmcnt(0) serialize. Bit-identical, raises MfmaUtil.
            b.s_waitcnt(vmcnt=LOOP_VMCNT)
            b.s_barrier_bare()
            s = do_qk(kbuf)
            if mask_lower or mask_upper:
                do_mask(s, j, lower=mask_lower, upper=mask_upper)
            if mask_kbound:
                do_kbound_mask(s, j)
            m_new, alpha, skip = softmax_max(s, m_i)
            if PV_FENCE:
                b.sched_barrier(PV_FENCE_MASK)
            # PV-only s_setprio: the PV MFMA cluster wins issue slots; paired with
            # PF this converts to ~+3.5% (Sq=8192 causal, ~852 -> ~877 TFLOPS).
            if PV_PRIORITY:
                b.s_setprio(PV_PRIORITY)
            o_acc, p_vals, l_tile = pv_fused_exp(o_acc, p_prev, vbuf_prev, s, m_new)
            if PV_PRIORITY:
                b.s_setprio(0)
            if LAZY_RESCALE:
                # Skip the O/l rescale via a wave-uniform 0/1-trip loop when the
                # max didn't move (>threshold): 0 trips -> pass o_acc/l_i through
                # unscaled; 1 trip -> scale by alpha. Compiles to a scalar branch.
                # Unique names per emit (called once per KV-loop phase).
                _rs_ctr[0] += 1
                tg = _rs_ctr[0]
                trips = b.select(skip, b.const_i32(0), b.const_i32(1))
                rs_args = [(f"ro{dt}_{tg}", o_acc[dt]) for dt in range(D_TILES)]
                rs_args.append((f"rl_{tg}", l_i))
                rs = b.scf_for_iter(
                    b.const_i32(0), trips, b.const_i32(1), rs_args, iv_name=f"rs{tg}"
                )
                with rs as (_iv, rc):
                    o_sc = rescale_o(list(rc[:D_TILES]), alpha)
                    b.scf_yield(*o_sc, b.fmul(rc[D_TILES], alpha))
                o_acc = list(rs.results[:D_TILES])
                l_new = b.fadd(rs.results[D_TILES], l_tile)
            else:
                l_new = b.fadd(b.fmul(l_i, alpha), l_tile)
                o_acc = rescale_o(o_acc, alpha)
            p_packs = relayout_p(p_vals)
            b.s_barrier_bare()
            load_tile(pbuf, b.add(j, b.const_i32(1)))
            b.scf_yield(m_new, l_new, *o_acc, *p_packs)

        if causal:
            n_upper = b.add(b.mul(qb, b.const_i32(n_per)), b.const_i32(n_per))
            n_upper = b.select(
                b.cmp_lt(n_upper, b.const_i32(n_ktiles)),
                n_upper,
                b.const_i32(n_ktiles),
            )
        else:
            n_upper = b.const_i32(n_ktiles)

        # Sliding-window start tile (see default builder). SW==0 -> start_tile=0.
        if causal and SW > 0:
            _diag0 = b.mul(qb, b.const_i32(n_per))
            _lo_raw = b.sub(_diag0, b.const_i32(SWt))
            start_tile = b.select(
                b.cmp_gt(_lo_raw, b.const_i32(0)), _lo_raw, b.const_i32(0)
            )
        else:
            start_tile = b.const_i32(0)
        start_buf = b.mod(start_tile, b.const_i32(NBUF))
        start_buf1 = b.mod(b.add(start_tile, b.const_i32(1)), b.const_i32(NBUF))

        load_tile(start_buf, start_tile)
        load_tile(start_buf1, b.add(start_tile, b.const_i32(1)))
        b.s_waitcnt(vmcnt=0)
        b.s_barrier_bare()
        s0 = do_qk(start_buf)
        if causal and SW > 0:
            do_mask(s0, start_tile, lower=True, upper=True)
        else:
            do_mask(s0, start_tile)
        if RAG_KBOUND:
            do_kbound_mask(s0, start_tile)

        if use_sinks:
            # Load sink value for this query head and convert to log2 domain
            sink_h = b.global_load(sinks, hq, dtype, align=2)
            sink_f = b.fmul(b.cast_to_f32(sink_h), rcp_ln2)
            m_init = sink_f
            l_init = one_f
            m0, _alpha0, p0, l0 = softmax_stats(s0, m_init, l_init)
        else:
            m0, _alpha0, p0, l0 = softmax_stats(s0, neg_inf)

        o0 = [b.zero_vec_f32(16) for _ in range(D_TILES)]
        pk0 = relayout_p(p0)

        iter_args = (
            [("m", m0), ("l", l0)]
            + [(f"o{dt}", o0[dt]) for dt in range(D_TILES)]
            + [(f"pk{kk}", pk0[kk]) for kk in range(KK_STEPS)]
        )

        if causal and SW > 0:
            # Sliding-window three-phase band loop (prologue did start_tile).
            diag_start = b.mul(qb, b.const_i32(n_per))
            a = b.add(start_tile, b.const_i32(1))
            left_end = b.add(diag_start, b.const_i32(n_per - SWt))

            def _clamp(x, lo, hi):
                x = b.select(b.cmp_lt(x, lo), lo, x)
                x = b.select(b.cmp_lt(x, hi), x, hi)
                return x

            mid_lo = _clamp(left_end, a, n_upper)
            mid_hi = _clamp(diag_start, mid_lo, n_upper)

            phL = b.scf_for_iter(a, mid_lo, b.const_i32(1), iter_args, iv_name="swl")
            with phL as (j, carry):
                emit_loop_body(j, carry, mask_lower=True, mask_upper=True)
            mid_args = [
                (name + "_m", val) for (name, _), val in zip(iter_args, phL.results)
            ]
            phM = b.scf_for_iter(
                mid_lo, mid_hi, b.const_i32(1), mid_args, iv_name="swm"
            )
            with phM as (j, carry):
                emit_loop_body(j, carry)
            rgt_args = [
                (name + "_r", val) for (name, _), val in zip(iter_args, phM.results)
            ]
            loop = b.scf_for_iter(
                mid_hi, n_upper, b.const_i32(1), rgt_args, iv_name="swr"
            )
            with loop as (j, carry):
                emit_loop_body(j, carry, mask_lower=True, mask_upper=True)
        elif causal and spec.causal_diag_split:
            diag_start = b.mul(qb, b.const_i32(n_per))
            body_upper = b.select(b.cmp_lt(diag_start, n_upper), diag_start, n_upper)
            body = b.scf_for_iter(
                b.const_i32(1), body_upper, b.const_i32(1), iter_args, iv_name="nb"
            )
            with body as (j, carry):
                emit_loop_body(j, carry)
            tail_args = [
                (name + "_t", val) for (name, _), val in zip(iter_args, body.results)
            ]
            tail_lo = b.select(
                b.cmp_lt(diag_start, b.const_i32(1)), b.const_i32(1), diag_start
            )
            loop = b.scf_for_iter(
                tail_lo, n_upper, b.const_i32(1), tail_args, iv_name="nt"
            )
            with loop as (j, carry):
                emit_loop_body(j, carry, mask_upper=True)
        else:
            loop = b.scf_for_iter(
                b.const_i32(1), n_upper, b.const_i32(1), iter_args, iv_name="nkt"
            )
            with loop as (j, carry):
                emit_loop_body(j, carry, mask_upper=causal, mask_kbound=RAG_KBOUND)

        res = loop.results
        l_i = res[1]
        if spec.defer_row_sum:
            l_i = b.fadd(l_i, b.warp_shuffle_xor(l_i, 32))
        o_acc = list(res[2 : 2 + D_TILES])
        p_prev = list(res[2 + D_TILES : 2 + D_TILES + KK_STEPS])

        # PF: drain the last iter's in-flight V prefetch before the epilogue do_pv.
        b.s_waitcnt(vmcnt=0)
        b.s_barrier_bare()
        last_vbuf = b.mod(b.add(n_upper, b.const_i32(NBUF - 1)), b.const_i32(NBUF))
        o_acc = do_pv(o_acc, p_prev, last_vbuf)

        # Epilogue: recompute (bt, hq) from the live loop IV so they need not cross
        # the KV loop (keeps the loop-carried live set minimal -> 0 spill). Must
        # mirror the work-item decode used at the top of the loop.
        if spec.emit_lse:
            positive_l = b.fcmp("ogt", l_i, b.const_f32(0.0))
            rcp_l = b.rcp(b.select(positive_l, l_i, b.const_f32(1.0)))
        else:
            rcp_l = b.rcp(l_i)
        if spec.resolved_persist_decode == "gqa_pair_2phase":
            cta_e = b.mod(wi, b.const_i32(NP))
            hql_e = b.mod(cta_e, b.const_i32(gqa))
            rem_e = b.div(cta_e, b.const_i32(gqa))
            bt_e = b.mod(rem_e, b.const_i32(B))
            rem_e = b.div(rem_e, b.const_i32(B))
            hkv_e = b.mod(rem_e, b.const_i32(Hkv))
            hq_e = b.add(b.mul(hkv_e, b.const_i32(gqa)), hql_e)
        elif spec.resolved_persist_decode == "gqa_pair":
            cta_e = b.mod(wi, b.const_i32(NP))
            phase_e = b.div(wi, b.const_i32(NP))
            pair_lane_e = b.mod(cta_e, b.const_i32(2))
            rem_e = b.div(cta_e, b.const_i32(2))
            bt_e = b.mod(rem_e, b.const_i32(B))
            rem_e = b.div(rem_e, b.const_i32(B))
            hkv_e = b.mod(rem_e, b.const_i32(Hkv))
            phase_half_e = b.mod(phase_e, b.const_i32(gqa // 2))
            hql_e = b.add(
                b.mul(pair_lane_e, b.const_i32(gqa // 2)),
                phase_half_e,
            )
            hq_e = b.add(b.mul(hkv_e, b.const_i32(gqa)), hql_e)
        elif spec.resolved_persist_decode == "hkv_major":
            bt_e = b.mod(wi, b.const_i32(B))
            rem_e = b.div(wi, b.const_i32(B))
            hql_e = b.mod(rem_e, b.const_i32(gqa))
            hkv_e = b.div(b.div(rem_e, b.const_i32(gqa)), b.const_i32(NQB))
            hq_e = b.add(b.mul(hkv_e, b.const_i32(gqa)), hql_e)
        else:
            bt_e = b.mod(wi, b.const_i32(B))
            hq_e = b.mod(b.div(wi, b.const_i32(B)), b.const_i32(Hq))
        o_base = b.add(
            b.mul(b.mul(bt_e, b.const_i32(Sq)), b.const_i32(stride_q_tok)),
            b.mul(hq_e, b.const_i32(D)),
        )
        qtok = b.add(q_tok0, _mfma_32x32_c_col(b, lane, 0))
        q_row_byte = b.add(o_base, b.mul(qtok, b.const_i32(stride_q_tok)))
        d_half = b.mul(lane_h, b.const_i32(4))
        if spec.emit_lse:
            row_has_key = dense_row_has_key(
                b,
                qtok,
                b.const_i32(Skv),
                causal=causal,
                diagonal=0,
                window=SW,
            )
            lse_index = b.add(
                b.mul(
                    b.add(b.mul(bt_e, b.const_i32(Hq)), hq_e),
                    b.const_i32(Sq),
                ),
                qtok,
            )
            row_exists = b.cmp_lt(qtok, b.const_i32(Sq))
            _dense_lse_output(
                b,
                lse,
                lse_index,
                res[0],
                l_i,
                row_exists,
                row_has_key,
                lane_h,
                sinks=sinks if use_sinks else None,
                head=hq_e,
                dtype=dtype,
            )
        # ragged: guard padded query rows (qtok >= seqlen_q) so they never write.
        o_store_ctx = (
            b.scf_if(b.cmp_lt(qtok, b.const_i32(Sq))) if RAGGED else _nullcontext()
        )
        with o_store_ctx:
            _emit_o_store(
                b,
                o,
                q_row_byte,
                d_half,
                o_acc,
                rcp_l,
                dtype,
                D_TILES,
                spec.o_store_width,
                row_has_key=row_has_key if spec.emit_lse else None,
            )

    b.ret()
    return b.kernel


# --------------------------------------------------------------------------- #
# Public launch geometry + ABI (promoted from the prefill builder so the kernel
# is dispatchable / framework-callable without the host script).
# --------------------------------------------------------------------------- #


def attention_dense_grid(spec: AttentionDenseSpec) -> Tuple[int, int, int]:
    """Launch grid for ``spec``. Persistent = 1-D grid of ``num_persistent`` CTAs;
    default = one CTA per (query-block, query-head, batch)."""
    if spec.persistent:
        return (spec.num_persistent, 1, 1)
    nqb = (
        spec.seqlen_q + spec.block_m - 1
    ) // spec.block_m  # ceil: ragged partial block
    return (nqb, spec.num_query_heads, spec.batch)


def attention_dense_block(spec: AttentionDenseSpec) -> Tuple[int, int, int]:
    """CTA block dims: ``num_waves`` wave64s (= 512 threads)."""
    return (spec.num_waves * 64, 1, 1)


def _has_shape_params(spec: AttentionDenseSpec) -> bool:
    """Whether the body for ``spec`` takes batch/seqlen_q/seqlen_kv as kernargs.

    ``build_attention_dense`` declares them unconditionally;
    ``_build_attention_dense_persistent`` declares none, since its work-item space
    ``NP = spec.num_persistent`` is validated against ``nqb*Hkv*B`` at spec
    construction and is itself baked -- the shape cannot become a param there
    without ``num_persistent`` following it.

    This is the ABI question. ``spec.runtime_shape`` is the narrower cache-identity
    question (does the body bake the shape *anywhere*), and the two differ for
    paged / ragged / varlen / sliding-window / moving bottom-right causal.
    """
    return not spec.persistent


def attention_dense_signature(spec: AttentionDenseSpec):
    """ABI signature for :class:`KernelLauncher`. q/k/v/o pointers + f32 scale,
    plus optional sink_ptr when ``spec.use_sinks``, and the two ``cu_seqlens``
    i32 pointers when ``spec.varlen`` (see :func:`build_attention_dense`)."""
    from rocke.helpers.spec import SignatureBuilder

    sig = (
        SignatureBuilder()
        .ptr("q_ptr", spec.dtype)
        .ptr("k_ptr", spec.dtype)
        .ptr("v_ptr", spec.dtype)
        .ptr("o_ptr", spec.dtype)
        .scalar("scale", "f32")
    )
    if _has_shape_params(spec):
        # Mirrors the batch/seqlen_q/seqlen_kv params declared right after scale in
        # build_attention_dense.
        sig = (
            sig.scalar("batch", "i32")
            .scalar("seqlen_q", "i32")
            .scalar("seqlen_kv", "i32")
        )
    if spec.use_sinks:
        sig = sig.ptr("sink_ptr", spec.dtype)
    if spec.varlen:
        sig = sig.ptr("cu_seqlens_q", "i32").ptr("cu_seqlens_kv", "i32")
    if spec.paged:
        sig = (
            sig.ptr("block_tables", "i32")
            .ptr("kv_lens", "i32")
            .scalar("block_table_stride", "i32")
        )
    if spec.emit_lse:
        sig = sig.ptr("lse_ptr", "f32")
    return sig.build()


_DENSE_LAUNCHER_CACHE: dict = {}


def align_up(n: int, mult: int) -> int:
    """Round ``n`` up to the next multiple of ``mult`` (kernel tile alignment)."""
    return ((int(n) + mult - 1) // mult) * mult


def run_attention_dense_torch(
    *,
    spec: AttentionDenseSpec,
    q,
    k,
    v,
    out,
    scale: float,
    stream: int = 0,
    arch: str = "gfx950",
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    block_tables=None,
    kv_lens=None,
    sinks=None,
    validate_paged: bool = True,
    lse=None,
):
    """High-level framework entry: compile (cached) + launch the dense prefill
    kernel on torch tensors. ``q``/``k``/``v``/``out`` are dense contiguous
    tensors ([B, S, H, D] for q/out, [B, Skv, Hkv, D] for k/v); ``scale`` is the
    softmax scale (1/sqrt(D)) and is supported in ``[2**-64, 2**4]`` (the
    ordinary kernel takes the row max on unscaled scores and folds the scale
    into an fma; see ``_MAX_SCALE``). A NaN or infinite scale raises
    ``ValueError``; a finite scale outside the range raises
    ``NotImplementedError``. Returns ``out``.
    torch is imported lazily by the launcher — this module stays torch-free at
    import time.

    ``spec.emit_lse=True`` requires a caller-owned contiguous FP32 ``lse`` tensor
    on the same GPU, separate from every input and output. Its layout is
    [B, Hq, Sq, 1], or [total_query_tokens, Hq, 1] for packed varlen. The kernel
    writes natural-log softmax normalization while still returning ``out``.
    Fully masked real rows write O=0 and LSE=-inf; with a finite attention sink,
    those rows instead write the sink logit as LSE. Invalid padded rows do not
    write. Supplying ``lse`` with ``emit_lse=False`` is an error.
    The enabled varlen path checks cumulative offsets on the host (a device
    sync): Q lengths must be block_m multiples, KV lengths positive block_n
    multiples, and both must be within their spec maxima.

    Arbitrary (non-256-multiple) sequence lengths are served WITHOUT host
    padding by the in-kernel ragged path: build ``spec`` with ``ragged=True``
    and the TRUE (un-rounded) ``seqlen_q``/``seqlen_kv`` and pass the true-length
    q/k/v/out tensors. The kernel pads the boundary tiles on-chip (register-zero
    OOB query rows, LDS-zero OOB keys) and drops the partial O rows; the grid is
    ceil-sized automatically. See the ``ragged`` spec field.

    Varlen (``spec.varlen``): q/k/v/o use packed [total_tok, H, D] storage.
    Pass both int32 ``cu_seqlens`` [batch+1]. The nonpersistent ABI also takes
    batch/seqlen_q/seqlen_kv scalars; optional LSE is the final pointer.

    Paged (``spec.paged``): K/V are a PAGED CACHE, not dense tensors -- ``k``/``v``
    are ``[num_kv_blocks, block_size, Hkv, D]`` and are addressed through
    ``block_tables`` indirection. Pass ``block_tables`` (int32
    ``[num_seqs, max_blocks_per_seq]``) and ``kv_lens`` (int32 ``[num_seqs]``); a
    ``ValueError`` is raised if either is missing (or is supplied when
    ``spec.paged`` is False). ``q``/``out`` stay dense/contiguous. Single-sequence
    only in this revision (``batch == 1``); ``spec.block_size`` is the cache page
    size and ``spec.num_kv_blocks`` MUST equal ``k.shape[0]``. ``validate_paged``
    (default True) host-checks the paged CONTENTS (a device->host sync): the used
    ``block_tables`` entries lie in ``[0, num_kv_blocks)``, and each
    ``kv_lens[i] == seqlen_kv`` (the kernel visits all compile-time ``seqlen_kv``
    tiles, so a shorter ``kv_len`` reads page 0 for the uncovered tiles ->
    wrong output). Pass False on the hot / graph-captured path to skip the sync
    (block ids then rely on the bounds-checked cache SRD reading 0, and the
    ``kv_lens == seqlen_kv`` contract becomes the caller's responsibility).

    Sinks (``spec.use_sinks``): Attention sinks -- learned scalar
    logits that participate in the softmax denominator but have no value vector.
    Pass ``sinks`` (``spec.dtype`` ``[num_query_heads]``); a ``ValueError`` is
    raised if ``sinks`` is ``None`` when ``spec.use_sinks`` is True, or if
    ``sinks`` is provided when ``spec.use_sinks`` is False."""
    ok, why = supports_attention_dense(spec, arch=arch)
    if not ok:
        raise NotImplementedError(f"attention_dense unsupported for spec: {why}")
    if not math.isfinite(scale):
        raise ValueError(f"scale must be finite, got {scale!r}")
    if not _MIN_SCALE <= scale <= _MAX_SCALE:
        raise NotImplementedError(
            "NOT_YET_IMPLEMENTED: the gfx950 dense kernel supports a softmax "
            f"scale in [2**-64, 2**4], got {scale!r}"
        )
    if spec.varlen and (cu_seqlens_q is None or cu_seqlens_kv is None):
        raise ValueError(
            "varlen=True requires cu_seqlens_q and cu_seqlens_kv (int32 [batch+1]); "
            "packed attention cannot be launched without its offset pointers"
        )
    if not spec.varlen and (cu_seqlens_q is not None or cu_seqlens_kv is not None):
        raise ValueError("cu_seqlens_* provided but spec.varlen is False")
    if spec.paged and (block_tables is None or kv_lens is None):
        raise ValueError("paged=True requires block_tables and kv_lens")
    if not spec.paged and (block_tables is not None or kv_lens is not None):
        raise ValueError("block_tables/kv_lens provided but spec.paged is False")
    if spec.paged:
        # Paged K/V shape guard: the paged buffer-resource bound is sized from the
        # SPEC (num_kv_blocks*block_size*num_kv_heads*head_size -- see
        # ``_kv_cache_elems`` in build_attention_dense), NOT from the tensor.
        # If the passed cache is smaller than the spec claims, that bound
        # over-reaches the real allocation, so the hardware bounds-check no longer
        # guards it and a block-table entry can drive an out-of-bounds paged-cache
        # read. Validate the cache shape against the spec that sizes the bound,
        # before any compile/launch, so a mismatch fails loudly instead of reading
        # OOB. (block_tables/kv_lens presence is already checked above.)
        want = (
            spec.num_kv_blocks,
            spec.block_size,
            spec.num_kv_heads,
            spec.head_size,
        )
        for name, t in (("k", k), ("v", v)):
            got = tuple(t.shape)
            if got != want:
                raise ValueError(
                    f"paged {name} cache shape {got} != spec-derived "
                    f"[num_kv_blocks, block_size, num_kv_heads, head_size]={want}; "
                    "a mismatch mis-sizes the buffer-resource bound and can read OOB"
                )
        if validate_paged:
            # Physical block-id bounds (a CONTENTS check, unlike the metadata checks
            # above). An entry outside [0, num_kv_blocks) addresses a page outside the
            # cache; the bounds-checked SRD (see _async_load) drops it to 0 rather
            # than reading OOB, but that is silently WRONG output on a malformed
            # table -- so reject it loudly. This reads the tensors (a device->host
            # sync): pass validate_paged=False to skip on the hot/graph-captured path.
            # Only the entries the kernel dereferences are checked -- pages
            # [0, ceil(kv_len/block_size)) per seq; the rest are masked on device.
            _kvl = kv_lens.tolist() if hasattr(kv_lens, "tolist") else list(kv_lens)
            for _i in range(spec.batch):
                _kl = int(_kvl[_i])
                # Single-seq contract: the kernel visits ALL compile-time seqlen_kv
                # tiles, but the page-bounds mask uses the runtime kv_len -- so a
                # kv_len shorter than seqlen_kv leaves the uncovered tiles reading
                # page 0 (the masked block-table default) and folds them into the
                # softmax -> silently wrong output. Enforce the contract here.
                if _kl != spec.seqlen_kv:
                    raise ValueError(
                        f"paged kv_lens[{_i}]={_kl} != seqlen_kv={spec.seqlen_kv}; the "
                        "kernel reads all seqlen_kv tiles, so a shorter kv_len leaves "
                        "uncovered tiles reading page 0 -> silently wrong output"
                    )
                _npages = (_kl + spec.block_size - 1) // spec.block_size
                if _npages <= 0:
                    continue
                _used = block_tables[_i][:_npages]
                _used = _used.tolist() if hasattr(_used, "tolist") else list(_used)
                for _phys in _used:
                    _p = int(_phys)
                    if _p < 0 or _p >= spec.num_kv_blocks:
                        raise ValueError(
                            f"paged block_tables[{_i}] physical block id {_p} "
                            f"outside [0, num_kv_blocks={spec.num_kv_blocks}); a "
                            "malformed entry reads 0 via the bounds-checked cache "
                            "SRD -> silently wrong output"
                        )
    if not spec.use_sinks and sinks is not None:
        raise ValueError("sinks provided but spec.use_sinks is False")
    if spec.use_sinks:
        if sinks is None:
            raise ValueError("spec.use_sinks=True requires sinks that are not None")
        if sinks.shape != (spec.num_query_heads,):
            raise ValueError(
                f"sinks must have shape ({spec.num_query_heads},), got {tuple(sinks.shape)}"
            )
        if sinks.dtype != q.dtype:
            raise ValueError(f"sinks dtype {sinks.dtype} must match q dtype {q.dtype}")
        if not sinks.is_contiguous():
            raise ValueError("sinks must be contiguous")
        if not sinks.is_cuda:
            raise ValueError("sinks must be a CUDA tensor")

    validate_dense_lse(
        spec,
        q,
        k,
        v,
        out,
        lse,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        block_tables=block_tables,
        kv_lens=kv_lens,
        sinks=sinks,
    )

    from rocke.helpers.compile import compile_kernel
    from rocke.runtime import KernelLauncher, LaunchConfig

    # Cache identity is the spec minus the fields it declares in
    # runtime_param_fields, so every shape on the runtime path shares one
    # compiled binary. Every field the body still bakes participates, whether or
    # not kernel_name() remembered to append a token for it.
    key = attention_dense_cache_key(spec, arch=arch)
    launcher = _DENSE_LAUNCHER_CACHE.get(key)
    if launcher is None:
        art = compile_kernel(
            build_attention_dense(spec, arch=arch),
            arch=arch,
            backend="python",
            capture_ir_text=False,
        )
        launcher = KernelLauncher(
            hsaco=art.hsaco,
            kernel_name=art.kernel_name,
            signature=attention_dense_signature(spec),
        )
        _DENSE_LAUNCHER_CACHE[key] = launcher
    vals = {"q_ptr": q, "k_ptr": k, "v_ptr": v, "o_ptr": out, "scale": float(scale)}
    if _has_shape_params(spec):
        vals["batch"] = int(spec.batch)
        vals["seqlen_q"] = int(spec.seqlen_q)
        vals["seqlen_kv"] = int(spec.seqlen_kv)
    if spec.varlen:
        vals["cu_seqlens_q"] = cu_seqlens_q
        vals["cu_seqlens_kv"] = cu_seqlens_kv
    if spec.paged:
        vals["block_tables"] = block_tables
        vals["kv_lens"] = kv_lens
        vals["block_table_stride"] = int(block_tables.stride(0))
    if spec.use_sinks:
        vals["sink_ptr"] = sinks
    if spec.emit_lse:
        vals["lse_ptr"] = lse
    launcher(
        vals,
        config=LaunchConfig(
            grid=attention_dense_grid(spec),
            block=attention_dense_block(spec),
            stream=int(stream),
        ),
    )
    return out
