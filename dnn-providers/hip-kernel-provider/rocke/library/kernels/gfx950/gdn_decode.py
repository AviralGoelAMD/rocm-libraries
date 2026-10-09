# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""GDN/KDA single-token decode kernel instance builder.

For each active sequence and value head, advance one linear-attention decode
step over a fixed-size recurrent state ``S`` (a ``head_v_dim x head_k_dim``
key->value matrix), per the gated delta rule:

    q_hat = l2norm(q) * head_k_dim**-0.5
    k_hat = l2norm(k)
    log_decay_gdn    = -exp(A_log[h]) * softplus(a[h] + dt_bias[h])
    log_decay_kda[d] = lower_bound * sigmoid(exp(A_log[h]) * (a[h,d] + dt_bias[h,d]))
    decay = exp(log_decay)
    beta  = sigmoid(b[h])
    S     = S * decay
    v_new = (v - S @ k_hat) * beta
    out   = S @ q_hat + v_new * dot(k_hat, q_hat)
    S     = S + outer(v_new, k_hat)

GDN uses one scalar decay per value head. KDA uses one decay per K channel and
therefore scales the columns of ``S``. Only gate production and the fade differ;
the remaining recurrence and serving infrastructure are shared.

Only pages named by ``read_indices`` / ``write_indices`` are touched. The
contract is that ``-1`` is the ONLY valid skip sentinel and every active lane
owns a unique write index. Repeated read pages are allowed, and one sequence
may update the page it read. The host guard rejects any other negative
value before launch. This code is a lenient superset of that contract: it skips
on ``read < 0 or write < 0``, so on a path that bypasses the host guard
(``validate_indices=False``) a stray ``-7`` is silently treated as a skip rather
than caught. Callers should treat ``-1`` as the only sentinel; the wider test
here is defence, not licence.

Tensors are assumed contiguous (row-major), matching the packed
linear-attention decode contract:

    query, key   : [B, 1, num_k_heads, head_k_dim]   dtype
    value, out   : [B, 1, num_v_heads, head_v_dim]   dtype
    a (GDN)      : [B, 1, num_v_heads]               dtype
    a (KDA)      : [B, 1, num_v_heads, head_k_dim]   dtype
    b            : [B, 1, num_v_heads]               dtype
    dt_bias GDN  : [num_v_heads]                     dtype
    dt_bias KDA  : [num_v_heads, head_k_dim]         f32
    A_log        : [num_v_heads]                     f32
    read/write_indices : [B]                         i32
    state        : [pool, num_v_heads, head_v_dim, head_k_dim]  state_dtype (f16/bf16/f32)

**Two emitters, one contract**, selected by ``GdnDecodeSpec.simple``.

``simple=False`` -- the default, and the only path dispatch can reach -- is
**warp-tiled**: ``num_warps * wave_size`` threads per workgroup and
``blocks_per_v_dim`` workgroups per ``(sequence, value_head)``. Each warp splits
the ``head_k_dim`` reduction across ``warp_threads_k`` lanes and recombines with
an XOR butterfly (``quad_perm`` at offsets 1-2, ``ds_swizzle`` wider), so no LDS
is allocated. Dispatch selects this tile from the gfx950 GDN registry: `auto`
uses a deterministic static priority, while an explicit `spec_id` pins a tile.

``simple=True`` is the v1 reference: one workgroup per ``(sequence, value_head)``,
``head_v_dim`` threads, thread ``t`` owning state row ``t`` (the full
``head_k_dim``-wide key vector for value-dim ``t``) in registers, so every dot
product is thread-local and needs no cross-thread reduction. Q/K L2 norms and
``dot(k,q)`` are recomputed per thread -- redundant but simple -- which makes it
VGPR-heavy by construction. It is **not reachable through dispatch**; it exists
as the correctness baseline the warp-tiled path is validated against, and is
selected only by naming the spec directly (see ``ALGORITHM.md`` section 4.7).

Built for gfx950 (wave64) and placed alongside the KDA chunkwise kernel in
``kernels/gfx950/``; the ``arch`` argument is a validation/target hook, not a
portability claim -- a new arch adds its own tuned specs here rather than
importing across folders.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Tuple, get_args

from rocke.helpers.activations import LN2, LOG2E, SOFTPLUS_THRESHOLD
from rocke.core.ir import F32, I32, I64, IRBuilder, KernelDef, PtrType, TemporalHint
from rocke.helpers.io import (
    io_ir_type,
    load_scalar_as_f32,
    load_vec_as_f32,
    pack_f32_to,
    store_scalar_from_f32,
    store_vec,
)
from rocke.helpers.reduction import block_lds_reduce_with_wave_prologue, tree_reduce
from rocke.helpers.spec import SignatureBuilder, ceil_div_grid, kernel_name_join

__all__ = [
    "GdnDecodeSpec",
    "is_valid_spec",
    "build_gdn_decode",
    "gdn_decode_grid",
    "gdn_decode_signature",
    "GDN_DTYPES",
    "STATE_DTYPES",
    "STATE_HINTS",
]

DType = Literal["f16", "bf16"]
# The dtypes this kernel can emit, derived from the type rather than restated
# beside it. Dispatch RE-EXPORTS this: a copy drifts in the direction that
# fails silently -- a prefilter rejecting a shape the kernel has since learned
# to run, or admitting one it cannot.
GDN_DTYPES = get_args(DType)
# The recurrent state may also be f32: production KDA/GDN decode keeps the
# state in f32, and prefill hands it over in f32. Only the state widens; q/k/v,
# the gates and the output stay in ``dtype``. The kernel already does every
# state update in f32 registers, so f32 changes only the state loads/stores.
StateDType = Literal["f16", "bf16", "f32"]
STATE_DTYPES = get_args(StateDType)

NORM_EPS = 1e-6
EXP2_CLAMP = 126.0  # f32 exp2 argument range; keeps exp2_fast inside its contract
# State elements per vector access: 16 B in f16/bf16, 32 B in f32 (the backend
# splits that into two 16 B accesses).
STATE_VEC = 8
# State element size in bytes; is_valid_spec bars any state dtype but these.
_STATE_BYTES = {"f16": 2, "bf16": 2, "f32": 4}
# The ``state`` param promises 16 B alignment (one pool row starts at a
# multiple of 16 B); an f32 access must not claim its 32 B payload size.
_STATE_ALIGN = 16
# Cache policy of the recurrent-state loads and stores, one knob each. The state
# is read once and written once per call, so "streaming" (LLVM !nontemporal)
# is the natural choice, and it wins on tiles that own one state row per lane.
# It is not universal: on a tile whose lanes own several rows (the BPV=1 fused
# tile (4,16,1) holds 8), nontemporal stores send each 64 B line to HBM ~1.56x
# (gfx950, measured: 822 vs 528 MB written at Hk=Hv=32, batch 256), and default
# stores are faster. Dispatch picks per mode; see ``auto_state_hints``.
StateHint = Literal["streaming", "default"]
STATE_HINTS = get_args(StateHint)
_TEMPORAL = {"streaming": TemporalHint.STREAMING, "default": TemporalHint.DEFAULT}
# kernel_name() token for a non-default hint; "streaming" adds nothing so every
# name recorded before these knobs existed stays the same.
_HINT_TOKEN = {"default": "def"}


def _state_ir_type(state_dtype: str):
    return F32 if state_dtype == "f32" else io_ir_type(state_dtype)


def _load_state_f32(b: IRBuilder, ptr, idx, *, state_dtype: str, n: int, hint: str):
    """Load ``n`` state elements as f32 values with cache policy ``hint``."""
    if state_dtype == "f32":
        v = b.global_load_vN(
            ptr,
            idx,
            F32,
            n,
            align=_STATE_ALIGN,
            temporal_hint=_TEMPORAL[hint],
        )
        return [b.vec_extract(v, i) for i in range(n)]
    return load_vec_as_f32(
        b, ptr, idx, dtype=state_dtype, n=n, temporal_hint=_TEMPORAL[hint]
    )


def _pack_state(b: IRBuilder, values, *, state_dtype: str):
    """Pack f32 ``values`` into a state-dtype vector (no conversion for f32)."""
    if state_dtype == "f32":
        return b.vec_pack(values, F32)
    return pack_f32_to(b, values, dtype=state_dtype)


def _store_state(
    b: IRBuilder, ptr, idx, vec, *, state_dtype: str, n: int, hint: str
) -> None:
    """Store a packed state vector with cache policy ``hint``."""
    if state_dtype == "f32":
        b.global_store_vN(
            ptr,
            idx,
            vec,
            n,
            align=_STATE_ALIGN,
            temporal_hint=_TEMPORAL[hint],
        )
        return
    store_vec(b, ptr, idx, vec, n=n, temporal_hint=_TEMPORAL[hint])


@dataclass(frozen=True)
class GdnDecodeSpec:
    """One GDN single-token decode instance."""

    num_k_heads: int = 16
    num_v_heads: int = 32
    head_k_dim: int = 128
    head_v_dim: int = 128
    dtype: DType = "bf16"
    state_dtype: StateDType = "bf16"
    use_qk_l2norm: bool = True
    # Forget-gate granularity. "gdn" applies one scalar decay per head; "kda"
    # applies a per-channel DK-vector decay. GDN is the special case of KDA in
    # which every channel shares a value, so the general kernel serves both --
    # but only the general one can express the vector, which is why this is a
    # kernel field and not a dispatch detail.
    gate_kind: Literal["gdn", "kda"] = "gdn"
    # KDA gate lower bound: log-decay = lower_bound * sigmoid(...), so the gate
    # is bounded in (lower_bound, 0). Unread when gate_kind == "gdn", whose
    # softplus gate is unbounded below.
    lower_bound: float = -5.0
    # True: the kernel computes the decay from raw logits (the shipping path,
    # one launch). False: `a` carries a precomputed NATURAL-LOG-domain decay and
    # the kernel only exponentiates and multiplies. The False mode is never
    # dispatched; it exists so this kernel can be timed against a competitor
    # recurrence-only kernel at an identical work boundary.
    fuse_gate: bool = True
    wave_size: int = 64
    # GDN's dispatcher default uses these values whenever they are legal.
    # Direct callers still get a valid general-purpose configuration.
    num_warps: int = 2
    warp_threads_k: int = 16
    blocks_per_v_dim: int = (
        8  # split a head's V-dim across this many CTAs (small-B fill)
    )
    simple: bool = False  # True => v1 one-thread-per-row reference path
    # Optional fusions of a hybrid layer's neighbours (ALGORITHM.md §4.9).
    # Both need one workgroup per head (blocks_per_v_dim == 1); fuse_conv
    # also needs Hk == Hv, since shared q/k conv channels would race.
    fuse_conv: bool = False  # width-4 causal conv1d + SiLU on q/k/v, in-place tap shift
    fuse_out_norm: bool = (
        False  # o * rsqrt(mean(o^2)+eps) * norm_weight * sigmoid(gate)
    )
    # Cache policy of the state loads / stores ("streaming" = !nontemporal).
    state_load_hint: StateHint = "streaming"
    state_store_hint: StateHint = "streaming"
    name: str = "rocke_gdn_decode"

    @property
    def block_size(self) -> int:
        return self.head_v_dim if self.simple else self.num_warps * self.wave_size

    @property
    def v_per_k_head(self) -> int:
        return self.num_v_heads // self.num_k_heads

    def kernel_name(self) -> str:
        # Every field that changes emitted code MUST appear here: this name is the
        # compile/launcher cache key, so two specs sharing a name means one of them
        # silently runs the other's kernel. Fields whose value equals the default
        # are folded into the deviation-only suffixes below to keep names stable.
        parts = (
            self.dtype,
            f"kh{self.num_k_heads}",
            f"vh{self.num_v_heads}",
            f"dk{self.head_k_dim}",
            f"dv{self.head_v_dim}",
            f"w{self.num_warps}k{self.warp_threads_k}b{self.blocks_per_v_dim}",
        )
        if self.state_dtype != self.dtype:
            parts += (f"st{self.state_dtype}",)
        # Deviation-only, so the KDA gate kind is additive: a default-gate spec
        # keeps the exact name it had before this field existed, and every
        # pinned GDN golden hash stays valid. lower_bound is nested because the
        # GDN gate never reads it -- letting it reach the name there would give
        # two names to two byte-identical kernels.
        if self.gate_kind != "gdn":
            parts += (self.gate_kind,)
            if self.fuse_gate:
                if self.lower_bound != -5.0:
                    parts += (f"lb{self.lower_bound:g}",)
            else:
                parts += ("nofg",)
        if self.wave_size != 64:
            parts += (f"ws{self.wave_size}",)
        # Deviation-only: a streaming hint adds nothing, so names (and every
        # golden hash) recorded before these knobs existed are unchanged.
        if self.state_load_hint != "streaming":
            parts += (
                f"lh{_HINT_TOKEN.get(self.state_load_hint, self.state_load_hint)}",
            )
        if self.state_store_hint != "streaming":
            parts += (
                f"sh{_HINT_TOKEN.get(self.state_store_hint, self.state_store_hint)}",
            )
        return kernel_name_join(
            self.name,
            *parts,
            flags={
                "l2": self.use_qk_l2norm,
                "s": self.simple,
                "cv": self.fuse_conv,
                "rn": self.fuse_out_norm,
            },
        )


def is_valid_spec(spec: GdnDecodeSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    """Reject impossible/unsupported GDN decode configs before IR is built."""
    from rocke.core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)

    if spec.wave_size != target.wave_size:
        # The wave_size rules further down check INTERNAL consistency
        # (wave_size % warp_threads_k). This one checks agreement with the
        # hardware: the lane layout and the xor butterfly both take the wave
        # width as given, so a wave32 target built with a wave64 spec emits IR
        # whose cross-lane arithmetic is simply wrong -- and silently, since
        # nothing downstream re-derives it.
        return (
            False,
            f"spec.wave_size {spec.wave_size} != {arch} wave size "
            f"{target.wave_size}",
        )
    if spec.dtype not in GDN_DTYPES or spec.state_dtype not in STATE_DTYPES:
        return False, f"unsupported dtype {spec.dtype}/{spec.state_dtype}"
    if spec.gate_kind not in ("gdn", "kda"):
        return False, f"gate_kind must be 'gdn' or 'kda' (got {spec.gate_kind!r})"
    if spec.gate_kind == "gdn" and not spec.fuse_gate:
        return False, "gate_kind='gdn' requires fuse_gate=True"
    if (
        spec.state_load_hint not in STATE_HINTS
        or spec.state_store_hint not in STATE_HINTS
    ):
        return False, (
            f"state_load_hint / state_store_hint must be one of {STATE_HINTS}, got "
            f"{spec.state_load_hint!r} / {spec.state_store_hint!r}"
        )
    if spec.fuse_conv or spec.fuse_out_norm:
        if spec.simple:
            return False, (
                "fuse_conv / fuse_out_norm require the warp-tiled path (simple=False)"
            )
        if spec.blocks_per_v_dim != 1:
            return False, (
                "fuse_conv / fuse_out_norm require blocks_per_v_dim == 1 "
                f"(one workgroup per head), got {spec.blocks_per_v_dim}"
            )
    if spec.fuse_conv and spec.num_k_heads != spec.num_v_heads:
        return False, (
            "fuse_conv requires num_k_heads == num_v_heads (shared q/k conv "
            f"channels would race), got {spec.num_k_heads}/{spec.num_v_heads}"
        )
    if (
        spec.gate_kind == "kda"
        and spec.fuse_gate
        and (not math.isfinite(spec.lower_bound) or spec.lower_bound >= 0.0)
    ):
        return False, (
            "lower_bound must be finite negative for the fused KDA gate, "
            f"got {spec.lower_bound}"
        )
    for _field, _value in (
        ("num_k_heads", spec.num_k_heads),
        ("num_v_heads", spec.num_v_heads),
        ("head_k_dim", spec.head_k_dim),
        ("head_v_dim", spec.head_v_dim),
    ):
        if _value <= 0:
            return False, f"{_field} must be positive, got {_value}"
    if spec.num_v_heads % spec.num_k_heads:
        return False, "num_v_heads must be divisible by num_k_heads"
    if spec.head_k_dim % STATE_VEC or spec.head_v_dim % STATE_VEC:
        return False, f"head dims must be multiples of {STATE_VEC}"
    if spec.block_size > target.max_threads_per_block:
        return False, (
            f"block_size {spec.block_size} > max_threads_per_block "
            f"{target.max_threads_per_block} on {arch}"
        )
    if not spec.simple:
        for _field, _value in (
            ("num_warps", spec.num_warps),
            ("warp_threads_k", spec.warp_threads_k),
            ("blocks_per_v_dim", spec.blocks_per_v_dim),
            ("wave_size", spec.wave_size),
        ):
            if _value <= 0:
                return False, f"{_field} must be positive, got {_value}"
        if spec.wave_size % spec.warp_threads_k:
            return False, "wave_size must be divisible by warp_threads_k"
        vpt = STATE_VEC
        warp_tile_k = spec.warp_threads_k * vpt
        wgroup_v = spec.num_warps * (spec.wave_size // spec.warp_threads_k)
        if spec.head_k_dim % warp_tile_k:
            return False, f"head_k_dim must be a multiple of {warp_tile_k}"
        if spec.head_v_dim % spec.blocks_per_v_dim:
            return False, "head_v_dim must be a multiple of blocks_per_v_dim"
        tile_v = spec.head_v_dim // spec.blocks_per_v_dim
        if tile_v % wgroup_v:
            return (
                False,
                f"head_v_dim/blocks_per_v_dim must be a multiple of {wgroup_v}",
            )
    return True, ""


def build_gdn_decode(spec: GdnDecodeSpec, arch: str = "gfx950") -> KernelDef:
    """Build the IR for one GDN single-token decode instance (dispatch)."""
    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid gdn_decode spec for {arch}: {why}")
    return _build_simple(spec) if spec.simple else _build_warp_tiled(spec)


def _build_simple(spec: GdnDecodeSpec) -> KernelDef:
    """v1 reference: one thread per value-dim (state row); no cross-thread reduce."""

    HK, HV = spec.num_k_heads, spec.num_v_heads
    DK, DV = spec.head_k_dim, spec.head_v_dim
    G = spec.v_per_k_head
    BS = spec.block_size
    scale = 1.0 / math.sqrt(DK)

    # Contiguous row-major strides (element counts).
    Q_HN, Q_HK = HK * DK, DK  # query/key: [B,1,HK,DK]
    V_HN, V_HK = HV * DV, DV  # value/out: [B,1,HV,DV]
    S_POOL, S_HV, S_VR = HV * DV * DK, DV * DK, DK  # state: [pool,HV,DV,DK]
    ST_BYTES = _STATE_BYTES[spec.state_dtype]

    io_ty = io_ir_type(spec.dtype)
    st_ty = _state_ir_type(spec.state_dtype)

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = BS

    Q = b.param(
        "query", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16
    )
    K = b.param("key", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16)
    Vv = b.param(
        "value", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16
    )
    Ag = b.param("a", PtrType(io_ty, "global"), noalias=True, readonly=True)
    Bg = b.param("b", PtrType(io_ty, "global"), noalias=True, readonly=True)
    # f32 in KDA mode: the per-channel gate is evaluated in f32 and KDA prefill
    # declares the same tensor f32, so the two families share one contract.
    DTB = b.param(
        "dt_bias",
        PtrType(F32 if spec.gate_kind == "kda" else io_ty, "global"),
        noalias=True,
        readonly=True,
    )
    ALOG = b.param("A_log", PtrType(F32, "global"), noalias=True, readonly=True)
    RIDX = b.param("read_indices", PtrType(I32, "global"), noalias=True, readonly=True)
    WIDX = b.param("write_indices", PtrType(I32, "global"), noalias=True, readonly=True)
    STATE = b.param("state", PtrType(st_ty, "global"), noalias=True, align=16)
    OUT = b.param(
        "out", PtrType(io_ty, "global"), noalias=True, writeonly=True, align=16
    )
    _ = b.param("batch_size", I32)  # noqa: F841

    tid = b.thread_id_x()  # value-dim row this thread owns (0..DV-1)
    bidx = b.block_id_x()
    b_i = b.div(bidx, b.const_i32(HV))
    hv_i = b.mod(bidx, b.const_i32(HV))
    hk_i = b.div(hv_i, b.const_i32(G))

    read_pool = b.global_load_i32(RIDX, b_i)
    write_pool = b.global_load_i32(WIDX, b_i)

    # Skip continuous-batching padding lanes (negative sentinel index).
    active = b.land(
        b.cmp_ge(read_pool, b.const_i32(0)), b.cmp_ge(write_pool, b.const_i32(0))
    )
    with b.scf_if(active):
        # ---- exp/sigmoid/softplus composed from exp2/log2/rcp (no native) ----
        # exp2_fast emits no range guard, so its contract is that the *caller*
        # bounds the argument. Every other exp2_fast in this repo is a softmax,
        # whose argument is <= 0 by construction; a gate argument is not -- a,
        # dt_bias, A_log and b come straight from caller tensors and nothing
        # bounds them. We meet the contract explicitly instead, with the same
        # fmin/fmax clamp the KDA emitter uses (``ex2``, kda_chunkwise.py).
        #
        # The two lowerings differ only in the f32 denormal window: unclamped
        # exp2_fast and the guarded b.exp2 agree on the whole positive range
        # (both saturate to +inf past ~88), and where they part, exp2_fast
        # flushes to 0 while exp2 returns the denormal. A denormal cannot
        # survive rounding into a bf16 state, so this clamp buys contract
        # compliance rather than accuracy -- which is why it is a 2-op clamp
        # and not the ~5-op guarded lowering.
        def exp_f32(x):
            arg = b.fmul(x, b.const_f32(LOG2E))
            arg = b.fmin(b.fmax(arg, b.const_f32(-EXP2_CLAMP)), b.const_f32(EXP2_CLAMP))
            return b.exp2_fast(arg)

        def log1p_f32(x):
            return b.fmul(b.log2(b.fadd(b.const_f32(1.0), x)), b.const_f32(LN2))

        # ---- load q,k rows [DK] -> f32 registers ----
        q_base = b.add(b.mul(b_i, b.const_i32(Q_HN)), b.mul(hk_i, b.const_i32(Q_HK)))
        qv, kv = [], []
        for c in range(0, DK, STATE_VEC):
            off = b.add(q_base, b.const_i32(c))
            qv += load_vec_as_f32(b, Q, off, dtype=spec.dtype, n=STATE_VEC)
            kv += load_vec_as_f32(b, K, off, dtype=spec.dtype, n=STATE_VEC)

        # ---- L2 normalize q (and *scale), k ----
        if spec.use_qk_l2norm:
            sum_q2 = tree_reduce(b, b.fadd, [b.fmul(x, x) for x in qv])
            sum_k2 = tree_reduce(b, b.fadd, [b.fmul(x, x) for x in kv])
            inv_q = b.rsqrt(b.fadd(sum_q2, b.const_f32(NORM_EPS)))
            inv_k = b.rsqrt(b.fadd(sum_k2, b.const_f32(NORM_EPS)))
            sq = b.fmul(inv_q, b.const_f32(scale))
            qn = [b.fmul(x, sq) for x in qv]
            kn = [b.fmul(x, inv_k) for x in kv]
        else:
            qn = [b.fmul(x, b.const_f32(scale)) for x in qv]
            kn = kv

        # ---- gates (per value head) ----
        # The GDN arm below is the original emission, verbatim and in its
        # original order -- moved into a branch, not rewritten. Order matters:
        # these calls append ops to the IR, so hoisting even a shared load out
        # of the arm would reorder GDN's instructions and move every golden
        # hash pinned against it. Duplicating two loads across the arms is the
        # cheap side of that trade.
        a_idx = b.add(b.mul(b_i, b.const_i32(HV)), hv_i)

        if spec.gate_kind == "gdn":
            ra = load_scalar_as_f32(b, Ag, a_idx, dtype=spec.dtype)
            rb = load_scalar_as_f32(b, Bg, a_idx, dtype=spec.dtype)
            rdt = load_scalar_as_f32(b, DTB, hv_i, dtype=spec.dtype)
            ral = b.global_load_f32(ALOG, hv_i)  # A_log is fp32

            x = b.fadd(ra, rdt)
            sp = b.select(
                b.fcmp("ogt", x, b.const_f32(SOFTPLUS_THRESHOLD)),
                x,
                log1p_f32(exp_f32(x)),
            )
            decay = exp_f32(b.fneg(b.fmul(exp_f32(ral), sp)))
            beta = b.rcp_fast(b.fadd(b.const_f32(1.0), exp_f32(b.fneg(rb))))
        else:
            # KDA: one decay per K channel.
            #   log_decay[d] = lower_bound * sigmoid(exp(A_log[h]) * (g[d] + dt_bias[h,d]))
            # exp(A_log[h]) is per head, so it is hoisted out of the channel loop.
            # This thread owns a whole state row, so it needs the full DK extent.
            rb = load_scalar_as_f32(b, Bg, a_idx, dtype=spec.dtype)
            ral = b.global_load_f32(ALOG, hv_i)  # per head in both gate kinds
            beta = b.rcp_fast(b.fadd(b.const_f32(1.0), exp_f32(b.fneg(rb))))
            exp_alog = exp_f32(ral)
            g_base = b.add(
                b.mul(b_i, b.const_i32(HV * DK)), b.mul(hv_i, b.const_i32(DK))
            )
            dtb_base = b.mul(hv_i, b.const_i32(DK))
            decay = []
            for c in range(0, DK, STATE_VEC):
                gv = load_vec_as_f32(
                    b, Ag, b.add(g_base, b.const_i32(c)), dtype=spec.dtype, n=STATE_VEC
                )
                # dt_bias is f32, so load_vec_as_f32 (a 16-bit ingest helper)
                # does not apply; global_load_vN lowers STATE_VEC f32 values to
                # one vector load rather than STATE_VEC scalar ones.
                dtvec = b.global_load_vN(
                    DTB, b.add(dtb_base, b.const_i32(c)), F32, STATE_VEC
                )
                dtv = [b.vec_extract(dtvec, j) for j in range(STATE_VEC)]
                for j in range(STATE_VEC):
                    if spec.fuse_gate:
                        inner = b.fmul(exp_alog, b.fadd(gv[j], dtv[j]))
                        sig = b.rcp_fast(
                            b.fadd(b.const_f32(1.0), exp_f32(b.fneg(inner)))
                        )
                        log_decay = b.fmul(b.const_f32(spec.lower_bound), sig)
                    else:
                        # `a` already carries natural-log-domain decay; only the
                        # exponential remains.
                        log_decay = gv[j]
                    decay.append(exp_f32(log_decay))

        # ---- dot(k_hat, q_hat) (scalar, redundant per thread) ----
        dot_kq = tree_reduce(b, b.fadd, [b.fmul(kn[j], qn[j]) for j in range(DK)])

        # ---- load state row t=tid: state[read_pool, hv, tid, 0:DK] ----
        # The pool base (read_pool * S_POOL) overflows i32 once the pool holds
        # >=4096 slots, so advance the pointer by a 64-bit byte offset and keep
        # the in-slot index (< S_POOL) in i32.
        #
        # The state is read once and written once per call, so its loads and
        # stores are streamed (nontemporal): caching it gains nothing. This is
        # always on by design, not a tuning knob; q/k/v, the gates and the
        # output keep the default cache policy.
        state_r = b.global_ptr_add(
            STATE, b.mul(b.sext(read_pool, I64), b.const_i64(S_POOL * ST_BYTES))
        )
        rs_base = b.add(b.mul(hv_i, b.const_i32(S_HV)), b.mul(tid, b.const_i32(S_VR)))
        sv = []
        for c in range(0, DK, STATE_VEC):
            off = b.add(rs_base, b.const_i32(c))
            sv += _load_state_f32(
                b,
                state_r,
                off,
                state_dtype=spec.state_dtype,
                n=STATE_VEC,
                hint=spec.state_load_hint,
            )
        # Gated forget. A scalar decay broadcasts over the row; a per-channel
        # decay zips with it -- `sv` and `decay` are both indexed by K channel,
        # in the same order, so position i of each is the same channel.
        if spec.gate_kind == "gdn":
            sv = [b.fmul(s, decay) for s in sv]
        else:
            sv = [b.fmul(s, d) for s, d in zip(sv, decay)]

        # ---- S_row . k_hat  and  S_row . q_hat ----
        sum_hk = tree_reduce(b, b.fadd, [b.fmul(sv[j], kn[j]) for j in range(DK)])
        sum_hq = tree_reduce(b, b.fadd, [b.fmul(sv[j], qn[j]) for j in range(DK)])

        # ---- delta value + read-out for this value-dim ----
        v_idx = b.add(
            b.add(b.mul(b_i, b.const_i32(V_HN)), b.mul(hv_i, b.const_i32(V_HK))), tid
        )
        rv = load_scalar_as_f32(b, Vv, v_idx, dtype=spec.dtype)
        v_new = b.fmul(b.fsub(rv, sum_hk), beta)
        out_val = b.fadd(sum_hq, b.fmul(v_new, dot_kq))
        store_scalar_from_f32(b, OUT, v_idx, out_val, dtype=spec.dtype)

        # ---- rank-1 state write: S_row += k_hat * v_new ----
        state_w = b.global_ptr_add(
            STATE, b.mul(b.sext(write_pool, I64), b.const_i64(S_POOL * ST_BYTES))
        )
        ws_base = b.add(b.mul(hv_i, b.const_i32(S_HV)), b.mul(tid, b.const_i32(S_VR)))
        new_s = [b.fma(kn[j], v_new, sv[j]) for j in range(DK)]
        for c in range(0, DK, STATE_VEC):
            vec = _pack_state(b, new_s[c : c + STATE_VEC], state_dtype=spec.state_dtype)
            _store_state(
                b,
                state_w,
                b.add(ws_base, b.const_i32(c)),
                vec,
                state_dtype=spec.state_dtype,
                n=STATE_VEC,
                hint=spec.state_store_hint,
            )

    return b.kernel


def _build_warp_tiled(spec: GdnDecodeSpec) -> KernelDef:
    """v2: warp-tiled. One CTA per (seq, value-head); the
    ``head_v_dim x head_k_dim`` state is distributed across the block's warps
    (WTV x WTK lanes, VPT values/lane), and the K-reductions (L2 norms,
    dot(k,q), S.k, S.q) are folded once per WTK-lane group via shuffle-xor,
    eliminating v1's per-thread redundant compute and register pressure.
    """
    HK, HV = spec.num_k_heads, spec.num_v_heads
    DK, DV = spec.head_k_dim, spec.head_v_dim
    G = spec.v_per_k_head
    WAVE = spec.wave_size
    WTK = spec.warp_threads_k
    WTV = WAVE // WTK
    NW = spec.num_warps
    VPT = STATE_VEC
    BS = NW * WAVE
    WARP_TILE_K = WTK * VPT
    WTK_ITERS = DK // WARP_TILE_K
    WGROUP_V = NW * WTV
    BPV = spec.blocks_per_v_dim
    TILE_V = DV // BPV
    WTV_ITERS = TILE_V // WGROUP_V
    scale = 1.0 / math.sqrt(DK)
    shfl = [1 << i for i in range(WTK.bit_length() - 1)]  # [1,2,4] for WTK=8

    Q_HN, Q_HK = HK * DK, DK
    V_HN, V_HK = HV * DV, DV
    S_POOL, S_HV, S_VR = HV * DV * DK, DV * DK, DK
    ST_BYTES = _STATE_BYTES[spec.state_dtype]
    CONV, RN = spec.fuse_conv, spec.fuse_out_norm
    CONV_DIM = 2 * HK * DK + HV * DV  # packed [q | k | v] channels
    IO_BYTES = 2  # bf16 / f16; the conv state carries the I/O dtype

    io_ty = io_ir_type(spec.dtype)
    st_ty = _state_ir_type(spec.state_dtype)
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = BS

    if CONV:
        QKV = b.param(
            "mixed_qkv", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16
        )
        Q = K = Vv = QKV
    else:
        Q = b.param(
            "query", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16
        )
        K = b.param(
            "key", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16
        )
        Vv = b.param(
            "value", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16
        )
    Ag = b.param("a", PtrType(io_ty, "global"), noalias=True, readonly=True)
    Bg = b.param("b", PtrType(io_ty, "global"), noalias=True, readonly=True)
    DTB = b.param(
        "dt_bias",
        PtrType(F32 if spec.gate_kind == "kda" else io_ty, "global"),
        noalias=True,
        readonly=True,
    )
    ALOG = b.param("A_log", PtrType(F32, "global"), noalias=True, readonly=True)
    RIDX = b.param("read_indices", PtrType(I32, "global"), noalias=True, readonly=True)
    WIDX = b.param("write_indices", PtrType(I32, "global"), noalias=True, readonly=True)
    STATE = b.param("state", PtrType(st_ty, "global"), noalias=True, align=16)
    OUT = b.param(
        "out", PtrType(io_ty, "global"), noalias=True, writeonly=True, align=16
    )
    if CONV:
        CST = b.param("conv_state", PtrType(io_ty, "global"), noalias=True, align=16)
        CW = b.param(
            "conv_weight", PtrType(F32, "global"), noalias=True, readonly=True, align=16
        )
    if RN:
        OG = b.param("out_gate", PtrType(io_ty, "global"), noalias=True, readonly=True)
        NWT = b.param(
            "norm_weight", PtrType(F32, "global"), noalias=True, readonly=True
        )
    _ = b.param("batch_size", I32)  # noqa: F841
    if CONV:
        qkv_stride = b.param("qkv_stride", I32)
    if RN:
        og_stride = b.param("og_stride", I32)
        norm_eps = b.param("norm_eps", F32)
        # one f32 partial per wave for the cross-wave sum of squares
        lds_rn = b.smem_alloc_f32([NW], name_hint="rn_partials") if NW > 1 else None

    tid = b.thread_id_x()
    bidx = b.block_id_x()
    b_hv_i = b.div(bidx, b.const_i32(BPV))
    tile_v_start = b.mul(b.mod(bidx, b.const_i32(BPV)), b.const_i32(TILE_V))
    b_i = b.div(b_hv_i, b.const_i32(HV))
    hv_i = b.mod(b_hv_i, b.const_i32(HV))
    hk_i = b.div(hv_i, b.const_i32(G))
    w_tid = b.mod(tid, b.const_i32(WAVE))
    wid = b.div(tid, b.const_i32(WAVE))
    k_lane = b.mod(w_tid, b.const_i32(WTK))
    v_lane = b.div(w_tid, b.const_i32(WTK))
    warp_k_start = b.mul(k_lane, b.const_i32(VPT))
    gv_start = b.add(b.mul(wid, b.const_i32(WTV)), v_lane)

    read_pool = b.global_load_i32(RIDX, b_i)
    write_pool = b.global_load_i32(WIDX, b_i)
    active = b.land(
        b.cmp_ge(read_pool, b.const_i32(0)), b.cmp_ge(write_pool, b.const_i32(0))
    )
    with b.scf_if(active):

        def exp_f32(x):  # clamped exp2_fast; see the simple path
            arg = b.fmul(x, b.const_f32(LOG2E))
            arg = b.fmin(b.fmax(arg, b.const_f32(-EXP2_CLAMP)), b.const_f32(EXP2_CLAMP))
            return b.exp2_fast(arg)

        def log1p_f32(x):
            return b.fmul(b.log2(b.fadd(b.const_f32(1.0), x)), b.const_f32(LN2))

        def wsum(v):  # xor-butterfly sum over the WTK-lane group (broadcast in-group)
            # xor 1/2 via quad_perm (VALU DPP: no LDS crossbar / no lgkmcnt(0) stall);
            # xor 4 crosses the 4-lane quad so it stays on ds_swizzle.
            #
            # This split is NOT applied at the four sibling sites that make the
            # same choice -- they are all still on plain warp_shuffle_xor, and
            # each would take the same win for masks 1 and 2:
            #   helpers/reduction.py `_warp_xor_reduce` (the generic form of
            #     this loop, in a module this kernel already imports)
            #   cpp/helpers/attention.cpp `rocke_warp_xor_reduce_sum` (its twin)
            #   kernels/gfx950/kda_chunkwise.py `_reduce16_fadd`
            #   attention_tiled_2d.py + its gfx950 twin, register transpose
            # Sweeping them touches two engines and three families, so it is
            # deliberately not done here; see the follow-up ticket. Anyone
            # editing this loop should consider whether the sibling is still
            # waiting.
            for off in shfl:
                if off <= 2:
                    v = b.fadd(v, b.warp_shuffle_xor_quad(v, off))
                else:
                    v = b.fadd(v, b.warp_shuffle_xor(v, off))
            return v

        def silu(x):  # x * sigmoid(x), sigmoid from the same clamped exp
            return b.fmul(x, b.rcp_fast(b.fadd(b.const_f32(1.0), exp_f32(b.fneg(x)))))

        def conv4(h, w, x):  # h: 3 taps oldest-first, w: 4 weights, x: current
            acc = b.fadd(b.fmul(h[0], w[0]), b.fmul(h[1], w[1]))
            return b.fadd(acc, b.fadd(b.fmul(h[2], w[2]), b.fmul(x, w[3])))

        def conv_taps8(ch):
            """3 taps x 8 channels from [slot, channel, tap]: 24 contiguous."""
            t24 = []
            for j in range(3):
                t24 += load_vec_as_f32(
                    b,
                    cst_r,
                    b.add(b.mul(ch, b.const_i32(3)), b.const_i32(8 * j)),
                    dtype=spec.dtype,
                    n=8,
                )
            return [t24[3 * i : 3 * i + 3] for i in range(VPT)]

        def conv_weights8(ch):
            """4 weights x 8 channels from [channel, tap]: 32 contiguous f32."""
            w32 = []
            for j in range(4):
                wv = b.global_load_vN(
                    CW, b.add(b.mul(ch, b.const_i32(4)), b.const_i32(8 * j)), F32, 8
                )
                w32 += [b.vec_extract(wv, i) for i in range(8)]
            return [w32[4 * i : 4 * i + 4] for i in range(VPT)]

        # Issue every global load before any math that consumes one. The AMDGPU
        # scheduler does not hoist a load above earlier math or across the
        # k_lane == 0 store branch, so emission order bounds how many separate
        # load batches (each ended by a vmcnt wait) a wave exposes. With loads
        # first, the default and small tiles issue one batch; larger tiles still
        # split into two or three, down from up to 18 when loads were interleaved
        # with math. The floating-point arithmetic below is unchanged and in the
        # same order. The V-row indices are now computed once (v_rows) and shared
        # by the loads and the stores, which drops a few integer adds.
        a_idx = b.add(b.mul(b_i, b.const_i32(HV)), hv_i)
        if spec.gate_kind == "gdn":
            ra = load_scalar_as_f32(b, Ag, a_idx, dtype=spec.dtype)
            rb = load_scalar_as_f32(b, Bg, a_idx, dtype=spec.dtype)
            rdt = load_scalar_as_f32(b, DTB, hv_i, dtype=spec.dtype)
            ral = b.global_load_f32(ALOG, hv_i)
        else:
            rb = load_scalar_as_f32(b, Bg, a_idx, dtype=spec.dtype)
            ral = b.global_load_f32(ALOG, hv_i)  # per head in both gate kinds
            g_row = b.add(
                b.mul(b_i, b.const_i32(HV * DK)), b.mul(hv_i, b.const_i32(DK))
            )
            dtb_row = b.mul(hv_i, b.const_i32(DK))
            gvs, dtvecs = [], []
            for ki in range(WTK_ITERS):
                # Same lane offset the state load uses, so slot i of this
                # slice is the same K channel as slot i of the state vector.
                koff = b.add(warp_k_start, b.const_i32(ki * WARP_TILE_K))
                gvs.append(
                    load_vec_as_f32(b, Ag, b.add(g_row, koff), dtype=spec.dtype, n=VPT)
                )
                dtvecs.append(b.global_load_vN(DTB, b.add(dtb_row, koff), F32, VPT))

        # this lane's q,k K-chunks -> f32
        if CONV:
            # q/k/v come from one packed row: q at channel hk*DK, k one q-span
            # later, v after both. The conv taps live at the read slot.
            qkv_row = b.mul(b_i, qkv_stride)
            qk_base = b.add(qkv_row, b.mul(hk_i, b.const_i32(DK)))
            k_shift = b.const_i32(HK * DK)
            cst_r = b.global_ptr_add(
                CST,
                b.mul(b.sext(read_pool, I64), b.const_i64(CONV_DIM * 3 * IO_BYTES)),
            )
            q_taps, k_taps, q_w, k_w = {}, {}, {}, {}
        else:
            qk_base = b.add(
                b.mul(b_i, b.const_i32(Q_HN)), b.mul(hk_i, b.const_i32(Q_HK))
            )
        qn = [None] * WTK_ITERS
        kn = [None] * WTK_ITERS
        for ki in range(WTK_ITERS):
            off = b.add(qk_base, b.add(warp_k_start, b.const_i32(ki * WARP_TILE_K)))
            qn[ki] = load_vec_as_f32(b, Q, off, dtype=spec.dtype, n=VPT)
            if CONV:
                kn[ki] = load_vec_as_f32(
                    b, K, b.add(off, k_shift), dtype=spec.dtype, n=VPT
                )
                qch = b.add(
                    b.mul(hk_i, b.const_i32(DK)),
                    b.add(warp_k_start, b.const_i32(ki * WARP_TILE_K)),
                )
                kch = b.add(qch, k_shift)
                q_taps[ki], q_w[ki] = conv_taps8(qch), conv_weights8(qch)
                k_taps[ki], k_w[ki] = conv_taps8(kch), conv_weights8(kch)
            else:
                kn[ki] = load_vec_as_f32(b, K, off, dtype=spec.dtype, n=VPT)

        # raw state tiles. The pool base overflows i32 for large pools, so
        # advance the pointer by a 64-bit byte offset once and keep the in-slot
        # index in i32.
        state_r = b.global_ptr_add(
            STATE, b.mul(b.sext(read_pool, I64), b.const_i64(S_POOL * ST_BYTES))
        )
        v_rows = [
            b.add(tile_v_start, b.add(gv_start, b.const_i32(vi * WGROUP_V)))
            for vi in range(WTV_ITERS)
        ]
        # The state is read once and written once per call, so its loads and
        # stores are streamed (nontemporal): caching it gains nothing. This is
        # always on by design, not a tuning knob. q/k/v, the gates and the
        # output keep the default cache policy.
        s_raw = {}
        for vi, v_row in enumerate(v_rows):
            rs_row = b.add(
                b.mul(hv_i, b.const_i32(S_HV)), b.mul(v_row, b.const_i32(S_VR))
            )
            for ki in range(WTK_ITERS):
                off = b.add(rs_row, b.add(warp_k_start, b.const_i32(ki * WARP_TILE_K)))
                s_raw[(vi, ki)] = _load_state_f32(
                    b,
                    state_r,
                    off,
                    state_dtype=spec.state_dtype,
                    n=VPT,
                    hint=spec.state_load_hint,
                )

        # this lane's v, one per V row
        v_idx = [
            b.add(
                b.add(b.mul(b_i, b.const_i32(V_HN)), b.mul(hv_i, b.const_i32(V_HK))),
                v_row,
            )
            for v_row in v_rows
        ]
        if CONV:
            # v channel of each owned row in the packed row / conv state
            vch = [
                b.add(
                    b.const_i32(2 * HK * DK),
                    b.add(b.mul(hv_i, b.const_i32(DV)), v_row),
                )
                for v_row in v_rows
            ]
            rv = [
                load_scalar_as_f32(b, Vv, b.add(qkv_row, c), dtype=spec.dtype)
                for c in vch
            ]
            v_taps = [
                [
                    load_scalar_as_f32(
                        b,
                        cst_r,
                        b.add(b.mul(c, b.const_i32(3)), b.const_i32(t)),
                        dtype=spec.dtype,
                    )
                    for t in range(3)
                ]
                for c in vch
            ]
            v_w = []
            for c in vch:
                wv = b.global_load_vN(CW, b.mul(c, b.const_i32(4)), F32, 4)
                v_w.append([b.vec_extract(wv, i) for i in range(4)])
        else:
            rv = [load_scalar_as_f32(b, Vv, i, dtype=spec.dtype) for i in v_idx]
        if RN:
            og_row = b.add(b.mul(b_i, og_stride), b.mul(hv_i, b.const_i32(DV)))
            r_og = [
                load_scalar_as_f32(b, OG, b.add(og_row, v_row), dtype=spec.dtype)
                for v_row in v_rows
            ]
            r_nw = [b.global_load_f32(NWT, v_row) for v_row in v_rows]

        # gates (per value head)
        if spec.gate_kind == "gdn":
            x = b.fadd(ra, rdt)
            sp = b.select(
                b.fcmp("ogt", x, b.const_f32(SOFTPLUS_THRESHOLD)),
                x,
                log1p_f32(exp_f32(x)),
            )
            decay = exp_f32(b.fneg(b.fmul(exp_f32(ral), sp)))
            beta = b.rcp_fast(b.fadd(b.const_f32(1.0), exp_f32(b.fneg(rb))))
        else:
            # KDA: one decay per K channel, for the slice THIS lane owns.
            #
            # The state tile is keyed (vi, ki) -- V row and K chunk -- but a
            # channel's decay does not depend on which V row is being faded, so
            # the decay is keyed by ki alone and reused across all WTV_ITERS
            # rows. That is what holds the extra register cost to WTK_ITERS*VPT
            # values instead of multiplying with the state tile.
            beta = b.rcp_fast(b.fadd(b.const_f32(1.0), exp_f32(b.fneg(rb))))
            exp_alog = exp_f32(ral)
            decay = {}
            for ki in range(WTK_ITERS):
                gv, dtvec = gvs[ki], dtvecs[ki]
                slice_decay = []
                for i in range(VPT):
                    if spec.fuse_gate:
                        inner = b.fmul(exp_alog, b.fadd(gv[i], b.vec_extract(dtvec, i)))
                        sig = b.rcp_fast(
                            b.fadd(b.const_f32(1.0), exp_f32(b.fneg(inner)))
                        )
                        log_decay = b.fmul(b.const_f32(spec.lower_bound), sig)
                    else:
                        log_decay = gv[i]
                    slice_decay.append(exp_f32(log_decay))
                decay[ki] = slice_decay

        if CONV:
            # conv1d + SiLU on the raw q/k/v; the raw values are kept, since
            # they become the newest conv tap.
            q_raw, k_raw, v_raw = qn, kn, rv
            qn = [
                [
                    silu(conv4(q_taps[ki][i], q_w[ki][i], q_raw[ki][i]))
                    for i in range(VPT)
                ]
                for ki in range(WTK_ITERS)
            ]
            kn = [
                [
                    silu(conv4(k_taps[ki][i], k_w[ki][i], k_raw[ki][i]))
                    for i in range(VPT)
                ]
                for ki in range(WTK_ITERS)
            ]
            rv = [
                silu(conv4(v_taps[vi], v_w[vi], v_raw[vi])) for vi in range(WTV_ITERS)
            ]

        if spec.use_qk_l2norm:
            pq = wsum(
                tree_reduce(
                    b,
                    b.fadd,
                    [
                        b.fmul(qn[ki][i], qn[ki][i])
                        for ki in range(WTK_ITERS)
                        for i in range(VPT)
                    ],
                )
            )
            pk = wsum(
                tree_reduce(
                    b,
                    b.fadd,
                    [
                        b.fmul(kn[ki][i], kn[ki][i])
                        for ki in range(WTK_ITERS)
                        for i in range(VPT)
                    ],
                )
            )
            inv_q = b.rsqrt(b.fadd(pq, b.const_f32(NORM_EPS)))
            inv_k = b.rsqrt(b.fadd(pk, b.const_f32(NORM_EPS)))
            sq = b.fmul(inv_q, b.const_f32(scale))
            qn = [
                [b.fmul(qn[ki][i], sq) for i in range(VPT)] for ki in range(WTK_ITERS)
            ]
            kn = [
                [b.fmul(kn[ki][i], inv_k) for i in range(VPT)]
                for ki in range(WTK_ITERS)
            ]
        else:
            qn = [
                [b.fmul(qn[ki][i], b.const_f32(scale)) for i in range(VPT)]
                for ki in range(WTK_ITERS)
            ]

        dot_kq = wsum(
            tree_reduce(
                b,
                b.fadd,
                [
                    b.fmul(kn[ki][i], qn[ki][i])
                    for ki in range(WTK_ITERS)
                    for i in range(VPT)
                ],
            )
        )

        # decay the state tiles. decay[ki] covers the same K channels as state
        # chunk ki, in the same order, and is reused across every vi.
        sv = {}
        for (vi, ki), vec in s_raw.items():
            if spec.gate_kind == "gdn":
                sv[(vi, ki)] = [b.fmul(s, decay) for s in vec]
            else:
                sv[(vi, ki)] = [b.fmul(s, d) for s, d in zip(vec, decay[ki])]

        state_w = b.global_ptr_add(
            STATE, b.mul(b.sext(write_pool, I64), b.const_i64(S_POOL * ST_BYTES))
        )
        outs = []
        for vi in range(WTV_ITERS):
            v_row = v_rows[vi]
            phk = wsum(
                tree_reduce(
                    b,
                    b.fadd,
                    [
                        b.fmul(sv[(vi, ki)][i], kn[ki][i])
                        for ki in range(WTK_ITERS)
                        for i in range(VPT)
                    ],
                )
            )
            phq = wsum(
                tree_reduce(
                    b,
                    b.fadd,
                    [
                        b.fmul(sv[(vi, ki)][i], qn[ki][i])
                        for ki in range(WTK_ITERS)
                        for i in range(VPT)
                    ],
                )
            )
            # v_new is in-group uniform (rv, broadcast phk, beta) - no bcast.
            v_new = b.fmul(b.fsub(rv[vi], phk), beta)
            out_val = b.fadd(phq, b.fmul(v_new, dot_kq))
            if RN:
                outs.append(out_val)  # stored after the cross-wave norm below
            else:
                with b.scf_if(b.cmp_eq(k_lane, b.const_i32(0))):
                    store_scalar_from_f32(b, OUT, v_idx[vi], out_val, dtype=spec.dtype)
            ws_row = b.add(
                b.mul(hv_i, b.const_i32(S_HV)), b.mul(v_row, b.const_i32(S_VR))
            )
            for ki in range(WTK_ITERS):
                new = [b.fma(kn[ki][i], v_new, sv[(vi, ki)][i]) for i in range(VPT)]
                vec = _pack_state(b, new, state_dtype=spec.state_dtype)
                off = b.add(ws_row, b.add(warp_k_start, b.const_i32(ki * WARP_TILE_K)))
                _store_state(
                    b,
                    state_w,
                    off,
                    vec,
                    state_dtype=spec.state_dtype,
                    n=VPT,
                    hint=spec.state_store_hint,
                )

        if RN:
            # Sum of o^2 over the head's DV rows. Every k-lane of a row holds
            # the same out_val, so only k-lane 0 contributes. For NW > 1 the
            # reduction ends with a workgroup barrier.
            ss = tree_reduce(b, b.fadd, [b.fmul(o, o) for o in outs])
            contrib = b.select(b.cmp_eq(k_lane, b.const_i32(0)), ss, b.const_f32(0.0))
            total = block_lds_reduce_with_wave_prologue(
                b, contrib, lds_rn, tid, block_size=BS, wave_size=WAVE
            )
            rstd = b.rsqrt(b.fadd(b.fmul(total, b.const_f32(1.0 / DV)), norm_eps))
            for vi in range(WTV_ITERS):
                gate = b.rcp_fast(b.fadd(b.const_f32(1.0), exp_f32(b.fneg(r_og[vi]))))
                y = b.fmul(b.fmul(b.fmul(outs[vi], rstd), r_nw[vi]), gate)
                with b.scf_if(b.cmp_eq(k_lane, b.const_i32(0))):
                    store_scalar_from_f32(b, OUT, v_idx[vi], y, dtype=spec.dtype)

        if CONV:
            if NW > 1 and not RN:
                # every wave has read its taps before any tap is overwritten
                b.sync()
            cst_w = b.global_ptr_add(
                CST,
                b.mul(b.sext(write_pool, I64), b.const_i64(CONV_DIM * 3 * IO_BYTES)),
            )
            # q/k channels: one writer group (wave 0, v-lane 0), each k-lane
            # writing its 8 channels x 3 shifted taps.
            with b.scf_if(b.cmp_eq(gv_start, b.const_i32(0))):
                for ki in range(WTK_ITERS):
                    qch = b.add(
                        b.mul(hk_i, b.const_i32(DK)),
                        b.add(warp_k_start, b.const_i32(ki * WARP_TILE_K)),
                    )
                    for ch, taps, raw in (
                        (qch, q_taps, q_raw),
                        (b.add(qch, b.const_i32(HK * DK)), k_taps, k_raw),
                    ):
                        new24 = []
                        for i in range(VPT):
                            new24 += [taps[ki][i][1], taps[ki][i][2], raw[ki][i]]
                        for j in range(3):
                            store_vec(
                                b,
                                cst_w,
                                b.add(b.mul(ch, b.const_i32(3)), b.const_i32(8 * j)),
                                pack_f32_to(
                                    b, new24[8 * j : 8 * j + 8], dtype=spec.dtype
                                ),
                                n=8,
                            )
            # v channels: the k-lane-0 owner of each row
            with b.scf_if(b.cmp_eq(k_lane, b.const_i32(0))):
                for vi in range(WTV_ITERS):
                    base = b.mul(vch[vi], b.const_i32(3))
                    for t, val in enumerate((v_taps[vi][1], v_taps[vi][2], v_raw[vi])):
                        store_scalar_from_f32(
                            b, cst_w, b.add(base, b.const_i32(t)), val, dtype=spec.dtype
                        )

    return b.kernel


def gdn_decode_grid(batch: int, spec: GdnDecodeSpec) -> Tuple[int, int, int]:
    """One workgroup per (sequence, value head, v-sub-block)."""
    bpv = 1 if spec.simple else spec.blocks_per_v_dim
    return ceil_div_grid((batch * spec.num_v_heads * bpv, 1))


def gdn_decode_signature(spec: GdnDecodeSpec):
    """Kernel ABI. The fused modes only add (or, for fuse_conv, replace q/k/v
    with one packed row) parameters, so the unfused ABI is unchanged."""
    sig = SignatureBuilder()
    if spec.fuse_conv:
        sig = sig.ptr("mixed_qkv", spec.dtype)
    else:
        sig = (
            sig.ptr("query", spec.dtype).ptr("key", spec.dtype).ptr("value", spec.dtype)
        )
    sig = (
        sig.ptr("a", spec.dtype)
        .ptr("b", spec.dtype)
        .ptr("dt_bias", "f32" if spec.gate_kind == "kda" else spec.dtype)
        .ptr("A_log", "f32")
        .ptr("read_indices", "i32")
        .ptr("write_indices", "i32")
        .ptr("state", spec.state_dtype)
        .ptr("out", spec.dtype)
    )
    if spec.fuse_conv:
        sig = sig.ptr("conv_state", spec.dtype).ptr("conv_weight", "f32")
    if spec.fuse_out_norm:
        sig = sig.ptr("out_gate", spec.dtype).ptr("norm_weight", "f32")
    sig = sig.scalar("batch_size", "i32")
    if spec.fuse_conv:
        sig = sig.scalar("qkv_stride", "i32")
    if spec.fuse_out_norm:
        sig = sig.scalar("og_stride", "i32").scalar("norm_eps", "f32")
    return sig.build()
