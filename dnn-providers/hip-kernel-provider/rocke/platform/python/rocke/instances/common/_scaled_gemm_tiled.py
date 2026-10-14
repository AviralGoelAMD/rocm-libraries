# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Register-tiled fp8 scaled-GEMM body shared by block_scale_gemm and mx_gemm.

Opt-in body selected by the non-default tiling knobs of
:class:`~rocke.instances.common.block_scale_gemm.BlockScaleGemmSpec` and
:class:`~rocke.instances.common.mx_gemm.MxGemmSpec`. The default specs keep
the original one-atom-per-CTA body byte-for-byte.

Structure (per CTA of ``warp_m * warp_n`` wave64 warps):

* each warp owns ``mfma_m x mfma_n`` 16x16 MFMA atoms (register tile), so
  one A fragment feeds ``mfma_n`` MFMAs and one B fragment feeds ``mfma_m``;
* ``b_layout="nk"`` takes B as (N, K) row-major (the weight layout), so the
  per-lane B fragment is K-contiguous and loads with one vector load, like A;
* scales are applied per *output* cell: a lane's accumulator slot ``i`` is
  output row ``(lane // 16) * 4 + i``, so the A scale is gathered for those
  four rows (not for the lane's A-operand row);
* MX with the 16x16x128 atom uses the in-instruction E8M0 scaled MFMA: lane
  ``l`` carries the K-chunk ``(l // 16) * 32 .. +32`` of row/col ``l % 16``,
  which is exactly one MX block, so the lane's scale byte is that block's;
* ``split_k > 1`` splits the K-groups over ``block_id_z`` and atomically
  adds f32 partials into C (caller zero-fills C).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from ...core.ir import F32, I32, I64, IRBuilder, Value
from ...helpers.atoms import MfmaAtom


@dataclass(frozen=True)
class TiledScaledGemmConfig:
    M: int
    N: int
    K: int
    atom: MfmaAtom
    # "block": f32 scales, A (M/gm, K/gk), B (K/gk, N/gn) [kn] or (N/gn, K/gk) [nk]
    # "mx":    e8m0 bytes,  A (M, K/32),     B (K/32, N)   [kn] or (N, K/32)   [nk]
    scale_mode: str
    group_mnk: Tuple[int, int, int]
    b_layout: str = "nk"
    mfma_m: int = 1
    mfma_n: int = 1
    warp_m: int = 1
    warp_n: int = 1
    split_k: int = 1
    scaled_mfma: bool = False
    prefetch: bool = False
    k_unroll: int = 1
    lds: bool = False
    # "f32" or "bf16" (rounded in the epilogue; needs split_k == 1).
    out_dtype: str = "f32"
    # 0: CTA (m, n) = (block_id_y, block_id_x). G > 0: XCD-aware remap, then
    # G-tile-row grouped order (see _cta_coords).
    tile_swizzle: int = 0
    # LDS staging only: unpadded 128-byte rows with XOR-permuted 16-byte chunks
    # instead of 144-byte padded rows (see _lds_mainloop.lds_col).
    lds_swizzle: bool = False
    # LDS staging only: double-buffered direct global->LDS copies
    # (global_load_lds_dwordx4) into the swizzled layout, one barrier per step.
    lds_dma: bool = False
    # lds_dma only: LDS buffers in the ring (stages - 1 K steps in flight).
    lds_stages: int = 2

    @property
    def tile_m(self) -> int:
        return 16 * self.mfma_m * self.warp_m

    @property
    def tile_n(self) -> int:
        return 16 * self.mfma_n * self.warp_n

    @property
    def block_size(self) -> int:
        return 64 * self.warp_m * self.warp_n


def _loop_k(cfg: TiledScaledGemmConfig) -> int:
    """K consumed per main-loop trip: one scale group, or 128 when the whole K
    is a single group (per-token x per-channel), whose scale is applied once
    in the epilogue."""
    gk = cfg.group_mnk[2]
    if cfg.scale_mode != "mx" and cfg.K // gk == 1:
        return max(128, cfg.atom.k)
    return gk


def tiled_grid(cfg: TiledScaledGemmConfig) -> Tuple[int, int, int]:
    return (cfg.N // cfg.tile_n, cfg.M // cfg.tile_m, cfg.split_k)


def validate_tiled(cfg: TiledScaledGemmConfig) -> Tuple[bool, str]:
    if cfg.b_layout not in ("kn", "nk"):
        return False, f"b_layout must be 'kn' or 'nk' (got {cfg.b_layout!r})"
    if cfg.atom.m != 16 or cfg.atom.n != 16:
        return False, "tiled body supports 16x16 atoms only"
    if min(cfg.mfma_m, cfg.mfma_n, cfg.warp_m, cfg.warp_n, cfg.split_k) < 1:
        return False, "tile knobs must be >= 1"
    if cfg.block_size > 1024:
        return False, f"block_size {cfg.block_size} > 1024"
    if cfg.M % cfg.tile_m or cfg.N % cfg.tile_n:
        return False, f"M/N must divide the CTA tile {cfg.tile_m}x{cfg.tile_n}"
    gm, gn, gk = cfg.group_mnk
    if gk % cfg.atom.k:
        return False, f"group_k ({gk}) must be a multiple of atom.k ({cfg.atom.k})"
    loop_k = _loop_k(cfg)
    if cfg.K % loop_k:
        return False, f"K ({cfg.K}) must be a multiple of the K step ({loop_k})"
    n_iters = cfg.K // loop_k
    if n_iters % cfg.split_k:
        return False, f"K steps ({n_iters}) must divide split_k ({cfg.split_k})"
    if cfg.k_unroll < 1 or (n_iters // cfg.split_k) % cfg.k_unroll:
        return False, f"K steps per split ({n_iters // cfg.split_k}) must divide k_unroll ({cfg.k_unroll})"
    if cfg.lds:
        if cfg.b_layout != "nk" or cfg.k_unroll != 1:
            return False, "lds staging needs b_layout='nk' and k_unroll=1"
        lk = _loop_k(cfg)
        for rows in (cfg.tile_m, cfg.tile_n):
            if (rows * (lk // 16)) % cfg.block_size:
                return False, f"lds copy: {rows}x{lk} panel must split evenly over {cfg.block_size} threads"
    if cfg.scaled_mfma:
        if cfg.scale_mode != "mx" or cfg.atom.k != 128:
            return False, "scaled_mfma needs MX scales and the 16x16x128 atom"
    if cfg.b_layout == "kn" and cfg.atom.k == 128:
        return False, "the 16x16x128 atom needs b_layout='nk' (K-contiguous B)"
    if cfg.scale_mode == "block" and gn % 16 and gn != 1:
        return False, "group_n must be 1 or a multiple of 16"
    if cfg.out_dtype not in ("f32", "bf16"):
        return False, f"out_dtype must be 'f32' or 'bf16' (got {cfg.out_dtype!r})"
    if cfg.out_dtype != "f32" and cfg.split_k > 1:
        return False, "split_k accumulates f32 atomically; out_dtype must be 'f32'"
    if cfg.tile_swizzle < 0 or (cfg.tile_swizzle and (cfg.M // cfg.tile_m) % cfg.tile_swizzle):
        return False, f"tile_swizzle ({cfg.tile_swizzle}) must divide the M tile count ({cfg.M // cfg.tile_m})"
    if cfg.lds_swizzle and (not cfg.lds or _loop_k(cfg) % 128):
        return False, "lds_swizzle needs lds=True and a K step that is a multiple of 128"
    if cfg.lds_dma and (not cfg.lds or _loop_k(cfg) != 128 or cfg.prefetch):
        return False, "lds_dma needs lds=True, a 128 K step, and prefetch=False (it double-buffers in LDS)"
    if cfg.lds_dma:
        fold = cfg.scale_mode == "block" and cfg.K // cfg.group_mnk[2] > 1
        if cfg.lds_stages < 2 or (fold and cfg.lds_stages != 2):
            return False, "lds_stages must be >= 2 (exactly 2 with per-K-group block scales)"
        if cfg.lds_stages * (cfg.tile_m + cfg.tile_n) * 128 > 160 * 1024:
            return False, f"lds_stages={cfg.lds_stages} panels exceed the 160 KiB LDS"
    return True, "ok"


def _load_fp8_frag(b: IRBuilder, ptr: Value, addr: Value, n: int, dtype) -> Value:
    """Per-lane fp8 operand fragment starting at ``addr``.

    For the 32-byte (16x16x128) fragment the hardware K order is NOT one
    contiguous 32-byte run: lane group ``g = lane // 16`` holds K
    ``[16g, 16g+16)`` in its first four VGPRs and ``[64+16g, 64+16g+16)`` in
    the last four (measured on gfx950 through the per-32-block E8M0 scale the
    scaled MFMA applies). ``addr`` is then the ``16g`` start; the second
    half is 64 bytes further on. Matching that order keeps every 32-element
    MX block on the lane group whose scale byte the instruction applies.
    """
    if n <= 16:
        return b.global_load_vN(ptr, addr, dtype, n)
    lo = b.global_load_vN(ptr, addr, dtype, 16)
    hi = b.global_load_vN(ptr, b.add(addr, b.const_i32(64)), dtype, 16)
    return b.vec_concat(lo, hi)


def _load_b_kn_frag(b, Bp, k_base, n_col, n_per_lane, N, dtype):
    out = b.zero_vec(dtype, n_per_lane)
    for j in range(n_per_lane):
        addr = b.add(b.mul(b.add(k_base, b.const_i32(j)), b.const_i32(N)), n_col)
        out = b.vec_insert(out, b.global_load(Bp, addr, dtype, align=1), j)
    return out


def emit_tiled_scaled_gemm(
    b: IRBuilder,
    cfg: TiledScaledGemmConfig,
    *,
    A: Value,
    AScale: Value,
    Bp: Value,
    BScale: Value,
    C: Value,
    decode_mx,
) -> None:
    """Emit the whole tiled kernel body into ``b`` (params already declared)."""
    # gfx950 computes v_pk_{add,mul}_f32 wrongly when src0 selects its low
    # half and src1 its high half (op_sel [0,1]); upstream LLVM commutes
    # such instructions (GCNHazardRecognizer::fixPkF32InvalidOpSel) but the
    # ROCm 7.13 compiler emits them. The scale folds here multiply by
    # broadcast scales whose loads LLVM merges into register pairs, which is
    # exactly how that form arises (e.g. k_unroll=2 B-scale pairs). Keep the
    # kernel on scalar f32 VALU ops so the form cannot be selected.
    b.kernel.attrs["target_features"] = "-packed-fp32-ops"
    atom = cfg.atom
    mx = cfg.scale_mode == "mx"
    M, N, K = cfg.M, cfg.N, cfg.K
    gm, gn, gk = cfg.group_mnk
    dtype = A.type.pointee
    # pt-style single K group: scale once in the epilogue, MFMA straight into acc.
    scale_in_epilogue = (not mx) and K // gk == 1
    loop_k = _loop_k(cfg)
    iters_per_split = (K // loop_k) // cfg.split_k
    atoms_per_iter = loop_k // atom.k
    c16 = b.const_i32(16)

    tid = b.thread_id_x()
    lane = b.mod(tid, b.const_i32(64))
    warp = b.div(tid, b.const_i32(64))
    wm = b.mod(warp, b.const_i32(cfg.warp_m))
    wn = b.div(warp, b.const_i32(cfg.warp_m))
    lane16 = b.mod(lane, c16)
    k_blk = b.div(lane, c16)
    cta_m, cta_n = _cta_coords(b, cfg)
    m_cta = b.mul(cta_m, b.const_i32(cfg.tile_m))
    n_cta = b.mul(cta_n, b.const_i32(cfg.tile_n))
    m_wave = b.add(m_cta, b.mul(wm, b.const_i32(16 * cfg.mfma_m)))
    n_wave = b.add(n_cta, b.mul(wn, b.const_i32(16 * cfg.mfma_n)))
    # Per-lane loop-invariant fragment bases (row * K + lane-group K start).
    # 16x16x32: 8 contiguous K per lane group; 16x16x128: see _load_fp8_frag.
    k_lane = b.mul(k_blk, b.const_i32(16 if atom.k == 128 else atom.a_per_lane))
    a_rows = [b.add(m_wave, b.add(b.const_i32(16 * mi), lane16)) for mi in range(cfg.mfma_m)]
    b_cols = [b.add(n_wave, b.add(b.const_i32(16 * ni), lane16)) for ni in range(cfg.mfma_n)]
    a_base = [b.add(b.mul(r, b.const_i32(K)), k_lane) for r in a_rows]
    b_base = [b.add(b.mul(c, b.const_i32(K)), k_lane) for c in b_cols] if cfg.b_layout == "nk" else None
    # Output rows of accumulator slot i: m_atom + k_blk*4 + i.
    out_row0 = [b.add(b.add(m_wave, b.const_i32(16 * mi)), b.mul(k_blk, b.const_i32(4))) for mi in range(cfg.mfma_m)]

    U = cfg.k_unroll
    trips_per_split = iters_per_split // U
    tr_lo = b.mul(b.block_id_z(), b.const_i32(trips_per_split)) if cfg.split_k > 1 else b.const_i32(0)
    tr_hi = b.add(tr_lo, b.const_i32(trips_per_split))

    def zero_accs():
        return [atom.zero_acc(b) for _ in range(cfg.mfma_m * cfg.mfma_n)]

    def load_iter(kidx):
        """All operand fragments (and scaled-MFMA scale bytes) of one K step,
        flattened: per atom t -> [a_frags..., b_frags..., a_sc..., b_sc...]."""
        k_group = b.mul(kidx, b.const_i32(loop_k))
        flat = []
        for t in range(atoms_per_iter):
            k0 = b.add(k_group, b.const_i32(t * atom.k)) if t else k_group
            flat += [_load_fp8_frag(b, A, b.add(base, k0), atom.a_per_lane, dtype) for base in a_base]
            if cfg.b_layout == "nk":
                flat += [_load_fp8_frag(b, Bp, b.add(base, k0), atom.b_per_lane, dtype) for base in b_base]
            else:
                kb = b.add(k0, k_lane)
                flat += [_load_b_kn_frag(b, Bp, kb, c, atom.b_per_lane, N, dtype) for c in b_cols]
            if cfg.scaled_mfma:
                # One E8M0 byte per lane per operand: block (k0 / 32 + k_blk).
                kgrp = b.add(b.div(k0, b.const_i32(32)), k_blk)
                flat += [
                    b.zext(b.global_load_i8(AScale, b.add(b.mul(r, b.const_i32(K // 32)), kgrp)), I32)
                    for r in a_rows
                ]
                flat += [
                    b.zext(b.global_load_i8(BScale, b.add(b.mul(c, b.const_i32(K // 32)), kgrp)), I32)
                    for c in b_cols
                ]
        return flat

    per_atom = cfg.mfma_m + cfg.mfma_n + (cfg.mfma_m + cfg.mfma_n if cfg.scaled_mfma else 0)
    per_step = per_atom * atoms_per_iter
    n_acc = cfg.mfma_m * cfg.mfma_n

    def load_trip(tr):
        """Fragments of the U consecutive K steps of trip ``tr``."""
        base = b.mul(tr, b.const_i32(U)) if U > 1 else tr
        flat = []
        for u in range(U):
            flat += load_iter(b.add(base, b.const_i32(u)) if u else base)
        return flat

    fold_scales = not cfg.scaled_mfma and not scale_in_epilogue

    def step_scales(kg):
        """Flattened per-(mi, ni) <4 x f32> scales of K group ``kg`` (fold path)."""
        s = _group_scales(b, cfg, AScale, BScale, kg, out_row0, b_cols, decode_mx)
        return [v for row in s for v in row]

    def compute_step(frags, kg, accs, scales=None):
        """MFMAs (+ per-group scale fold) of one K step on its fragments.
        ``scales``: the step's step_scales(kg) if already loaded."""
        ab_scale = None
        if fold_scales:
            flat_sc = scales if scales is not None else step_scales(kg)
            ab_scale = [flat_sc[mi * cfg.mfma_n : (mi + 1) * cfg.mfma_n] for mi in range(cfg.mfma_m)]
        work = list(accs) if (cfg.scaled_mfma or scale_in_epilogue) else zero_accs()
        for t in range(atoms_per_iter):
            f = frags[t * per_atom : (t + 1) * per_atom]
            a_frags = f[: cfg.mfma_m]
            b_frags = f[cfg.mfma_m : cfg.mfma_m + cfg.mfma_n]
            a_sc = f[cfg.mfma_m + cfg.mfma_n : 2 * cfg.mfma_m + cfg.mfma_n]
            b_sc = f[2 * cfg.mfma_m + cfg.mfma_n :]
            for mi in range(cfg.mfma_m):
                for ni in range(cfg.mfma_n):
                    i = mi * cfg.mfma_n + ni
                    if cfg.scaled_mfma:
                        work[i] = b.mfma_scale_f32_16x16x128_f8f6f4(
                            a_frags[mi], b_frags[ni], work[i], a_sc[mi], b_sc[ni]
                        )
                    else:
                        work[i] = atom.emit(b, a_frags[mi], b_frags[ni], work[i])
        if cfg.scaled_mfma or scale_in_epilogue:
            return work
        return [
            b.vector_fma(work[mi * cfg.mfma_n + ni], ab_scale[mi][ni], accs[mi * cfg.mfma_n + ni])
            for mi in range(cfg.mfma_m)
            for ni in range(cfg.mfma_n)
        ]

    init = [(f"acc_{i // cfg.mfma_n}_{i % cfg.mfma_n}", z) for i, z in enumerate(zero_accs())]
    if cfg.lds:
        results = _lds_mainloop(
            b, cfg, A=A, Bp=Bp, AScale=AScale, BScale=BScale, dtype=dtype, loop_k=loop_k,
            atoms_per_iter=atoms_per_iter, init=init, tr_lo=tr_lo, tr_hi=tr_hi, tid=tid,
            wm=wm, wn=wn, lane16=lane16, k_blk=k_blk, a_rows=a_rows, b_cols=b_cols,
            m_cta=m_cta, n_cta=n_cta,
            compute_step=compute_step, step_scales=step_scales if fold_scales else None,
        )
    else:
        if cfg.prefetch:
            init += [(f"pf_{j}", v) for j, v in enumerate(load_trip(tr_lo))]
        loop = b.scf_for_iter(tr_lo, tr_hi, b.const_i32(1), init, iv_name="kt")
        with loop as (tr, carried):
            carried = list(carried)
            accs = carried[:n_acc]
            if cfg.prefetch:
                frags_all = carried[n_acc:]
                tn = b.add(tr, b.const_i32(1))
                nxt = load_trip(b.select(b.cmp_lt(tn, tr_hi), tn, tr))
            else:
                frags_all = load_trip(tr)
                nxt = []
            for u in range(U):
                kg = b.mul(tr, b.const_i32(U)) if U > 1 else tr
                kg = b.add(kg, b.const_i32(u)) if u else kg
                accs = compute_step(frags_all[u * per_step : (u + 1) * per_step], kg, accs)
            b.scf_yield(*accs, *nxt)
        results = list(loop.results)[:n_acc]

    # ---- epilogue
    for mi in range(cfg.mfma_m):
        for ni in range(cfg.mfma_n):
            acc = results[mi * cfg.mfma_n + ni]
            if scale_in_epilogue:
                sc = _group_scales(b, cfg, AScale, BScale, b.const_i32(0), [out_row0[mi]], [b_cols[ni]], decode_mx)
                acc = b.vector_mul(acc, sc[0][0])
            for i in range(atom.c_per_lane):
                row = b.add(out_row0[mi], b.const_i32(i)) if i else out_row0[mi]
                addr = b.add(b.mul(row, b.const_i32(N)), b_cols[ni])
                v = b.vec_extract(acc, i)
                if cfg.split_k > 1:
                    b.global_atomic_add(C, addr, v)
                elif cfg.out_dtype == "bf16":
                    b.global_store(C, addr, b.trunc_f32_to_bf16(v), align=2)
                else:
                    b.global_store(C, addr, v, align=4)


_LDS_PAD = 16  # bytes of row padding: 144-byte rows keep 16 lanes' ds_read_b128 conflict-free
_NUM_XCD = 8  # gfx950 dispatches consecutive workgroups round-robin over its 8 XCDs


def _cta_coords(b, cfg):
    """(m, n) tile indices of this CTA.

    Default: (block_id_y, block_id_x). With ``tile_swizzle = G`` the linear
    workgroup id is first remapped so each XCD (one L2) receives a
    contiguous run of tiles instead of every 8th tile, then the run is
    walked in groups of G tile-rows (column-major inside a group), so the
    CTAs that share an L2 also share A row panels and B column panels."""
    if not cfg.tile_swizzle:
        return b.block_id_y(), b.block_id_x()
    gx, gy = cfg.N // cfg.tile_n, cfg.M // cfg.tile_m
    total = gx * gy
    pid = b.add(b.mul(b.block_id_y(), b.const_i32(gx)), b.block_id_x())
    if total % _NUM_XCD == 0:
        per = total // _NUM_XCD
        pid = b.add(
            b.mul(b.mod(pid, b.const_i32(_NUM_XCD)), b.const_i32(per)),
            b.div(pid, b.const_i32(_NUM_XCD)),
        )
    G = cfg.tile_swizzle
    in_group = G * gx
    first_m = b.mul(b.div(pid, b.const_i32(in_group)), b.const_i32(G))
    local = b.mod(pid, b.const_i32(in_group))
    return b.add(first_m, b.mod(local, b.const_i32(G))), b.div(local, b.const_i32(G))


def _lds_mainloop(
    b, cfg, *, A, Bp, AScale, BScale, dtype, loop_k, atoms_per_iter, init, tr_lo, tr_hi,
    tid, wm, wn, lane16, k_blk, a_rows, b_cols, m_cta, n_cta, compute_step, step_scales,
):
    """CTA-shared operand staging: every K step the whole CTA copies its
    ``tile_m x loop_k`` A panel and ``tile_n x loop_k`` B panel into LDS with
    coalesced 16-byte loads (8 lanes per 128-byte row), then each warp reads
    its MFMA fragments from LDS. Warps that share rows (columns) of the CTA
    tile thereby share one global read instead of each issuing its own.
    With ``prefetch`` the next step's global loads are issued right after the
    LDS store so they overlap the current step's MFMAs.

    ``lds_dma``: the panels go global -> LDS directly (global_load_lds_dwordx4,
    no VGPR round trip) into a ring of ``lds_stages`` LDS buffers; see
    ``_lds_dma_loop``."""
    atom = cfg.atom
    K = cfg.K
    TM, TN, BS = cfg.tile_m, cfg.tile_n, cfg.block_size
    dma = cfg.lds_dma
    swz = cfg.lds_swizzle or dma
    row_bytes = loop_k if swz else loop_k + _LDS_PAD
    chunks = loop_k // 16
    nbuf = cfg.lds_stages if dma else 1
    A_s = b.smem_alloc(dtype, [nbuf * TM, row_bytes], name_hint="A_s")
    B_s = b.smem_alloc(dtype, [nbuf * TN, row_bytes], name_hint="B_s")

    def lds_col(row, byte):
        """LDS byte column of logical (row, byte). Swizzled layout: unpadded
        128-byte rows whose 16-byte chunks are XOR-permuted by
        f(row) = (row & 7) ^ ((row >> 3) & 1), so the 16 rows a ds_read_b128
        wavefront slice touches land on 16 distinct (bank-half, chunk) slots
        and 8 consecutive rows on 8 distinct chunks."""
        if not swz:
            return byte
        f = b.xor(b.mod(row, b.const_i32(8)), b.mod(b.div(row, b.const_i32(8)), b.const_i32(2)))
        chunk = b.xor(b.div(byte, b.const_i32(16)), f)
        return b.add(b.mul(chunk, b.const_i32(16)), b.mod(byte, b.const_i32(16)))

    def copy_slots(rows, base_row):
        slots = []
        for i in range(rows * chunks // BS):
            c = b.add(tid, b.const_i32(i * BS)) if i else tid
            r = b.div(c, b.const_i32(chunks))
            col = b.mul(b.mod(c, b.const_i32(chunks)), b.const_i32(16))
            g = b.add(b.mul(b.add(base_row, r), b.const_i32(K)), col)
            slots.append((r, lds_col(r, col), g))
        return slots

    a_slots = copy_slots(TM, m_cta)
    b_slots = copy_slots(TN, n_cta)

    def scale_bytes(tr):
        """E8M0 scale bytes (one per lane per operand row/col) of step ``tr``.
        Issued together with the step's panel loads so their latency overlaps
        the panel loads (and, with ``prefetch``, the previous step's MFMAs)
        instead of stalling the MFMAs after the LDS barrier."""
        if not cfg.scaled_mfma:
            return []
        out = []
        k_step = b.mul(tr, b.const_i32(loop_k))
        for t in range(atoms_per_iter):
            k0 = b.add(k_step, b.const_i32(t * atom.k)) if t else k_step
            kgrp = b.add(b.div(k0, b.const_i32(32)), k_blk)
            out += [
                b.zext(b.global_load_i8(AScale, b.add(b.mul(r, b.const_i32(K // 32)), kgrp)), I32)
                for r in a_rows
            ]
            out += [
                b.zext(b.global_load_i8(BScale, b.add(b.mul(c, b.const_i32(K // 32)), kgrp)), I32)
                for c in b_cols
            ]
        return out

    def gload(tr):
        k0 = b.mul(tr, b.const_i32(loop_k))
        out = [b.global_load_vN(A, b.add(g, k0), dtype, 16) for _, _, g in a_slots]
        out += [b.global_load_vN(Bp, b.add(g, k0), dtype, 16) for _, _, g in b_slots]
        return out + scale_bytes(tr)

    n_panel = len(a_slots) + len(b_slots)

    def sstore(vals):
        na = len(a_slots)
        for (r, col, _), v in zip(a_slots, vals[:na]):
            b.smem_store_vN(A_s, [r, col], v, 16)
        for (r, col, _), v in zip(b_slots, vals[na:n_panel]):
            b.smem_store_vN(B_s, [r, col], v, 16)

    a_lrow = [b.add(b.mul(wm, b.const_i32(16 * cfg.mfma_m)), b.add(b.const_i32(16 * mi), lane16))
              for mi in range(cfg.mfma_m)]
    b_lrow = [b.add(b.mul(wn, b.const_i32(16 * cfg.mfma_n)), b.add(b.const_i32(16 * ni), lane16))
              for ni in range(cfg.mfma_n)]

    def read_frag(smem, row, t):
        if atom.k == 128:
            c0 = b.mul(k_blk, b.const_i32(16))
            lo = b.smem_load_vN(smem, row, lds_col(row, c0), dtype=dtype, n=16)
            hi = b.smem_load_vN(smem, row, lds_col(row, b.add(c0, b.const_i32(64))), dtype=dtype, n=16)
            return b.vec_concat(lo, hi)
        c0 = b.add(b.const_i32(t * atom.k), b.mul(k_blk, b.const_i32(atom.a_per_lane)))
        return b.smem_load_vN(smem, row, lds_col(row, c0), dtype=dtype, n=atom.a_per_lane)

    def frags_from_lds(scales, a_row0=None, b_row0=None):
        flat = []
        n_sc = cfg.mfma_m + cfg.mfma_n
        a_rows_l = a_lrow if a_row0 is None else [b.add(a_row0, r) for r in a_lrow]
        b_rows_l = b_lrow if b_row0 is None else [b.add(b_row0, r) for r in b_lrow]
        for t in range(atoms_per_iter):
            flat += [read_frag(A_s, r, t) for r in a_rows_l]
            flat += [read_frag(B_s, r, t) for r in b_rows_l]
            flat += scales[t * n_sc : (t + 1) * n_sc]
        return flat

    n_acc = cfg.mfma_m * cfg.mfma_n
    init = list(init)
    if dma:
        return _lds_dma_loop(
            b, cfg, A=A, Bp=Bp, A_s=A_s, B_s=B_s, row_bytes=row_bytes, loop_k=loop_k, tid=tid,
            m_cta=m_cta, n_cta=n_cta, init=init, tr_lo=tr_lo, tr_hi=tr_hi, n_acc=n_acc,
            frags_from_lds=frags_from_lds, scale_bytes=scale_bytes, step_scales=step_scales,
            compute_step=compute_step,
        )
    if cfg.prefetch:
        init += [(f"st_{j}", v) for j, v in enumerate(gload(tr_lo))]
    loop = b.scf_for_iter(
        tr_lo, tr_hi, b.const_i32(1), init, iv_name="kt", elide_trailing_barrier=False
    )
    with loop as (tr, carried):
        carried = list(carried)
        accs = carried[:n_acc]
        regs = carried[n_acc:] if cfg.prefetch else gload(tr)
        b.sync()  # previous step's LDS reads are done before overwriting
        sstore(regs)
        b.sync()
        nxt = []
        if cfg.prefetch:
            tn = b.add(tr, b.const_i32(1))
            nxt = gload(b.select(b.cmp_lt(tn, tr_hi), tn, tr))
        accs = compute_step(frags_from_lds(regs[n_panel:]), tr, accs)
        b.scf_yield(*accs, *nxt)
    return list(loop.results)[:n_acc]


def _lds_dma_loop(
    b, cfg, *, A, Bp, A_s, B_s, row_bytes, loop_k, tid, m_cta, n_cta, init, tr_lo, tr_hi, n_acc,
    frags_from_lds, scale_bytes, step_scales, compute_step,
):
    """Multi-buffered direct-to-LDS main loop (see ``_lds_mainloop``).

    ``S = cfg.lds_stages`` LDS buffers; S - 1 steps are in flight. Iteration t
    waits for step t's copy, reads its fragments from buffer t % S, issues
    step t + S - 1 into the buffer step t - 1 used, then runs its MFMAs while
    the copies are in flight. One barrier per step.

    ``global_load_lds_dwordx4`` writes each wave's 64 x 16 bytes contiguously
    at a wave-uniform LDS address, so the XOR chunk permutation of the
    swizzled layout is applied on the *source* side: the lane that lands on
    LDS chunk slot ``p`` of row ``r`` fetches logical chunk ``p ^ f(r)``."""
    K = cfg.K
    TM, TN, BS = cfg.tile_m, cfg.tile_n, cfg.block_size
    S = cfg.lds_stages
    chunks = loop_k // 16
    warp_bytes = b.mul(b.div(tid, b.const_i32(64)), b.const_i32(64 * 16))

    def dma_slots(rows, base_row):
        out = []
        for i in range(rows * chunks // BS):
            c = b.add(tid, b.const_i32(i * BS)) if i else tid
            r = b.div(c, b.const_i32(chunks))
            f = b.xor(b.mod(r, b.const_i32(8)), b.mod(b.div(r, b.const_i32(8)), b.const_i32(2)))
            lc = b.xor(b.mod(c, b.const_i32(chunks)), f)
            src = b.add(b.mul(b.add(base_row, r), b.const_i32(K)), b.mul(lc, b.const_i32(16)))
            out.append((src, b.add(warp_bytes, b.const_i32(i * BS * 16))))
        return out

    a_dma = dma_slots(TM, m_cta)
    b_dma = dma_slots(TN, n_cta)
    a_base = b.smem_addr_of(A_s)
    b_base = b.smem_addr_of(B_s)
    # VMEM instructions one step issues per wave (copies + MX scale bytes);
    # S - 2 younger steps may stay in flight when step t must have landed.
    per_step = len(a_dma) + len(b_dma)
    if cfg.scaled_mfma:
        per_step += (loop_k // cfg.atom.k) * (cfg.mfma_m + cfg.mfma_n)

    def issue(tr, buf):
        k0 = b.mul(tr, b.const_i32(loop_k))
        for slots, base, ptr, rows in ((a_dma, a_base, A, TM), (b_dma, b_base, Bp, TN)):
            buf_off = b.mul(buf, b.const_i32(rows * row_bytes))
            for src, off in slots:
                dst = b.smem_ptr_add(base, b.sext(b.add(buf_off, off), I64))
                b.global_load_lds(ptr, b.add(src, k0), dst, 16)

    def next_scales(tr):
        if cfg.scaled_mfma:
            return scale_bytes(tr)
        if step_scales is not None:
            return step_scales(tr)
        return []

    last = b.sub(tr_hi, b.const_i32(1))
    pending = []
    for s in range(S - 1):
        ts = b.smin(b.add(tr_lo, b.const_i32(s)), last) if s else tr_lo
        sc = next_scales(ts)
        issue(ts, b.const_i32(s))
        pending.append(sc)
    n_sc = len(pending[0])
    init = list(init) + [(f"sc_{j}", v) for j, v in enumerate(x for p in pending for x in p)]
    loop = b.scf_for_iter(
        tr_lo, tr_hi, b.const_i32(1), init, iv_name="kt", elide_trailing_barrier=False
    )
    with loop as (tr, carried):
        carried = list(carried)
        accs, scales = carried[:n_acc], carried[n_acc:]
        b.s_waitcnt(vmcnt=(S - 2) * per_step)  # step tr's copy has landed
        b.sync()  # ... in every wave; and every wave is done with step tr - 1
        cur = b.mod(b.sub(tr, tr_lo), b.const_i32(S))
        now = scales[:n_sc]
        frags = frags_from_lds(
            now if cfg.scaled_mfma else [],
            a_row0=b.mul(cur, b.const_i32(TM)),
            b_row0=b.mul(cur, b.const_i32(TN)),
        )
        tn = b.smin(b.add(tr, b.const_i32(S - 1)), last)
        # Keep every fragment read ahead of the next copy and every MFMA behind
        # it, so the copy is in flight for the whole MFMA block (the scheduler
        # otherwise defers fragment reads, which the copy must then follow).
        b.sched_barrier(0)
        nsc = next_scales(tn)
        issue(tn, b.mod(b.add(cur, b.const_i32(S - 1)), b.const_i32(S)))
        b.sched_barrier(0)
        accs = compute_step(frags, tr, accs, scales=now if (step_scales is not None and not cfg.scaled_mfma) else None)
        b.scf_yield(*accs, *scales[n_sc:], *nsc)
    return list(loop.results)[:n_acc]


def _group_scales(b, cfg, AScale, BScale, kg, out_row0, b_cols, decode_mx) -> List[List[Value]]:
    """Per-(mi, ni) ``<4 x f32>`` combined scale for the lane's 4 output cells."""
    gm, gn, gk = cfg.group_mnk
    K, N = cfg.K, cfg.N
    n_kg = K // gk
    mx = cfg.scale_mode == "mx"
    a_vecs = []
    for r0 in out_row0:
        vals = []
        for i in range(4):
            row = b.add(r0, b.const_i32(i)) if i else r0
            if mx:
                off = b.add(b.mul(row, b.const_i32(K // 32)), kg)
                vals.append(decode_mx(b, b.global_load_i8(AScale, off)))
            else:
                r = b.div(row, b.const_i32(gm)) if gm != 1 else row
                vals.append(b.global_load_f32(AScale, b.add(b.mul(r, b.const_i32(n_kg)), kg)))
        a_vecs.append(b.vec_pack(vals, F32))
    b_vals = []
    for c in b_cols:
        if mx:
            if cfg.b_layout == "nk":
                off = b.add(b.mul(c, b.const_i32(K // 32)), kg)
            else:
                off = b.add(b.mul(kg, b.const_i32(N)), c)
            b_vals.append(decode_mx(b, b.global_load_i8(BScale, off)))
        else:
            n_ng = (N + gn - 1) // gn
            cb = b.div(c, b.const_i32(gn)) if gn != 1 else c
            if cfg.b_layout == "nk":
                off = b.add(b.mul(cb, b.const_i32(n_kg)), kg)
            else:
                off = b.add(b.mul(kg, b.const_i32(n_ng)), cb)
            b_vals.append(b.global_load_f32(BScale, off))
    return [[b.vector_mul(av, b.vector_splat(bv, 4)) for bv in b_vals] for av in a_vecs]
