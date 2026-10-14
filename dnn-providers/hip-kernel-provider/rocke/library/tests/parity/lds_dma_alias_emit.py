#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/lds_dma_alias_emit.py -- Python reference emitter for LDS-DMA
# alias rings (IRBuilder.lds_alias_ring + the lds_alias= tag on LDS-DMA writes
# and LDS reads) and the counted barrier sync(keep_lds_dma=N).
#
# Each config builds one small two-ring kernel:
#   1. DMA fills slot 0 of rings A and B; full barrier.
#   2. DMA into slot 1 of A and of B, with an ordinary global load issued
#      between the two; slot 0 of A and B is read meanwhile.
#   3. sync(keep_lds_dma=N) publishes A slot 1 while B slot 1 may stay in
#      flight; A slot 1 and B slot 0 are read.
# Every tagged op kind appears: async_buffer_load_lds_addr,
# async_buffer_load_lds, global_load_lds, smem_load_vN (vec 1 and vec 8),
# ds_read_tr16_b64 and (config 3) ds_read_tr16_b128. Byte-compared against
# lds_dma_alias_emit.c.
#
#   config 0: no rings, sync()                     -- the untagged baseline
#   config 1: rings A(2) B(2), static slots, keep_lds_dma=1
#   config 2: rings A(2) B(3), dynamic-slot reads (any_slot), keep_lds_dma=0
#   config 3: rings A(2) B(2), static slots, keep_lds_dma=63 (both vmcnt fields),
#             step-3 B read as ds_read_tr16_b128 (lowering-only: gfx950 LLVM has
#             no llvm.amdgcn.ds.read.tr16.b128 to link; configs 0-2 use b64)
from rocke.core.ir import BF16, F32, I64, IRBuilder, KernelDef, PtrType

from _emit_common import run_emit

# (tagged?, slots of ring B, any-slot reads?, keep_lds_dma, b128 read?, arch)
CONFIGS = [
    (False, 2, False, 0, False, "gfx950"),
    (True, 2, False, 1, False, "gfx950"),
    (True, 3, True, 0, False, "gfx950"),
    (True, 2, False, 63, True, "gfx950"),
]

_ROWS = 32
_COLS = 64
_SLOT_BYTES = _ROWS * _COLS * 2
_DWORDS = 4


def _spec(idx: int):
    if not 0 <= idx < len(CONFIGS):
        raise SystemExit(f"unknown config index {idx}")
    return CONFIGS[idx][:5], CONFIGS[idx][5]


def _build(spec, *, arch: str = "gfx950") -> KernelDef:
    tagged, b_slots, any_slot, keep, wide_tr = spec
    b = IRBuilder("lds_dma_alias")
    b.kernel.attrs["max_workgroup_size"] = 256
    src = b.param("S", PtrType(BF16, "global"), noalias=True, readonly=True, align=16)
    dst = b.param("D", PtrType(F32, "global"), noalias=True, align=16)
    a_lds = b.smem_alloc(BF16, [2, _ROWS, _COLS], name_hint="A_lds")
    b_lds = b.smem_alloc(BF16, [b_slots, _ROWS, _COLS], name_hint="B_lds")
    ring_a = b.lds_alias_ring(a_lds, 2) if tagged else None
    ring_b = b.lds_alias_ring(b_lds, b_slots) if tagged else None

    def tag(ring, slot):
        if ring is None:
            return None
        return ring.slot(slot) if slot is not None else ring.any_slot

    def read_tag(ring, slot):
        return tag(ring, None if any_slot else slot)

    tid = b.thread_id_x()
    rsrc = b.buffer_rsrc(src, b.const_i32(1 << 20))
    voff = b.mul(tid, b.const_i32(_DWORDS * 4))
    soff = b.const_i32(0)
    a_base = b.smem_addr_of(a_lds)
    b_base = b.smem_addr_of(b_lds)
    slot_off = b.zext(b.const_i32(_SLOT_BYTES), I64)
    a_slot1 = b.smem_ptr_add(a_base, slot_off)
    b_slot1 = b.smem_ptr_add(b_base, slot_off)

    # 1. Fill slot 0 of both rings, then a full barrier.
    b.async_buffer_load_lds_addr(rsrc, a_base, voff, soff, _DWORDS, lds_alias=tag(ring_a, 0))
    b.global_load_lds(src, voff, b_base, _DWORDS * 4, lds_alias=tag(ring_b, 0))
    b.sync()

    # 2. Fill slot 1 (an ordinary load in between) while slot 0 is read.
    b.async_buffer_load_lds(rsrc, a_slot1, voff, soff, _DWORDS, lds_alias=tag(ring_a, 1))
    g = b.global_load(src, b.add(tid, b.const_i32(4096)), BF16, align=2)
    b.async_buffer_load_lds_addr(rsrc, b_slot1, voff, soff, _DWORDS, lds_alias=tag(ring_b, 1))
    row = b.div(tid, b.const_i32(8))
    col = b.mul(b.mod(tid, b.const_i32(8)), b.const_i32(8))
    c0 = b.const_i32(0)
    c1 = b.const_i32(1)
    x0 = b.smem_load_vN(a_lds, c0, row, col, dtype=BF16, n=8, lds_alias=read_tag(ring_a, 0))
    x1 = b.smem_load_vN(a_lds, c0, row, c1, dtype=BF16, n=1, lds_alias=read_tag(ring_a, 0))
    y0 = b.ds_read_tr16_b64(b_lds, c0, row, col, dtype=BF16, lds_alias=read_tag(ring_b, 0))
    acc = b.fadd(b.cast_to_f32(b.vec_extract(x0, 7)), b.cast_to_f32(b.vec_extract(x1, 0)))
    acc = b.fadd(acc, b.cast_to_f32(b.vec_extract(y0, 3)))
    acc = b.fadd(acc, b.cast_to_f32(g))

    # 3. Publish A slot 1 (B slot 1 may stay in flight), then read it.
    b.sync(keep_lds_dma=keep)
    read_tr = b.ds_read_tr16_b128 if wide_tr else b.ds_read_tr16_b64
    z0 = read_tr(b_lds, c0, row, col, dtype=BF16, lds_alias=read_tag(ring_b, 0))
    z1 = b.smem_load_vN(a_lds, c1, row, col, dtype=BF16, n=8, lds_alias=read_tag(ring_a, 1))
    # b64 returns 4 lanes, b128 8: lane 5 only exists in the wide form.
    acc = b.fadd(acc, b.cast_to_f32(b.vec_extract(z0, 5 if wide_tr else 1)))
    acc = b.fadd(acc, b.cast_to_f32(b.vec_extract(z1, 2)))
    b.global_store(dst, tid, acc, align=4)
    b.ret()
    return b.kernel


def main() -> int:
    return run_emit(
        _spec,
        _build,
        usage="usage: lds_dma_alias_emit.py <config_index> [ll|ir|verify]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
