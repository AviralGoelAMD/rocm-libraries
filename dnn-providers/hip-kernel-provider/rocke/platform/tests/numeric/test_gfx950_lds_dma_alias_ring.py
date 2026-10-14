# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""LDS-DMA alias rings (``IRBuilder.lds_alias_ring``) on a depth-3 ring.

One kernel streams ``_TILES`` tiles through a 3-slot LDS ring with
``async_buffer_load_lds_addr``, issuing each tile two tiles ahead and an
ordinary ``global_load`` right after every DMA write. Every iteration publishes
its tile with ``sync(keep_lds_dma=1)`` (one later LDS-DMA write may stay in
flight; the ordinary loads are not counted) and every lane reads a row another
wave wrote. The slot of every copy is a Python int, so the ring loop is fully
unrolled.

  * GPU (gfx950): the untagged, static-slot and ``any_slot`` forms all match
    an exact integer reference.
  * Host: kernels that lie about their ring are rejected -- by the builder
    (foreign ring, slot out of range, untraceable DMA address) or by the
    lowering of every available engine (an untagged DMA in a ring kernel).
"""

from __future__ import annotations

import importlib.util
import struct

import numpy as np
import pytest

from rocke.core.ir import F32, I64, IRBuilder, KernelDef, PtrType
from rocke.core.ir_serialize import serialize
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.runtime.hip_module import get_device_arch

_ARCH = "gfx950"
_THREADS = 256
_WAVE = 64
_DWORDS = 4
_DEPTH = 3
_TILES = 9
_BLOCKS = 8
_LANE_BYTES = _DWORDS * 4
_SLOT_ELEMS = _THREADS * _DWORDS
_SLOT_BYTES = _SLOT_ELEMS * 4
_MODES = ("untagged", "static", "any_slot")
_MISTAGS = (
    "foreign_ring_read",
    "foreign_ring_dma",
    "slot_out_of_range",
    "untraceable_dma_addr",
)


def _row(tid, t: int):
    """Row (lane chunk) a thread reads in tile ``t``: always another lane's,
    usually another wave's. Mirrored on the host by :func:`_reference`."""
    return (tid * 5 + 3 * t + 17) % _THREADS


def _build(mode: str, *, mistag: str | None = None) -> KernelDef:
    tagged = mode != "untagged"
    b = IRBuilder(f"lds_dma_alias_ring_{mode}")
    b.kernel.attrs["max_workgroup_size"] = _THREADS
    src = b.param("S", PtrType(F32, "global"), noalias=True, readonly=True, align=16)
    gsrc = b.param("G", PtrType(F32, "global"), noalias=True, readonly=True, align=4)
    dst = b.param("D", PtrType(F32, "global"), noalias=True, align=4)
    lds = b.smem_alloc(F32, [_DEPTH, _SLOT_ELEMS], name_hint="ring_lds")
    ring = b.lds_alias_ring(lds, _DEPTH) if tagged else None
    other = None
    if mistag in ("foreign_ring_read", "foreign_ring_dma"):
        other = b.lds_alias_ring(b.smem_alloc(F32, [1, 64], name_hint="other_lds"), 1)

    def write_tag(slot: int):
        if ring is None:
            return None
        if mistag == "foreign_ring_dma" and slot == 1:
            return other.slot(0)
        if mistag == "slot_out_of_range" and slot == 1:
            return ring.slot(_DEPTH)
        return ring.slot(slot)

    def read_tag(slot: int):
        if ring is None:
            return None
        if mistag == "foreign_ring_read" and slot == 1:
            return other.slot(0)
        return ring.any_slot if mode == "any_slot" else ring.slot(slot)

    tid = b.thread_id_x()
    bid = b.block_id_x()
    rsrc = b.buffer_rsrc(src, b.const_i32(_BLOCKS * _TILES * _SLOT_BYTES))
    voff = b.add(
        b.mul(bid, b.const_i32(_TILES * _SLOT_BYTES)),
        b.mul(tid, b.const_i32(_LANE_BYTES)),
    )
    # Each wave writes its own 64-lane run of a slot: the DMA base is per wave.
    wave_off = b.to_sgpr_u32(b.mul(b.div(tid, b.const_i32(_WAVE)), b.const_i32(_WAVE * _LANE_BYTES)))
    wave_base = b.smem_ptr_add(b.smem_addr_of(lds), b.zext(wave_off, I64))
    g_base = b.add(b.mul(bid, b.const_i32(_TILES * _THREADS)), tid)

    pending = []

    def issue(t: int) -> None:
        """DMA tile t into its slot, then one ordinary load (never counted)."""
        slot = t % _DEPTH
        addr = b.smem_ptr_add(wave_base, b.zext(b.const_i32(slot * _SLOT_BYTES), I64))
        if mistag == "untraceable_dma_addr" and t == 1:
            addr = b.zext(b.const_i32(slot * _SLOT_BYTES), I64)
        b.async_buffer_load_lds_addr(
            rsrc, addr, voff, b.const_i32(t * _SLOT_BYTES), _DWORDS, lds_alias=write_tag(slot)
        )
        pending.append(b.global_load_f32(gsrc, b.add(g_base, b.const_i32(t * _THREADS))))

    issue(0)
    issue(1)
    acc = b.const_f32(0.0)
    for t in range(_TILES):
        # LDS-DMA writes issued after tile t's: only tile t+1's (last iteration).
        b.sync(keep_lds_dma=1 if t + 1 < _TILES else 0)
        if t + 2 < _TILES:
            # Slot (t+2)%3 was last read in iteration t-1; the barrier above
            # retired those reads in every wave.
            issue(t + 2)
        slot = t % _DEPTH
        row = b.mod(  # _row(tid, t)
            b.add(b.mul(tid, b.const_i32(5)), b.const_i32(3 * t + 17)), b.const_i32(_THREADS)
        )
        elem = b.mul(row, b.const_i32(_DWORDS))
        v = b.smem_load_vN(lds, b.const_i32(slot), elem, dtype=F32, n=_DWORDS, lds_alias=read_tag(slot))
        part = b.fadd(
            b.fadd(b.vec_extract(v, 0), b.vec_extract(v, 1)),
            b.fadd(b.vec_extract(v, 2), b.vec_extract(v, 3)),
        )
        acc = b.fadd(acc, b.fmul(part, b.const_f32(float(t + 1))))
    # The ordinary loads are consumed only here, so no in-loop wait is forced
    # by their uses.
    for g in pending:
        acc = b.fadd(acc, g)
    b.global_store(dst, b.add(b.mul(bid, b.const_i32(_THREADS)), tid), acc, align=4)
    b.ret()
    return b.kernel


def _reference(s: np.ndarray, g: np.ndarray) -> np.ndarray:
    """``s``: [blocks, tiles, threads, dwords]; ``g``: [blocks, tiles, threads]."""
    tid = np.arange(_THREADS)
    out = np.zeros((_BLOCKS, _THREADS), dtype=np.float64)
    for t in range(_TILES):
        out += (t + 1) * s[:, t, _row(tid, t), :].sum(axis=-1)
    out += g.sum(axis=1)
    return out


def _engines() -> list[str]:
    return ["python"] + (["cpp"] if importlib.util.find_spec("rocke_engine") else [])


def _lower(kernel: KernelDef, engine: str) -> str:
    if engine == "python":
        return _lower_kernel_to_llvm_python(kernel, arch=_ARCH)
    from rocke.core.backend import _lower_via_cpp_engine

    return _lower_via_cpp_engine(serialize(kernel), _ARCH, None)


@pytest.mark.parametrize("engine", _engines())
@pytest.mark.parametrize("mode", _MODES)
def test_ring_lowering_tags_every_access(mode, engine):
    ll = _lower(_build(mode), engine)
    tagged = ll.count("!alias.scope")
    # One DMA write and one read per tile, nothing else tagged.
    assert tagged == (0 if mode == "untagged" else 2 * _TILES)
    if mode == "static":
        for s in range(_DEPTH):
            assert f'!alias.scope !{{!{{!"rocke.lds.ring0.slot{s}"' in ll


@pytest.mark.parametrize("mistag", _MISTAGS)
def test_builder_rejects_mistagged_ring(mistag):
    with pytest.raises(ValueError):
        _build("static", mistag=mistag)


@pytest.mark.parametrize("engine", _engines())
def test_lowering_rejects_untagged_dma_in_ring_kernel(engine):
    kernel = _build("static")
    dmas = [op for op in kernel.body.ops if op.name == "tile.async_buffer_load_lds_addr"]
    assert len(dmas) == _TILES
    # A builder-valid kernel with one DMA write stripped of its tag: the
    # backend would skip it when checking the tagged reads, so both engines
    # must refuse to lower it.
    del dmas[4].attrs["lds_ring"]
    del dmas[4].attrs["lds_slot"]
    with pytest.raises(Exception, match="must name its ring slot"):
        _lower(kernel, engine)


@pytest.mark.parametrize("mode", _MODES)
def test_ring_gpu_numeric(mode):
    if (get_device_arch(0) or "").split(":")[0] != _ARCH:
        pytest.skip("requires a gfx950 GPU")
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.host_buffers import as_u8_buffer
    from rocke.runtime.hip_module import Runtime

    rng = np.random.default_rng(20261014)
    # Small integers: every partial sum is exact in fp32, so the check is exact.
    s = rng.integers(-8, 9, size=(_BLOCKS, _TILES, _THREADS, _DWORDS)).astype(np.float32)
    g = rng.integers(-8, 9, size=(_BLOCKS, _TILES, _THREADS)).astype(np.float32)
    ref = _reference(s.astype(np.float64), g.astype(np.float64))

    art = compile_kernel(_build(mode), arch=_ARCH)
    # Torch-free (rocke runtime + numpy): importing torch after the rocke HIP
    # runtime is loaded would put a second HIP runtime in the process.
    rt = Runtime()
    module = rt.load_module(art.hsaco)
    fn = module.get_function(art.kernel_name)
    out = np.empty((_BLOCKS, _THREADS), dtype=np.float32)
    d_s, d_g, d_out = rt.alloc(s.nbytes), rt.alloc(g.nbytes), rt.alloc(out.nbytes)
    try:
        rt.memcpy_h2d(d_s, as_u8_buffer(s), s.nbytes)
        rt.memcpy_h2d(d_g, as_u8_buffer(g), g.nbytes)
        args = struct.pack("<3Q", d_s, d_g, d_out)
        for _ in range(3):
            rt.memset(d_out, 0xFF, out.nbytes)  # NaN in every output
            rt.launch(fn, (_BLOCKS, 1, 1), (_THREADS, 1, 1), args)
            rt.sync()
            rt.memcpy_d2h(as_u8_buffer(out), d_out, out.nbytes)
            bad = int(np.count_nonzero(out.astype(np.float64) != ref))
            assert bad == 0, (
                f"{mode}: {bad} of {out.size} outputs differ; "
                f"max_abs={np.nanmax(np.abs(out - ref)):.3e}"
            )
    finally:
        for ptr in (d_s, d_g, d_out):
            rt.free(ptr)
        module.unload()
