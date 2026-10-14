# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""GPU regression: ``pad_m`` / ``pad_n`` universal GEMM must not read past A / B.

The last M (N) tile of a padded GEMM overhangs the operand. The epilogue masks
the overhanging rows' stores, so the loads must stay in bounds too. This test
places the operand at the very end of a virtual-memory mapping whose next page
is reserved but unmapped, so any read past the operand's last row faults
(``hipErrorIllegalAddress`` / memory access fault) instead of silently reading
a neighbour allocation. Each case runs in a child process because the fault
aborts the process.

Cases cover the decode-shaped compv4 path (VGPR-staged global loads) and the
gfx950 direct-to-LDS path (``buffer_load ... lds``), for both operands.
Skips when no gfx950 device, numpy, or HIP virtual-memory API is available.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import unittest

ARCH = "gfx950"

# name: (tile, trait kwargs, M, N, K, operand placed at the mapping end)
_TILE_DECODE = (16, 64, 64, 1, 4, 1, 16, 16, 32)
_TILE_DTL = (128, 128, 64, 2, 2, 1, 16, 16, 32)
_DTL = dict(direct_to_lds=True, dtl_prefetch=True, lds_swizzle=True)
CASES = {
    "decode_pad_m_A": (_TILE_DECODE, dict(pad_m=True), 1, 256, 1024, "A"),
    "decode_pad_n_B": (_TILE_DECODE, dict(pad_m=True, pad_n=True), 16, 100, 1024, "B"),
    "dtl_pad_m_A": (_TILE_DTL, dict(_DTL, pad_m=True), 100, 256, 1024, "A"),
    "dtl_pad_n_B": (_TILE_DTL, dict(_DTL, pad_n=True), 256, 200, 1024, "B"),
}


# --------------------------------------------------------------- HIP VMM
class _MemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _AllocFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
    ]


class _AllocProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),  # hipMemAllocationTypePinned = 1
        ("requestedHandleType", ctypes.c_int),  # hipMemHandleTypeNone = 0
        ("location", _MemLocation),  # hipMemLocationTypeDevice = 1
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", _AllocFlags),
    ]


class _AccessDesc(ctypes.Structure):
    _fields_ = [("location", _MemLocation), ("flags", ctypes.c_int)]  # ReadWrite = 3


def _hip():
    from rocke.runtime.hip_module import _resolve_hip

    return _resolve_hip()


def _ck(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what} failed: hipError {rc}")


def _guarded_alloc(nbytes):
    """Device pointer to ``nbytes`` that end exactly at an unmapped page."""
    hip = _hip()
    prop = _AllocProp()
    prop.type = 1
    prop.location = _MemLocation(1, 0)
    gran = ctypes.c_size_t(0)
    _ck(hip.hipMemGetAllocationGranularity(ctypes.byref(gran), ctypes.byref(prop), 0),
        "hipMemGetAllocationGranularity")
    g = gran.value
    size = -(-nbytes // g) * g
    base = ctypes.c_void_p(0)
    # Reserve twice the mapped size; only the first half is ever mapped.
    _ck(hip.hipMemAddressReserve(ctypes.byref(base), ctypes.c_size_t(2 * size),
                                 ctypes.c_size_t(g), None, ctypes.c_ulonglong(0)),
        "hipMemAddressReserve")
    handle = ctypes.c_void_p(0)
    _ck(hip.hipMemCreate(ctypes.byref(handle), ctypes.c_size_t(size), ctypes.byref(prop),
                         ctypes.c_ulonglong(0)), "hipMemCreate")
    _ck(hip.hipMemMap(base, ctypes.c_size_t(size), ctypes.c_size_t(0), handle,
                      ctypes.c_ulonglong(0)), "hipMemMap")
    acc = _AccessDesc(_MemLocation(1, 0), 3)
    _ck(hip.hipMemSetAccess(base, ctypes.c_size_t(size), ctypes.byref(acc), ctypes.c_size_t(1)),
        "hipMemSetAccess")
    return base.value + size - nbytes


# --------------------------------------------------------------- child
def _to_bf16_bits(x):
    import numpy as np

    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def _from_bf16_bits(h):
    import numpy as np

    return (h.astype(np.uint32) << 16).view(np.float32)


def _run_case(name):
    import struct

    import numpy as np

    from rocke.helpers.compile import compile_kernel
    from rocke.instances.common.gemm_universal import (
        DataSpec,
        TileSpec,
        TraitSpec,
        UniversalGemmSpec,
        build_universal_gemm,
    )
    from rocke.runtime.hip_module import Runtime

    tile, tr, M, N, K, edge = CASES[name]
    spec = UniversalGemmSpec(
        name="pad_oob_" + name,
        tile=TileSpec(*tile),
        trait=TraitSpec(pipeline="compv4", scheduler="intrawave", epilogue="default", **tr),
        data=DataSpec(dtype_a="bf16", dtype_b="bf16", dtype_c="bf16", layout="RCR"),
        wave_size=64,
    )
    kern = build_universal_gemm(spec, arch=ARCH)
    art = compile_kernel(kern, arch=ARCH, capture_ir_text=False)

    rng = np.random.default_rng(0)
    a = _to_bf16_bits(rng.standard_normal((M, K), dtype=np.float32))
    bm = _to_bf16_bits(rng.standard_normal((N, K), dtype=np.float32) / np.sqrt(K))
    ref = _from_bf16_bits(a) @ _from_bf16_bits(bm).T

    rt = Runtime()
    a_bytes, b_bytes, c_bytes = a.nbytes, bm.nbytes, M * N * 2
    pa = _guarded_alloc(a_bytes) if edge == "A" else rt.alloc(a_bytes)
    pb = _guarded_alloc(b_bytes) if edge == "B" else rt.alloc(b_bytes)
    pc = rt.alloc(c_bytes)
    rt.memcpy_h2d(pa, (ctypes.c_ubyte * a_bytes).from_buffer_copy(a.tobytes()), a_bytes)
    rt.memcpy_h2d(pb, (ctypes.c_ubyte * b_bytes).from_buffer_copy(bm.tobytes()), b_bytes)
    rt.memset(pc, 0xFF, c_bytes)

    fn = rt.load_module(art.hsaco).get_function(kern.name)
    t = spec.tile
    grid = (-(-N // t.tile_n), -(-M // t.tile_m), 1)
    rt.launch(fn, grid, (spec.block_size, 1, 1), struct.pack("<QQQiii", pa, pb, pc, M, N, K))
    rt.sync()
    out = (ctypes.c_ubyte * c_bytes)()
    rt.memcpy_d2h(out, pc, c_bytes)
    c = _from_bf16_bits(np.frombuffer(bytes(out), dtype=np.uint16).reshape(M, N))
    rms = float(np.sqrt(np.mean(ref * ref)))
    rel = float(np.max(np.abs(c - ref))) / rms
    print(f"{name} rel={rel:.4g}", flush=True)
    if not np.isfinite(rel) or rel >= 0.05:
        raise SystemExit(f"{name}: numeric mismatch rel={rel}")
    print("OK", flush=True)


# --------------------------------------------------------------- parent
def _gpu_ready():
    try:
        import numpy  # noqa: F401

        from rocke.runtime.hip_module import get_device_arch, get_device_count

        if get_device_count() < 1 or get_device_arch() != ARCH:
            return False, f"needs a {ARCH} device"
        if not hasattr(_hip(), "hipMemAddressReserve"):
            return False, "HIP virtual-memory API unavailable"
    except Exception as e:  # noqa: BLE001
        return False, f"runtime unavailable: {e}"
    return True, ""


class TestUniversalGemmPadNoOobRead(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ok, why = _gpu_ready()
        if not ok:
            raise unittest.SkipTest(why)

    def _check(self, name):
        # An out-of-bounds read of the unmapped page either aborts the child
        # (memory access fault) or, with retry-on-fault enabled, hangs it;
        # both are failures.
        try:
            p = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--child", name],
                capture_output=True,
                text=True,
                timeout=240,
            )
        except subprocess.TimeoutExpired:
            self.fail(f"{name}: kernel hung (unmapped-page read past the operand)")
        tail = (p.stdout + p.stderr)[-2000:]
        self.assertEqual(p.returncode, 0, f"{name} child failed (rc={p.returncode}):\n{tail}")
        self.assertIn("OK", p.stdout, tail)

    def test_decode_pad_m_reads_stay_in_A(self):
        self._check("decode_pad_m_A")

    def test_decode_pad_n_reads_stay_in_B(self):
        self._check("decode_pad_n_B")

    def test_dtl_pad_m_reads_stay_in_A(self):
        self._check("dtl_pad_m_A")

    def test_dtl_pad_n_reads_stay_in_B(self):
        self._check("dtl_pad_n_B")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        _run_case(sys.argv[2])
    else:
        unittest.main()
