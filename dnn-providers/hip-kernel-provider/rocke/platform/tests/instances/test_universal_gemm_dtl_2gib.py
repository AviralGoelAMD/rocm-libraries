# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""GPU regression: direct-to-LDS universal GEMM on operands of 2 GiB and more.

The gfx950 direct-to-LDS path (``buffer_load ... lds``) reads A and B through
a buffer resource whose ``num_records`` is 0x7FFF0000 and whose voffset is
32-bit. If the resource base is the tensor start, bytes at offset 0x7FFF0000
or more read as zero (or wrap), so an operand of 2 GiB or more gives wrong
output in its last rows. The kernel must put the batch + tile-row origin in
the 64-bit resource base so any operand size works.

Each case fills only the first and last ``_ROWS`` rows of the large operand
(the rest is zero), so the host reference stays small: the output columns
(B case) or rows (A case) for those slabs must match, and the rest must be
exactly zero. Cases run in child processes so a fault cannot take down the
runner. Skips when no gfx950 device, numpy, or enough device memory.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest

ARCH = "gfx950"
_ROWS = 256

_TILE_256 = (256, 256, 64, 2, 4, 1, 16, 16, 32)
_TILE_128 = (128, 128, 64, 2, 2, 1, 16, 16, 32)
_DTL = dict(
    direct_to_lds=True,
    dtl_prefetch=True,
    dtl_lds_alias=True,
    lds_swizzle=True,
)
# name: (tile, trait kwargs, M, N, K, large operand)
CASES = {
    # B is exactly 2**31 bytes (N=65536, K=16384 bf16).
    "dtl256_B_2gib": (_TILE_256, _DTL, 256, 65536, 16384, "B"),
    # A is exactly 2**31 bytes.
    "dtl256_A_2gib": (_TILE_256, _DTL, 65536, 256, 16384, "A"),
    # B is past 4 GiB, through the 128x128 depth-2 ring.
    "dtl128_B_4gib": (_TILE_128, dict(_DTL, dtl_prefetch_depth=2), 128, 139264, 16384, "B"),
}


def _to_bf16_bits(x):
    import numpy as np

    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def _from_bf16_bits(h):
    import numpy as np

    return (h.astype(np.uint32) << 16).view(np.float32)


def _run_case(name):
    import ctypes
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

    tile, tr, M, N, K, big = CASES[name]
    spec = UniversalGemmSpec(
        name="dtl_2gib_" + name,
        tile=TileSpec(*tile),
        trait=TraitSpec(pipeline="compv4", scheduler="intrawave", epilogue="default", **tr),
        data=DataSpec(dtype_a="bf16", dtype_b="bf16", dtype_c="bf16", layout="RCR"),
        wave_size=64,
    )
    kern = build_universal_gemm(spec, arch=ARCH)
    art = compile_kernel(kern, arch=ARCH, capture_ir_text=False)

    rng = np.random.default_rng(0)
    big_rows = M if big == "A" else N
    small_rows = N if big == "A" else M
    small = _to_bf16_bits(rng.standard_normal((small_rows, K), dtype=np.float32))
    head = _to_bf16_bits(rng.standard_normal((_ROWS, K), dtype=np.float32) / np.sqrt(K))
    tail = _to_bf16_bits(rng.standard_normal((_ROWS, K), dtype=np.float32) / np.sqrt(K))
    row_bytes = K * 2
    big_bytes = big_rows * row_bytes
    assert big_bytes >= 1 << 31, (name, big_bytes)

    rt = Runtime()
    p_big = rt.alloc(big_bytes)
    p_small = rt.alloc(small.nbytes)
    c_bytes = M * N * 2
    pc = rt.alloc(c_bytes)
    rt.memset(p_big, 0, big_bytes)
    for off, slab in ((0, head), (big_bytes - _ROWS * row_bytes, tail)):
        rt.memcpy_h2d(
            p_big + off, (ctypes.c_ubyte * slab.nbytes).from_buffer_copy(slab.tobytes()), slab.nbytes
        )
    rt.memcpy_h2d(
        p_small, (ctypes.c_ubyte * small.nbytes).from_buffer_copy(small.tobytes()), small.nbytes
    )
    rt.memset(pc, 0xFF, c_bytes)
    pa, pb = (p_big, p_small) if big == "A" else (p_small, p_big)

    fn = rt.load_module(art.hsaco).get_function(kern.name)
    t = spec.tile
    grid = (-(-N // t.tile_n), -(-M // t.tile_m), 1)
    rt.launch(fn, grid, (spec.block_size, 1, 1), struct.pack("<QQQiii", pa, pb, pc, M, N, K))
    rt.sync()
    out = (ctypes.c_ubyte * c_bytes)()
    rt.memcpy_d2h(out, pc, c_bytes)
    c = _from_bf16_bits(np.frombuffer(bytes(out), dtype=np.uint16).reshape(M, N))
    if big == "A":
        c = c.T  # rows of the large operand on axis 1, like the B case
    s = _from_bf16_bits(small)
    worst = 0.0
    for label, slab, cols in (
        ("head", head, slice(0, _ROWS)),
        ("tail", tail, slice(big_rows - _ROWS, big_rows)),
    ):
        ref = s @ _from_bf16_bits(slab).T
        rms = float(np.sqrt(np.mean(ref * ref)))
        rel = float(np.max(np.abs(c[:, cols] - ref))) / rms
        print(f"{name} {label} rel={rel:.4g}", flush=True)
        worst = max(worst, rel) if np.isfinite(rel) else float("inf")
    mid = c[:, _ROWS : big_rows - _ROWS]
    mid_bad = int(np.count_nonzero(mid))
    print(f"{name} zero-slab nonzero={mid_bad}", flush=True)
    if not worst < 0.05 or mid_bad:
        raise SystemExit(f"{name}: numeric mismatch rel={worst} nonzero_mid={mid_bad}")
    print("OK", flush=True)


def _gpu_ready():
    try:
        import numpy  # noqa: F401

        from rocke.runtime.hip_module import get_device_arch, get_device_count

        if get_device_count() < 1 or get_device_arch() != ARCH:
            return False, f"needs a {ARCH} device"
    except Exception as e:  # noqa: BLE001
        return False, f"runtime unavailable: {e}"
    return True, ""


class TestUniversalGemmDtlLargeOperand(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ok, why = _gpu_ready()
        if not ok:
            raise unittest.SkipTest(why)

    def _check(self, name):
        try:
            p = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--child", name],
                capture_output=True,
                text=True,
                timeout=600,
            )
        except subprocess.TimeoutExpired:
            self.fail(f"{name}: timed out")
        tail = (p.stdout + p.stderr)[-2000:]
        if p.returncode != 0 and "hipMalloc(" in tail:
            self.skipTest(f"{name}: device allocation failed:\n{tail}")
        self.assertEqual(p.returncode, 0, f"{name} child failed (rc={p.returncode}):\n{tail}")
        self.assertIn("OK", p.stdout, tail)

    def test_dtl256_B_of_2gib(self):
        self._check("dtl256_B_2gib")

    def test_dtl256_A_of_2gib(self):
        self._check("dtl256_A_2gib")

    def test_dtl128_B_past_4gib(self):
        self._check("dtl128_B_4gib")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        _run_case(sys.argv[2])
    else:
        unittest.main()
