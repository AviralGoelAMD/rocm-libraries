# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""GPU regression: ``result.bind()`` on a dispatcher-routed split-K bf16 GEMM.

A split-K spec writes an f32 ``Cf32[M, N]`` workspace, twice the bytes of a
storage-dtype C. The binding must hand the kernel a buffer of that size and
verify the full output after the host-side f32 -> bf16 finalisation cast.
Each case dispatches a shape the gfx950 bf16 router sends to split-K, launches
it through the binding, and checks every output element plus the size of the
buffer the kernel wrote into. Each case runs in a child process because an
overrun can abort the process. Skips without a gfx950 device or numpy.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest

ARCH = "gfx950"

# name: (M, N, K)
CASES = {
    "llama405b_down_m128": (128, 16384, 53248),
    "m64_n1024_k8192": (64, 1024, 8192),
}


class _SizeRecordingRuntime:
    """Pass-through runtime that records each allocation's size."""

    def __init__(self, inner):
        self.inner, self.sizes = inner, {}

    def alloc(self, n):
        p = self.inner.alloc(n)
        self.sizes[p] = n
        return p

    def __getattr__(self, k):
        return getattr(self.inner, k)


def _run_case(name):
    from rocke.dispatch.gemm.bf16_rcr import dispatch_gemm_bf16
    from rocke.dispatch.gemm.common import GemmRequest
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.hip_module import Runtime

    M, N, K = CASES[name]
    req = GemmRequest(M=M, N=N, K=K, arch=ARCH, dtype="bf16")
    res = dispatch_gemm_bf16(req)
    split_k = int(res.spec.trait.split_k)
    print(f"{name}: candidate={res.candidate.name} split_k={split_k} grid={res.grid}", flush=True)
    if split_k <= 1:
        print(f"NOT_SPLIT {name}", flush=True)
        sys.exit(3)
    kern = res.build()
    art = compile_kernel(kern, arch=ARCH, capture_ir_text=False)
    rt = _SizeRecordingRuntime(Runtime())
    fn = rt.load_module(art.hsaco).get_function(kern.name)
    binding = res.bind(verify=True)
    args, ptrs = binding.make_args(rt)
    c_bytes, need = rt.sizes[ptrs[2]], M * N * 4
    rt.launch(fn, tuple(res.grid), tuple(res.block), args)
    rt.sync()
    diff, bad, total = binding.check(rt, ptrs)
    print(f"{name}: C bytes={c_bytes} need={need} max_abs={diff:.4g} bad={bad}/{total}", flush=True)
    if c_bytes < need or bad != 0 or total != M * N:
        sys.exit(1)
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


class TestGemmBindSplitK(unittest.TestCase):
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
                timeout=900,
            )
        except subprocess.TimeoutExpired:
            self.fail(f"{name}: timed out")
        tail = (p.stdout + p.stderr)[-2000:]
        self.assertEqual(p.returncode, 0, f"{name} child failed (rc={p.returncode}):\n{tail}")
        self.assertIn("OK", p.stdout, tail)

    def test_llama405b_down_m128_bind_split_k(self):
        self._check("llama405b_down_m128")

    def test_m64_bind_split_k(self):
        self._check("m64_n1024_k8192")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        _run_case(sys.argv[2])
    else:
        unittest.main()
