# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GPU regression tests for the scale handling of the quantised GEMMs.

``block_scale_gemm`` and ``mx_gemm`` are launched with scale patterns that a
uniform-scale parity test cannot see, and compared against an fp32 torch
reference of ``(A * a_scale) @ (B * b_scale)``. Every scale is a power of two
and the mantissas are exact fp8 values, so the kernel and the reference only
differ by f32 summation order.

* E8M0 exponent bytes ``>= 128`` encode MX scales ``>= 2.0``; the byte must be
  decoded unsigned.

Requires a gfx942 / gfx950 GPU and torch; skipped otherwise.
Run:  PYTHONPATH=python python3 -m pytest tests/instances/test_quant_gemm_scales_numeric.py
"""

from __future__ import annotations

import importlib.util
import unittest

from rocke.runtime.hip_module import get_device_arch

ARCH = get_device_arch(0)
_HAS_TORCH = importlib.util.find_spec("torch") is not None
_RELERR_TOL = 1e-3


@unittest.skipUnless(
    ARCH in ("gfx942", "gfx950") and _HAS_TORCH,
    "needs a gfx942 / gfx950 GPU and torch",
)
class TestQuantGemmScales(unittest.TestCase):
    M, N = 64, 128

    @classmethod
    def setUpClass(cls):
        import torch

        cls.torch = torch
        cls.fp8 = torch.float8_e4m3fnuz if ARCH == "gfx942" else torch.float8_e4m3fn
        cls.gen = torch.Generator(device="cuda").manual_seed(1234)

    def _operands(self, K):
        torch = self.torch
        A = (torch.randn(self.M, K, device="cuda", generator=self.gen) * 1.5).to(self.fp8)
        B = (torch.randn(K, self.N, device="cuda", generator=self.gen) * 1.5).to(self.fp8)
        return A, B

    def _launch(self, kernel, signature, grid, block, args):
        from rocke.helpers import compile_kernel
        from rocke.runtime.launcher import KernelLauncher, LaunchConfig

        torch = self.torch
        C = torch.full((self.M, self.N), float("nan"), device="cuda")
        art = compile_kernel(kernel, arch=ARCH)
        launcher = KernelLauncher(hsaco=art.hsaco, kernel_name=kernel.name, signature=signature)
        launcher(
            {**args, "C": C},
            config=LaunchConfig(grid=grid, block=(block, 1, 1)),
        )
        torch.cuda.synchronize()
        return C

    def _assert_close(self, C, ref):
        torch = self.torch
        self.assertTrue(bool(torch.isfinite(C).all()), "kernel left non-finite output")
        scale = ref.abs().max().item()
        self.assertGreater(scale, 0.0)
        relerr = (C - ref).abs().max().item() / scale
        self.assertLess(relerr, _RELERR_TOL, f"relerr {relerr:.3e}")

    # ------------------------------------------------------------- MX GEMM

    def _run_mx(self, a_e8m0, b_e8m0, K):
        """``a_e8m0``: (M, K/32), ``b_e8m0``: (K/32, N) integer exponents."""
        from rocke.instances.common.mx_gemm import (
            MxGemmSpec,
            build_mx_gemm,
            mx_gemm_grid,
            mx_gemm_signature,
        )

        torch = self.torch
        A, B = self._operands(K)
        spec = MxGemmSpec(M=self.M, N=self.N, K=K, mantissa_dtype="fp8e4m3")
        C = self._launch(
            build_mx_gemm(spec, arch=ARCH),
            mx_gemm_signature(spec),
            mx_gemm_grid(spec),
            spec.block_size,
            {
                "A": A,
                "AScale": a_e8m0.to(torch.uint8).contiguous().view(torch.int8),
                "B": B,
                "BScale": b_e8m0.to(torch.uint8).contiguous().view(torch.int8),
                "M": self.M,
                "N": self.N,
                "K": K,
            },
        )

        def decode(e):
            s = torch.exp2(e.float() - 127.0)
            return torch.where((e == 0) | (e == 255), torch.zeros_like(s), s)

        a_s = decode(a_e8m0).repeat_interleave(32, dim=1)
        b_s = decode(b_e8m0).repeat_interleave(32, dim=0)
        ref = (A.float() * a_s) @ (B.float() * b_s)
        self._assert_close(C, ref)

    def test_mx_e8m0_uniform_scale_four(self):
        # Exponent byte 129 == scale 4.0 everywhere.
        K = 256
        torch = self.torch
        a = torch.full((self.M, K // 32), 129, device="cuda", dtype=torch.int32)
        b = torch.full((K // 32, self.N), 129, device="cuda", dtype=torch.int32)
        self._run_mx(a, b, K)

    def test_mx_e8m0_exponents_above_127(self):
        # Exponents span 120..135 along K (A) and along K x N (B), plus the
        # 0 / 255 sentinels in B. A's scale is the same for every row, so only
        # the exponent decode is exercised here.
        K = 256
        torch = self.torch
        kg = K // 32
        a_k = torch.randint(120, 136, (1, kg), device="cuda", generator=self.gen)
        a = a_k.expand(self.M, kg).contiguous()
        b = torch.randint(120, 136, (kg, self.N), device="cuda", generator=self.gen)
        b[0, 3] = 255
        b[1, 5] = 0
        self._run_mx(a, b, K)


if __name__ == "__main__":
    unittest.main()
