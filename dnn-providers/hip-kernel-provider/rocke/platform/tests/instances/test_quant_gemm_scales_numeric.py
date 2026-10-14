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
* A scales that vary by row: each accumulator slot of the 16x16 MFMA holds a
  different output row than the lane's A-operand row, so every slot needs the
  scale of its own output row.

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

    def test_mx_row_varying_scales_below_two(self):
        # A exponents vary by row and K group, all <= 127 (scales <= 1.0).
        K = 256
        torch = self.torch
        a = torch.randint(120, 128, (self.M, K // 32), device="cuda", generator=self.gen)
        b = torch.full((K // 32, self.N), 127, device="cuda", dtype=torch.int64)
        self._run_mx(a, b, K)

    def test_mx_row_varying_scales_full_range(self):
        K = 256
        torch = self.torch
        a = torch.randint(120, 136, (self.M, K // 32), device="cuda", generator=self.gen)
        b = torch.randint(120, 136, (K // 32, self.N), device="cuda", generator=self.gen)
        self._run_mx(a, b, K)

    # ---------------------------------------------------- block-scale GEMM

    def _run_block_scale(self, a_scale, b_scale, K, group_size_mnk):
        """``a_scale``: (M/gm, K/gk), ``b_scale``: (K/gk, N/gn) f32."""
        from rocke.instances.common.block_scale_gemm import (
            BlockScaleGemmSpec,
            block_scale_gemm_grid,
            block_scale_gemm_signature,
            build_block_scale_gemm,
        )

        gm, gn, gk = group_size_mnk
        A, B = self._operands(K)
        spec = BlockScaleGemmSpec(
            M=self.M,
            N=self.N,
            K=K,
            quant_mode="abquant",
            mantissa_dtype="fp8e4m3",
            group_size_mnk=group_size_mnk,
        )
        C = self._launch(
            build_block_scale_gemm(spec, arch=ARCH),
            block_scale_gemm_signature(spec),
            block_scale_gemm_grid(spec),
            spec.block_size,
            {
                "A": A,
                "AScale": a_scale.contiguous(),
                "B": B,
                "BScale": b_scale.contiguous(),
                "M": self.M,
                "N": self.N,
                "K": K,
            },
        )
        a_s = a_scale.repeat_interleave(gm, dim=0).repeat_interleave(gk, dim=1)
        b_s = b_scale.repeat_interleave(gk, dim=0).repeat_interleave(gn, dim=1)
        ref = (A.float() * a_s) @ (B.float() * b_s)
        self._assert_close(C, ref)

    def _pow2(self, shape):
        torch = self.torch
        e = torch.randint(-2, 3, shape, device="cuda", generator=self.gen)
        return torch.exp2(e.float())

    def test_block_scale_row_varying_1x128_by_128x128(self):
        # A: 1x128 groups (one scale per row per K group); B: 128x128 blocks.
        K = 512
        self._run_block_scale(
            self._pow2((self.M, K // 128)),
            self._pow2((K // 128, self.N // 128)),
            K,
            (1, 128, 128),
        )

    def test_block_scale_per_token_per_channel(self):
        # One A scale per row (token), one B scale per column (channel).
        K = 512
        self._run_block_scale(
            self._pow2((self.M, 1)),
            self._pow2((1, self.N)),
            K,
            (1, 1, K),
        )


if __name__ == "__main__":
    unittest.main()
