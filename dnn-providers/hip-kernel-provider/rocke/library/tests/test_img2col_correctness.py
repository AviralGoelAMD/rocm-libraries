# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""img2col: the validator gate (host) and bit-exact output values (GPU).

img2col is a pure copy, so every output element must equal the reference
exactly. Checking only for NaN misses the failure this guards: a ``vec_k``
that does not divide ``K_gemm`` makes each row's last chunk straddle into the
next row, which leaves most of the output finite but wrong.
"""

from __future__ import annotations

import importlib.util
import unittest

from rocke.runtime.hip_module import get_device_arch

_HAS_TORCH = importlib.util.find_spec("torch") is not None
if _HAS_TORCH:
    # Claim the HIP context for torch before rocke's runtime touches it (see
    # test_direct_conv_correctness.py).
    import torch

    torch.cuda.is_available()

GPU_ARCH = get_device_arch(0)

# (N, H, W, C, R, S, stride, pad): the K_gemm % 8 != 0 stems and a C % 8 == 0 control.
_SHAPES = [
    (2, 224, 224, 3, 7, 7, 2, 3),  # K_gemm 147
    (2, 56, 56, 5, 3, 3, 1, 1),  # K_gemm 45
    (2, 33, 31, 6, 3, 3, 2, 1),  # K_gemm 54: vec_k 2
    (2, 28, 28, 12, 3, 3, 1, 1),  # K_gemm 108: vec_k 4
    (2, 56, 56, 8, 3, 3, 1, 1),  # K_gemm 72: vec_k 8
]


def _problem(N, H, W, C, R, S, st, pd):
    from kernels.common.img2col import ConvProblem

    return ConvProblem(N=N, Hi=H, Wi=W, C=C, K=16, Y=R, X=S, sH=st, sW=st, pH=pd, pW=pd)


def _spec(p, vec_k):
    from kernels.common.img2col import Img2ColSpec, img2col_block_tile_m_for_M

    return Img2ColSpec(
        problem=p, block_tile_m=img2col_block_tile_m_for_M(p.M), vec_k=vec_k
    )


def _widest_vec_k(p):
    from kernels.common.img2col import is_valid_spec

    for v in (8, 4, 2, 1):
        if is_valid_spec(_spec(p, v), arch="gfx950")[0]:
            return v
    raise AssertionError(f"no valid vec_k for {p}")


class TestImg2ColValidator(unittest.TestCase):
    def test_vec_k_must_divide_k_gemm(self):
        from kernels.common.img2col import is_valid_spec

        for shape in _SHAPES:
            p = _problem(*shape)
            for v in (8, 4, 2, 1):
                ok, why = is_valid_spec(_spec(p, v), arch="gfx950")
                if p.K_gemm % v:
                    self.assertFalse(ok, (shape, v))
                    self.assertIn("not divisible by vec_k", why)

    def test_default_spec_rejected_on_odd_k_gemm(self):
        from kernels.common.img2col import Img2ColSpec, is_valid_spec

        ok, why = is_valid_spec(Img2ColSpec(problem=_problem(*_SHAPES[0])), arch="gfx950")
        self.assertFalse(ok)
        self.assertEqual(why, "K_gemm 147 not divisible by vec_k 8")


@unittest.skipUnless(
    _HAS_TORCH and GPU_ARCH in ("gfx942", "gfx950"), "needs torch and gfx942/gfx950"
)
class TestImg2ColGpu(unittest.TestCase):
    def test_values_bit_exact(self):
        import torch.nn.functional as F
        from kernels.common.img2col import build_img2col, img2col_grid, img2col_signature
        from rocke import compile_kernel
        from rocke.runtime import synchronize_and_release
        from rocke.runtime.launcher import KernelLauncher, LaunchConfig

        for shape in _SHAPES:
            N, H, W, C, R, S, st, pd = shape
            p = _problem(*shape)
            v = _widest_vec_k(p)
            with self.subTest(shape=shape, vec_k=v):
                s = _spec(p, v)
                art = compile_kernel(build_img2col(s, arch=GPU_ARCH), arch=GPU_ARCH)
                launch = KernelLauncher(
                    hsaco=art.hsaco,
                    kernel_name=art.kernel_name,
                    signature=list(img2col_signature(s)),
                )
                gen = torch.Generator(device="cuda").manual_seed(3)
                x = torch.empty(N, H, W, C, device="cuda", dtype=torch.float16)
                x.uniform_(-1, 1, generator=gen)
                y = torch.full((p.M, p.K_gemm), float("nan"), device="cuda", dtype=torch.float16)
                launch.bind(
                    {
                        "X": int(x.data_ptr()),
                        "Y": int(y.data_ptr()),
                        "X_bytes": x.numel() * 2,
                        "Y_bytes": y.numel() * 2,
                    },
                    config=LaunchConfig(
                        grid=tuple(img2col_grid(s)), block=(s.block_size, 1, 1)
                    ),
                )()
                torch.cuda.synchronize()
                synchronize_and_release(0)
                xp = F.pad(x.cpu(), (0, 0, pd, pd, pd, pd))
                a, b, c, d = xp.stride()
                ref = xp.as_strided(
                    (N, p.Ho, p.Wo, R, S, C), (a, b * st, c * st, b, c, d)
                ).reshape(p.M, p.K_gemm)
                got = y.cpu()
                self.assertFalse(bool(torch.isnan(got).any()), "NaN in output")
                self.assertEqual(int((got != ref).sum()), 0, "mismatched values")


if __name__ == "__main__":
    unittest.main(verbosity=2)
