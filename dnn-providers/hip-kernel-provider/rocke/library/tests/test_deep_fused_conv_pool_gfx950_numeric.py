# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GPU numerics for the gfx950 deep-fused conv0 -> conv1 -> maxpool kernel.

Pins the input-footprint LDS cache (``cache_input_footprint`` and
``direct_conv0_from_input_cache``). Its fill loop strides the block over
``foot_h * foot_w * C`` elements; when the block size does not divide that
count, the lanes past the end clamp to index 0. Those lanes must not store,
or they overwrite footprint element ``[0, 0]`` (top-left halo pixel, channel
0) with zero. The damage shows only in tiles whose halo corner lies inside
the image, i.e. ``block_y > 0`` and ``block_z > 0``, so every case here has
interior tiles and a footprint the block does not divide.

The default path (both options off) runs alongside as a control. Each case
compares against a float32 CPU reference whose conv0 output is rounded to
fp16, the intermediate the kernel keeps. Requires a gfx950 GPU and torch.
"""

from __future__ import annotations

import ctypes
import importlib.util
import unittest

import pytest

from rocke.runtime.hip_module import get_device_arch

pytestmark = pytest.mark.gpu

_HAS_TORCH = importlib.util.find_spec("torch") is not None
GPU_ARCH = get_device_arch(0)

# Max error relative to max |ref|. The clean kernel stays below 1e-3 on these
# inputs. A clobbered halo pixel can land anywhere from a few 1e-3 (when it
# barely moves the pooled max) to above 1e-1, so the bound sits just above
# clean rather than at a loose 5e-2.
_TOL = 2e-3

# (h, w, c, k0, k1, pool_tile_h, pool_tile_w, tile_k); every case has interior
# tiles and a footprint the 128-thread block does not divide.
_SHAPES = (
    (32, 32, 8, 32, 32, 4, 4, 32),  # foot 10x10x8 = 800
    (32, 64, 8, 32, 32, 4, 8, 16),  # foot 10x18x8 = 1440
    (32, 32, 24, 32, 32, 4, 4, 32),  # foot 10x10x24 = 2400
)
_MODES = ("default", "cache_input_footprint", "direct_conv0_from_input_cache")


def _skip_reason() -> str:
    if not GPU_ARCH:
        return "no ROCm GPU detected"
    if not _HAS_TORCH:
        return "torch not importable"
    if GPU_ARCH != "gfx950":
        return f"gfx950 deep-fused kernel; got {GPU_ARCH}"
    return ""


_SKIP_REASON = _skip_reason()


def _u8(t):
    return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())


def _run(spec, h, w, c, k0, k1):
    import torch
    import torch.nn.functional as F

    from rocke import compile_kernel
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig
    from kernels.common.deep_fused_conv_pool import (
        deep_fused_conv_pool_problem_values,
    )
    from kernels.gfx950.deep_fused_conv_pool import (
        build_deep_fused_conv_pool,
        deep_fused_conv_pool_grid,
        deep_fused_conv_pool_signature,
    )

    p = spec.problem
    art = compile_kernel(build_deep_fused_conv_pool(spec, arch=GPU_ARCH), arch=GPU_ARCH)
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=art.kernel_name,
        signature=deep_fused_conv_pool_signature(spec),
    )
    g = torch.Generator().manual_seed(0)
    X = torch.empty(1, h, w, c).normal_(0, 0.5, generator=g).half()
    W0 = torch.empty(k0, 3, 3, c).normal_(0, 0.25, generator=g).half()
    W1 = torch.empty(k1, k0).normal_(0, 0.25, generator=g).half()
    Y = torch.empty(1, p.pool_ho, p.pool_wo, k1, dtype=torch.float16)

    c0 = F.conv2d(X.float().permute(0, 3, 1, 2), W0.float().permute(0, 3, 1, 2), padding=1)
    c0 = F.relu(c0).half().float()
    c1 = F.relu(F.conv2d(c0, W1.float().view(k1, k0, 1, 1)))
    ref = F.max_pool2d(c1, 2, 2).permute(0, 2, 3, 1).contiguous()

    rt = Runtime()
    bufs = [rt.alloc(t.nbytes) for t in (X, W0, Y, W1)]
    try:
        rt.memcpy_h2d(bufs[0], _u8(X), X.nbytes)
        rt.memcpy_h2d(bufs[1], _u8(W0), W0.nbytes)
        rt.memcpy_h2d(bufs[3], _u8(W1), W1.nbytes)
        rt.memset(bufs[2], 0xFF, Y.nbytes)  # NaN: an unwritten output fails
        values = dict(
            A=bufs[0],
            B=bufs[1],
            Y=bufs[2],
            W1=bufs[3],
            W1_bytes=W1.nbytes,
            A_bytes=X.nbytes,
            B_bytes=W0.nbytes,
            Y_bytes=Y.nbytes,
        )
        values.update(
            deep_fused_conv_pool_problem_values(
                p.conv, tile_m=spec.tile_m, tile_n=spec.tile_n
            )
        )
        launcher(
            values,
            config=LaunchConfig(
                grid=deep_fused_conv_pool_grid(spec),
                block=(spec.block_size, 1, 1),
                fence=True,
            ),
        )
        rt.memcpy_d2h(_u8(Y), bufs[2], Y.nbytes)
    finally:
        for buf in bufs:
            rt.free(buf)
    return Y.float(), ref


@unittest.skipIf(bool(_SKIP_REASON), _SKIP_REASON)
class TestDeepFusedConvPoolInputCacheGfx950(unittest.TestCase):
    def test_footprint_tail_lanes_do_not_clobber_cache(self):
        import torch

        from kernels.gfx950.deep_fused_conv_pool import (
            is_valid_spec,
            make_deep_fused_conv_pool_spec,
        )

        for h, w, c, k0, k1, pth, ptw, tk in _SHAPES:
            for mode in _MODES:
                with self.subTest(shape=(h, w, c, k0, k1), pool_tile=(pth, ptw), mode=mode):
                    flags = {} if mode == "default" else {mode: True}
                    spec = make_deep_fused_conv_pool_spec(
                        h=h,
                        w=w,
                        c=c,
                        k0=k0,
                        k1=k1,
                        pool_tile_h=pth,
                        pool_tile_w=ptw,
                        tile_k=tk,
                        tile_n=32,
                        warp_m=2,
                        warp_n=1,
                        **flags,
                    )
                    ok, why = is_valid_spec(spec, arch=GPU_ARCH)
                    self.assertTrue(ok, why)
                    foot = (2 * pth + 2) * (2 * ptw + 2) * c
                    # The case only guards the bug if the block leaves a tail.
                    self.assertNotEqual(foot % spec.block_size, 0)
                    out, ref = _run(spec, h, w, c, k0, k1)
                    self.assertFalse(bool(torch.isnan(out).any()), "unwritten output")
                    scale = float(ref.abs().max().clamp(min=1.0))
                    err = float((out - ref).abs().max()) / scale
                    self.assertLess(err, _TOL)


if __name__ == "__main__":
    unittest.main()
