# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""On-GPU numerics of the gfx950 grouped-conv dispatcher, end to end.

Each request goes through ``dispatch_conv_grouped`` and is launched the way a
caller launches it: the pre-pass first when the spec declares one (channel
pad for small-C stems, weight transpose for grouped dgrad) into a caller
workspace, then the dispatched kernel with ``launch_values_for``; a zeroed
output for split-K atomics; the workspace reduce for two-stage wgrad. The
output is NaN-filled first and compared with an fp32 CPU reference.

Covers every gfx950 route the dispatcher adds on top of the implicit GEMM:
depthwise forward (direct tiled), depthwise dgrad (direct streaming, each
strip width), small-cpg grouped forward and grouped dgrad (direct MFMA), the
merged depthwise wgrad, and the channel-padded stem.
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


def _skip_reason() -> str:
    if not _HAS_TORCH:
        return "torch not importable"
    if GPU_ARCH != "gfx950":
        return f"needs gfx950 (got {GPU_ARCH!r})"
    return ""


_SKIP_REASON = _skip_reason()
_TOL = {"fp16": 1e-2, "bf16": 3e-2}

# (label, N, C, K, H, W, KH, KW, stride, pad, G, direction, expected route)
_CASES = [
    ("dw fwd 3x3 s2", 2, 96, 96, 30, 30, 3, 3, 2, 1, 96, "fwd", "dw_tiled"),
    ("dw fwd 7x7", 1, 64, 64, 17, 17, 7, 7, 1, 3, 64, "fwd", "dw_tiled"),
    ("dw dgrad 3x3 bw1", 1, 96, 96, 20, 20, 3, 3, 1, 1, 96, "dgrad", "dw_dgrad_stream"),
    ("dw dgrad 5x5 s2", 2, 72, 72, 27, 27, 5, 5, 2, 2, 72, "dgrad", "dw_dgrad_stream"),
    ("dw dgrad 7x7 bw4", 32, 96, 96, 28, 28, 7, 7, 1, 3, 96, "dgrad", "dw_dgrad_stream"),
    ("dw dgrad 3x3 s2 bw2", 32, 144, 144, 28, 28, 3, 3, 2, 1, 144, "dgrad", "dw_dgrad_stream"),
    # The grouped direct routes need _GROUPED_DIRECT_MIN_IGEMM_TILES implicit-GEMM
    # output tiles (ceil(N*Ho*Wo/64)*G), so these are sized just past it.
    ("grouped fwd cpg4", 2, 128, 128, 47, 47, 3, 3, 1, 1, 32, "fwd", "grouped"),
    ("grouped fwd cpg16", 2, 256, 256, 65, 65, 3, 3, 1, 1, 16, "fwd", "grouped"),
    ("grouped dgrad cpg4", 2, 128, 128, 47, 47, 3, 3, 1, 1, 32, "dgrad", "grouped_dgrad"),
    ("grouped dgrad cpg16", 2, 64, 64, 129, 129, 3, 3, 1, 1, 4, "dgrad", "grouped_dgrad"),
    ("dw wgrad 3x3 s2", 32, 144, 144, 28, 28, 3, 3, 2, 1, 144, "wgrad", "merged"),
    ("dw wgrad 7x7", 1, 96, 96, 28, 28, 7, 7, 1, 3, 96, "wgrad", "merged"),
    ("stem C3 7x7 s2", 2, 3, 64, 45, 45, 7, 7, 2, 3, 1, "fwd", "channel_pad"),
    # C=1 pads only above _GFX950_STEM_PAD_C1_MAX_UNPADDED_ROWS output pixels.
    ("stem C1 3x3", 2, 1, 32, 157, 157, 3, 3, 1, 1, 1, "fwd", "channel_pad"),
    ("stem C1 3x3 small", 2, 1, 32, 33, 33, 3, 3, 1, 1, 1, "fwd", ""),
]


def _launch(req, outs, tensors, nbytes):
    """Run the dispatched kernel(s) for ``req`` on ``tensors`` (A, B, D)."""
    from dispatch import grouped_convolution as gc
    from kernels.common import conv_direct_grouped as cdg
    from kernels.common.conv_implicit_gemm import build_implicit_gemm_conv
    from kernels.common.conv_implicit_gemm_wgrad import build_implicit_gemm_conv_wgrad
    from kernels.common.conv_implicit_gemm_wgrad_two_stage import (
        build_implicit_gemm_conv_wgrad_two_stage,
    )
    from kernels.common.conv_wgrad_workspace_reduce import (
        _DEFAULT_TILE_M,
        _DEFAULT_TILE_N,
    )
    from rocke import compile_kernel
    from rocke.runtime import synchronize_and_release
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    def launcher(kernel, sig):
        art = compile_kernel(kernel, arch="gfx950")
        return KernelLauncher(
            hsaco=art.hsaco, kernel_name=art.kernel_name, signature=list(sig)
        )

    res = gc.dispatch_conv_grouped(req)
    spec = res.spec
    p = gc._problem(req)
    A, B, D = tensors
    ab, bb, db = nbytes
    ptrs = dict(
        A_ptr=int(A.data_ptr()), B_ptr=int(B.data_ptr()), D_ptr=int(D.data_ptr()),
        A_bytes=ab, B_bytes=bb, D_bytes=db,
    )  # fmt: skip
    cfg = LaunchConfig(grid=tuple(res.grid), block=tuple(res.block))
    route = spec.direct_kind or spec.prepass

    if spec.direct_kind:
        builder = {
            "dw_tiled": cdg.build_direct_depthwise_tiled,
            "dw_dgrad_stream": cdg.build_direct_depthwise_dgrad_streaming,
            "grouped": cdg.build_direct_conv,
            "grouped_dgrad": cdg.build_direct_conv,
        }[spec.direct_kind]
        kernel = builder(spec.to_direct_spec(req), arch="gfx950")
    elif req.direction == "fwd":
        kernel = build_implicit_gemm_conv(spec.to_fwd_spec(p), arch="gfx950")
    else:
        inst = spec.to_wgrad_spec(p)
        route = "merged" if inst.group_merge > 1 else "igemm"
        if inst.two_stage:
            stage1, ws_bytes = build_implicit_gemm_conv_wgrad_two_stage(inst, arch="gfx950")
            ws = torch.zeros(ws_bytes // 4 + 1, dtype=torch.float32, device="cuda")
            wg_m, wg_n = req.K // req.G, req.Y * req.X * (req.C // req.G)
            v1 = gc.launch_values_for(
                req, spec, ws_ptr=int(ws.data_ptr()), ws_bytes=ws_bytes, **ptrs
            )
            v2 = {
                "ws_ptr": int(ws.data_ptr()), "dw_ptr": int(D.data_ptr()),
                "wg_M": wg_m, "wg_N": wg_n, "ws_bytes": ws_bytes, "dw_bytes": db,
                "groups": req.G,
            }  # fmt: skip
            grid2 = (
                -(-wg_n // _DEFAULT_TILE_N),
                -(-wg_m // _DEFAULT_TILE_M),
                req.G,
            )
            stage1(
                (v1, v2),
                (cfg, LaunchConfig(grid=grid2, block=(_DEFAULT_TILE_M * _DEFAULT_TILE_N, 1, 1))),
            )
            torch.cuda.synchronize()
            synchronize_and_release(0)
            outs.append(route)
            return
        kernel = build_implicit_gemm_conv_wgrad(inst, arch="gfx950")
        split_k, _, _ = gc._resolve_wgrad_split_k(spec, p)
        if split_k > 1:
            D.zero_()

    extra = {}
    keep = []
    pre = gc.conv_grouped_prepass(req, spec)
    if pre is not None:
        ws = torch.empty(pre.workspace_bytes // 2 + 64, dtype=torch.int16, device="cuda")
        keep.append(ws)
        extra = dict(ws_ptr=int(ws.data_ptr()), ws_bytes=pre.workspace_bytes)
        pv = gc.prepass_launch_values_for(
            req, spec, A_ptr=ptrs["A_ptr"], B_ptr=ptrs["B_ptr"], A_bytes=ab,
            B_bytes=bb, **extra,
        )  # fmt: skip
        launcher(pre.build("gfx950"), pre.signature).bind(
            pv, config=LaunchConfig(grid=tuple(pre.grid), block=tuple(pre.block))
        )()
    mv = gc.launch_values_for(req, spec, **ptrs, **extra)
    launcher(kernel, res.signature).bind(mv, config=cfg)()
    torch.cuda.synchronize()
    synchronize_and_release(0)
    outs.append(route)


def _reference(direction, x, w, y, stride, pad, groups):
    import torch.nn.functional as F

    x, w, y = (t.float().cpu().permute(0, 3, 1, 2) for t in (x, w, y))
    if direction == "fwd":
        r = F.conv2d(x, w, None, stride, pad, 1, groups)
    elif direction == "dgrad":
        r = torch.nn.grad.conv2d_input(x.shape, w, y, stride, pad, 1, groups)
    else:
        r = torch.nn.grad.conv2d_weight(x, w.shape, y, stride, pad, 1, groups)
    return r.permute(0, 2, 3, 1).contiguous()


@unittest.skipUnless(not _SKIP_REASON, _SKIP_REASON or "no gfx950")
class TestGroupedConvDispatchGpu(unittest.TestCase):
    def _run(self, case, dtype):
        from dispatch.grouped_convolution import ConvGroupedRequest

        label, n, c, k, h, w, kh, kw, s, pd, g, direction, route = case
        req = ConvGroupedRequest(
            N=n, C=c, K=k, Hi=h, Wi=w, Y=kh, X=kw, arch="gfx950", G=g,
            stride_h=s, stride_w=s, pad_h=pd, pad_w=pd, dtype=dtype,
            direction=direction,
        )  # fmt: skip
        ho, wo = (h + 2 * pd - kh) // s + 1, (w + 2 * pd - kw) // s + 1
        td = torch.bfloat16 if dtype == "bf16" else torch.float16
        gen = torch.Generator(device="cuda").manual_seed(7)

        def rand(*shape):
            return torch.empty(*shape, device="cuda", dtype=td).uniform_(-1, 1, generator=gen)

        x, wt, y = rand(n, h, w, c), rand(k, kh, kw, c // g), rand(n, ho, wo, k)
        roles = {"fwd": (x, wt, y), "dgrad": (y, wt, x), "wgrad": (y, x, wt)}[direction]
        out = roles[2]
        out.fill_(float("nan"))
        outs = []
        _launch(req, outs, roles, tuple(t.numel() * 2 for t in roles))
        self.assertEqual(outs, [route], label)
        ref = _reference(direction, x, wt, y, s, pd, g)
        got = out.float().cpu()
        self.assertFalse(bool(torch.isnan(got).any()), f"{label}: NaN in output")
        err = float((got - ref).abs().max()) / max(float(ref.abs().max()), 1.0)
        self.assertLess(err, _TOL[dtype], f"{label} {dtype}: rel err {err:.3e}")

    def test_dispatched_routes(self):
        for case in _CASES:
            for dtype in ("fp16", "bf16"):
                with self.subTest(case=case[0], dtype=dtype):
                    self._run(case, dtype)


if __name__ == "__main__":
    unittest.main(verbosity=2)
