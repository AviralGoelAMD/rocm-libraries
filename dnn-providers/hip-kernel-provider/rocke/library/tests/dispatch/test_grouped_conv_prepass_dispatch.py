# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-side checks for the grouped-conv pre-pass routes (no GPU).

* Small-C stem forward (C odd, < 8, G == 1) -> channel-pad pre-pass + the C=8
  implicit GEMM.
* Grouped dgrad, stride 1, cpg == kpg in 4..16 -> weight-transpose pre-pass +
  the forward direct kernel on the transposed problem.
"""

from __future__ import annotations

import unittest


def _req(**kw):
    from dispatch.grouped_convolution import ConvGroupedRequest

    d = dict(
        N=4, C=3, K=64, Hi=32, Wi=32, Y=3, X=3, arch="gfx950", G=1,
        stride_h=1, stride_w=1, pad_h=1, pad_w=1, dtype="bf16", direction="fwd",
    )
    d.update(kw)
    return ConvGroupedRequest(**d)


class TestStemChannelPad(unittest.TestCase):
    def test_odd_small_c_takes_the_pad(self):
        from dispatch.grouped_convolution import dispatch_conv_grouped

        for c in (1, 3, 5, 7):
            spec = dispatch_conv_grouped(_req(C=c)).spec
            self.assertEqual(spec.pad_c, 8, c)
            self.assertEqual(spec.prepass, "channel_pad")
            self.assertIn("padc8", spec.kernel_name())

    def test_other_shapes_do_not(self):
        from dispatch.grouped_convolution import dispatch_conv_grouped

        for kw in (dict(C=8), dict(C=4), dict(C=16), dict(C=6, K=6, G=3)):
            spec = dispatch_conv_grouped(_req(**kw)).spec
            self.assertEqual(spec.pad_c, 0, kw)
            self.assertEqual(spec.prepass, "", kw)

    def test_values_point_the_conv_at_the_workspace(self):
        from dispatch.grouped_convolution import (
            conv_grouped_prepass,
            dispatch_conv_grouped,
            launch_values_for,
            prepass_launch_values_for,
        )

        req = _req()
        spec = dispatch_conv_grouped(req).spec
        pre = conv_grouped_prepass(req, spec)
        x_rows, w_rows = 4 * 32 * 32, 64 * 3 * 3
        self.assertEqual(pre.workspace_bytes, (x_rows + w_rows) * 8 * 2)
        ws = 1 << 30
        pv = prepass_launch_values_for(
            req, spec, A_ptr=1, B_ptr=2, A_bytes=0, B_bytes=0,
            ws_ptr=ws, ws_bytes=pre.workspace_bytes,
        )
        self.assertEqual((pv["X"], pv["W"], pv["Xp"]), (1, 2, ws))
        self.assertEqual(pv["Wp"], ws + x_rows * 16)
        mv = launch_values_for(
            req, spec, A_ptr=1, B_ptr=2, D_ptr=3, A_bytes=0, B_bytes=0, D_bytes=0,
            ws_ptr=ws, ws_bytes=pre.workspace_bytes,
        )
        self.assertEqual((mv["A"], mv["B"], mv["D"]), (ws, ws + x_rows * 16, 3))
        with self.assertRaises(ValueError):
            launch_values_for(
                req, spec, A_ptr=1, B_ptr=2, D_ptr=3, A_bytes=0, B_bytes=0, D_bytes=0
            )

    def test_kernel_is_the_c8_problem(self):
        from dispatch.grouped_convolution import _problem, dispatch_conv_grouped

        req = _req()
        spec = dispatch_conv_grouped(req).spec
        self.assertEqual(spec.to_fwd_spec(_problem(req)).problem.C, 8)

    def test_pad_kernel_builds(self):
        from kernels.common.conv_channel_pad import (
            ChannelPadSpec,
            build_conv_channel_pad,
            is_valid_channel_pad_spec,
        )

        self.assertTrue(is_valid_channel_pad_spec(ChannelPadSpec(3, 8))[0])
        self.assertFalse(is_valid_channel_pad_spec(ChannelPadSpec(8, 8))[0])
        self.assertIsNotNone(build_conv_channel_pad(ChannelPadSpec(3, 8, "fp16")))


class TestGroupedDgradTranspose(unittest.TestCase):
    def _dg(self, **kw):
        d = dict(N=32, C=128, K=128, Hi=28, Wi=28, G=32, direction="dgrad")
        d.update(kw)
        return _req(**d)

    def test_small_cpg_stride1_takes_the_direct_route(self):
        from dispatch.grouped_convolution import dispatch_conv_grouped

        for c, g in ((128, 32), (256, 32), (512, 32)):
            spec = dispatch_conv_grouped(self._dg(C=c, K=c, G=g)).spec
            self.assertEqual(spec.direct_kind, "grouped_dgrad", (c, g))
            self.assertEqual(spec.prepass, "weight_transpose")

    def test_stride2_and_cpg32_stay_on_the_igemm(self):
        from dispatch.grouped_convolution import dispatch_conv_grouped

        for kw in (dict(stride_h=2, stride_w=2), dict(C=1024, K=1024)):
            spec = dispatch_conv_grouped(self._dg(**kw)).spec
            self.assertEqual(spec.direct_kind, "", kw)

    def test_small_work_stays_on_the_igemm(self):
        """Below _GROUPED_DIRECT_MIN_IGEMM_TILES implicit-GEMM tiles both
        grouped direct routes decline; at or above it they take the request."""
        from dispatch.grouped_convolution import (
            _GROUPED_DIRECT_MIN_IGEMM_TILES,
            dispatch_conv_grouped,
        )

        def tiles(n):  # 64-pixel output tiles over 32 groups of 28x28
            return -(-(n * 28 * 28) // 64) * 32

        for direction, kind in (("dgrad", "grouped_dgrad"), ("fwd", "grouped")):
            for n in (1, 2, 4, 5, 8, 32):
                want = kind if tiles(n) >= _GROUPED_DIRECT_MIN_IGEMM_TILES else ""
                spec = dispatch_conv_grouped(self._dg(N=n, direction=direction)).spec
                self.assertEqual(spec.direct_kind, want, (direction, n))
            self.assertEqual(
                dispatch_conv_grouped(self._dg(N=1, direction=direction)).spec.direct_kind,
                "",
            )

    def test_values_read_the_transposed_filter(self):
        from dispatch.grouped_convolution import (
            conv_grouped_prepass,
            dispatch_conv_grouped,
            launch_values_for,
            prepass_launch_values_for,
        )

        req = self._dg()
        res = dispatch_conv_grouped(req)
        pre = conv_grouped_prepass(req, res.spec)
        self.assertEqual(pre.workspace_bytes, 128 * 9 * 4 * 2)
        ws = 1 << 30
        pv = prepass_launch_values_for(
            req, res.spec, A_ptr=1, B_ptr=2, A_bytes=0, B_bytes=0,
            ws_ptr=ws, ws_bytes=pre.workspace_bytes,
        )
        self.assertEqual((pv["A"], pv["D"]), (2, ws))
        mv = launch_values_for(
            req, res.spec, A_ptr=1, B_ptr=2, D_ptr=3, A_bytes=0, B_bytes=0, D_bytes=0,
            ws_ptr=ws, ws_bytes=pre.workspace_bytes,
        )
        self.assertEqual((mv["A"], mv["B"], mv["D"]), (1, ws, 3))
        self.assertEqual(
            [e["name"] for e in res.signature][:3], ["A", "B", "D"]
        )


if __name__ == "__main__":
    unittest.main()
