# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-side checks for the gfx950 grouped-conv direct routes (no GPU).

* Depthwise forward -> the output-stationary tiled kernel, with the row tile
  sized to the grid.
* Depthwise dgrad -> the streaming kernel, with the strip width sized to the
  grid.
* Grouped forward, cpg == kpg in 4..16 -> the direct MFMA kernel once the
  implicit GEMM would launch enough tiles; wider groups stay on the GEMM.
"""

from __future__ import annotations

import unittest


def _req(**kw):
    from dispatch.grouped_convolution import ConvGroupedRequest

    d = dict(
        N=32, C=144, K=144, Hi=56, Wi=56, Y=3, X=3, arch="gfx950", G=144,
        stride_h=1, stride_w=1, pad_h=1, pad_w=1, dtype="bf16", direction="fwd",
    )  # fmt: skip
    d.update(kw)
    return ConvGroupedRequest(**d)


def _spec(**kw):
    from dispatch.grouped_convolution import dispatch_conv_grouped

    return dispatch_conv_grouped(_req(**kw)).spec


class TestDepthwiseForward(unittest.TestCase):
    def test_routes_to_the_tiled_kernel(self):
        for dtype in ("fp16", "bf16"):
            self.assertEqual(_spec(dtype=dtype).direct_kind, "dw_tiled")

    def test_row_tile_follows_the_grid(self):
        from dispatch.grouped_convolution import _DW_TILED_MIN_CTAS

        # (request, expected (block_w, block_h, waves))
        cases = [
            # Large grids keep the tallest tile: 7 rows for 5x5+/big stride-1
            # maps, else 4.
            (dict(N=32, Y=7, X=7, pad_h=3, pad_w=3, C=96, K=96, G=96), (7, 7, 1)),
            (dict(N=32, stride_h=2, stride_w=2), (7, 4, 1)),
            # 7x7 C1024 at 7x7: 16 strips per image; at N=32, 7 rows -> 512
            # CTAs, 4 rows -> 1024 CTAs.
            (dict(N=32, Hi=7, Wi=7, Y=7, X=7, pad_h=3, pad_w=3, C=1024, K=1024, G=1024), (7, 4, 1)),
            # Small grids drop to 1 row.
            (dict(N=1, Hi=14, Wi=14, Y=5, X=5, pad_h=2, pad_w=2, stride_h=2, stride_w=2, C=672, K=672, G=672), (7, 1, 1)),
        ]  # fmt: skip
        for kw, want in cases:
            with self.subTest(kw=kw):
                self.assertEqual(tuple(_spec(**kw).direct_tile), want)
        self.assertGreater(_DW_TILED_MIN_CTAS, 0)


class TestDepthwiseDgrad(unittest.TestCase):
    def test_strip_width_follows_the_grid(self):
        cases = [
            (dict(N=1), (1, 1)),  # 14 * 3 strips: under one CTA per CU
            (dict(N=32, Hi=28, Wi=28, stride_h=2, stride_w=2), (2, 1)),
            (dict(N=32), (4, 1)),
        ]
        for kw, want in cases:
            with self.subTest(kw=kw):
                spec = _spec(direction="dgrad", **kw)
                self.assertEqual(spec.direct_kind, "dw_dgrad_stream")
                self.assertEqual(tuple(spec.direct_tile), want)


class TestGroupedForward(unittest.TestCase):
    def test_small_cpg_large_work_takes_the_direct_route(self):
        for c, g in ((128, 32), (256, 32), (256, 16)):
            spec = _spec(C=c, K=c, G=g)
            self.assertEqual(spec.direct_kind, "grouped", (c, g))

    def test_wide_groups_and_small_work_stay_on_the_igemm(self):
        for kw in (dict(C=1024, K=1024, G=32), dict(N=1, Hi=28, Wi=28, C=128, K=128, G=32)):
            self.assertEqual(_spec(**kw).direct_kind, "", kw)


if __name__ == "__main__":
    unittest.main()
