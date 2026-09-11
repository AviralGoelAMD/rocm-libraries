# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx942 GDN decode wiring: bands select, and every selection is buildable.

These tests pin the *wiring* -- arch gate, band -> tile -> spec agreement,
buildability -- and deliberately not any particular tile value. The tile table
is measured and will be re-measured; the wiring must hold whatever it says.
"""

from __future__ import annotations

import unittest

from dispatch.gdn import GdnDecodeRequest, dispatch_gdn_decode
from dispatch.gdn.gfx942 import ARCH, spec_id_for_work, tile_for_work, work_for
from kernels.common.gdn_decode import is_valid_spec

_TILE = lambda s: (s.num_warps, s.warp_threads_k, s.blocks_per_v_dim)  # noqa: E731


def _req(batch: int, **kw) -> GdnDecodeRequest:
    kw.setdefault("arch", ARCH)
    return GdnDecodeRequest(batch=batch, **kw)


class TestGfx942Dispatches(unittest.TestCase):
    def test_gfx942_request_is_accepted(self):
        result = dispatch_gdn_decode(_req(8))
        self.assertIn("gfx942", result.candidate.name)

    def test_selected_spec_is_always_buildable(self):
        for batch in (1, 4, 5, 8, 16, 32, 33, 64, 129, 256, 8192):
            with self.subTest(batch=batch):
                ok, why = is_valid_spec(dispatch_gdn_decode(_req(batch)).spec, arch=ARCH)
                self.assertTrue(ok, why)

    def test_tile_agrees_with_dispatch(self):
        for batch in (1, 2, 5, 17, 63, 100, 200, 4096):
            with self.subTest(batch=batch):
                spec = dispatch_gdn_decode(_req(batch)).spec
                work = work_for(batch, spec.num_v_heads)
                self.assertEqual(_TILE(spec), tile_for_work(work))

    def test_spec_id_matches_the_named_band(self):
        for batch in (1, 4, 5, 32, 33, 128, 129):
            with self.subTest(batch=batch):
                res = dispatch_gdn_decode(_req(batch))
                work = work_for(batch, res.spec.num_v_heads)
                self.assertEqual(res.candidate.spec_id, spec_id_for_work(work))

    def test_head_count_changes_the_tile_at_fixed_batch(self):
        """The point of keying on work rather than batch. Tensor-parallel
        sharding divides num_v_heads across ranks, so the same batch on a
        4-head shard has an eighth of the parallelism of a 32-head one and
        must not be served the same tile."""
        wide = dispatch_gdn_decode(_req(8, num_k_heads=16, num_v_heads=32))
        narrow = dispatch_gdn_decode(_req(8, num_k_heads=2, num_v_heads=4))
        self.assertNotEqual(
            _TILE(wide.spec),
            _TILE(narrow.spec),
            "batch 8 x 32 heads (work 256) and batch 8 x 4 heads (work 32) "
            "landed on the same tile; the table is behaving as if keyed on "
            "batch alone",
        )

    def test_equal_work_gets_the_same_tile(self):
        """Measured invariant: cost depends on batch * num_v_heads, not on the
        two separately (work=256 cells agreed to 2.0%, work=1024 to 0.2%)."""
        for (b1, hk1, hv1), (b2, hk2, hv2) in (
            ((8, 16, 32), (64, 2, 4)),      # work 256 both
            ((16, 16, 32), (128, 2, 4)),    # work 512 both
            ((8, 8, 16), (16, 4, 8)),       # work 128 both
        ):
            with self.subTest(pair=((b1, hv1), (b2, hv2))):
                a = dispatch_gdn_decode(_req(b1, num_k_heads=hk1, num_v_heads=hv1))
                b = dispatch_gdn_decode(_req(b2, num_k_heads=hk2, num_v_heads=hv2))
                self.assertEqual(work_for(b1, hv1), work_for(b2, hv2))
                self.assertEqual(_TILE(a.spec), _TILE(b.spec))

    def test_gfx950_and_gfx942_do_not_serve_each_other(self):
        """Both arches are registered now, so the arch gate is the only thing
        keeping them apart. Prove it in both directions."""
        self.assertIn("gfx942", dispatch_gdn_decode(_req(8)).candidate.name)
        self.assertIn(
            "gfx950", dispatch_gdn_decode(_req(8, arch="gfx950")).candidate.name
        )


if __name__ == "__main__":
    unittest.main()
