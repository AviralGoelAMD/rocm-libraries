# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx942 GDN decode wiring: bands select, and every selection is buildable.

The gfx942 tile table is still provisional, so these tests pin the *wiring*
(arch gate, band -> tile -> spec agreement, buildability) rather than any
particular measured tile.
"""

from __future__ import annotations

import unittest

from dispatch.gdn import GdnDecodeRequest, dispatch_gdn_decode
from dispatch.gdn.gfx942 import ARCH, spec_id_for_batch, tile_for_batch
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

    def test_tile_for_batch_agrees_with_dispatch(self):
        for batch in (1, 2, 5, 17, 63, 100, 200, 4096):
            with self.subTest(batch=batch):
                spec = dispatch_gdn_decode(_req(batch)).spec
                self.assertEqual(_TILE(spec), tile_for_batch(batch))

    def test_spec_id_matches_the_named_band(self):
        for batch in (1, 4, 5, 32, 33, 128, 129):
            with self.subTest(batch=batch):
                self.assertEqual(
                    dispatch_gdn_decode(_req(batch)).candidate.spec_id,
                    spec_id_for_batch(batch),
                )

    def test_gfx950_and_gfx942_do_not_serve_each_other(self):
        """Both arches are registered now, so the arch gate is the only thing
        keeping them apart. Prove it in both directions."""
        self.assertIn("gfx942", dispatch_gdn_decode(_req(8)).candidate.name)
        self.assertIn(
            "gfx950", dispatch_gdn_decode(_req(8, arch="gfx950")).candidate.name
        )


if __name__ == "__main__":
    unittest.main()
