# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""GDN decode dispatch wiring: the ``problem -> spec`` direction.

CPU-only by construction -- it builds specs and reads the registry, never
compiling or launching. That is deliberate: a family whose only tests need a
GPU contributes nothing on a CPU CI machine, so the selection logic is covered
here and the numeric behaviour is covered separately by the on-device test.
"""

from __future__ import annotations

import re
import unittest
from dataclasses import asdict, replace

from dispatch.gdn import (
    GDN_REGISTRY,
    GdnDecodeRequest,
    dispatch_gdn_decode,
    gdn_candidates,
    gdn_sweep_space,
    request_errors,
)
from dispatch.gdn.gfx950 import (
    ARCH,
    CONFIGURED_TILES,
    DEFAULT_TILE,
    KDA_DEFAULT_TILE,
    KDA_DEFAULT_TILE_F32,
    make_spec,
)
from kernels.gfx950.gdn_decode import (
    gdn_decode_grid,
    gdn_decode_signature,
    is_valid_spec,
)

_TILE = lambda s: (s.num_warps, s.warp_threads_k, s.blocks_per_v_dim)  # noqa: E731


def _req(batch: int, **kw) -> GdnDecodeRequest:
    kw.setdefault("arch", ARCH)
    return GdnDecodeRequest(batch=batch, **kw)


class TestRegistration(unittest.TestCase):
    def test_every_configured_tile_is_registered(self):
        names = {
            c.spec_id for c in gdn_candidates() if not c.spec_id.startswith("kda_")
        }
        expected = {f"nw{nw}_wtk{wtk}_bpv{bpv}" for nw, wtk, bpv in CONFIGURED_TILES}
        self.assertEqual(names, expected)

    def test_registry_family_is_consistent(self):
        for cand in gdn_candidates():
            self.assertEqual(cand.family, GDN_REGISTRY.family)


class TestStaticSelection(unittest.TestCase):
    """Dispatcher auto must use one static default rather than batch winners.

    The shipped tile is pinned as a literal, not as ``DEFAULT_TILE``: a test
    that follows the constant cannot notice the default moving. Changing a
    pinned value here changes what every gfx950 GDN ``auto`` user runs, so it
    needs GDN measurements (``tune.py --gate-kind gdn``) in the same change.
    """

    _SHIPPED_TILE = (2, 16, 8)
    # Known limitation (see CONFIGURED_TILES): an illegal default falls back to
    # the first legal tile in product order. Pinned so a change is deliberate.
    _FALLBACK_TILE = (1, 1, 1)

    def test_default_tile_is_the_shipped_tile(self):
        self.assertEqual(DEFAULT_TILE, self._SHIPPED_TILE)

    def test_selection_is_frozen_across_head_counts_and_batches(self):
        # Sharded-head deployments see head counts other than the default
        # geometry, so cover them, including Hk == Hv.
        for num_k_heads, num_v_heads in (
            (2, 4),
            (4, 8),
            (8, 16),
            (16, 32),
            (32, 64),
            (16, 16),
        ):
            for batch in (1, 4, 5, 16, 32, 33, 64, 128, 129, 256):
                with self.subTest(hk=num_k_heads, hv=num_v_heads, batch=batch):
                    result = dispatch_gdn_decode(
                        _req(batch, num_k_heads=num_k_heads, num_v_heads=num_v_heads)
                    )
                    self.assertEqual(_TILE(result.spec), self._SHIPPED_TILE)
                    self.assertEqual(result.spec.gate_kind, "gdn")

    def test_selected_spec_is_always_buildable(self):
        for batch in (1, 4, 5, 16, 33, 64, 129, 256, 8192):
            with self.subTest(batch=batch):
                ok, why = is_valid_spec(
                    dispatch_gdn_decode(_req(batch)).spec, arch=ARCH
                )
                self.assertTrue(ok, why)

    def test_supported_geometry_falls_back_when_default_is_invalid(self):
        for head_k_dim in (64, 192):
            for batch in (1, 256):
                with self.subTest(head_k_dim=head_k_dim, batch=batch):
                    result = dispatch_gdn_decode(_req(batch, head_k_dim=head_k_dim))
                    ok, why = is_valid_spec(result.spec, arch=ARCH)
                    self.assertTrue(ok, why)
                    self.assertEqual(result.spec.head_k_dim, head_k_dim)
                    self.assertEqual(_TILE(result.spec), self._FALLBACK_TILE)


class TestRequestRejection(unittest.TestCase):
    def test_other_arch_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            dispatch_gdn_decode(_req(8, arch="gfx942"))
        self.assertIn("gfx942", str(ctx.exception))

    def test_head_ratio_must_divide(self):
        with self.assertRaises(ValueError) as ctx:
            dispatch_gdn_decode(_req(8, num_v_heads=33))
        self.assertIn("multiple", str(ctx.exception))

    def test_non_positive_batch_is_rejected(self):
        with self.assertRaises(ValueError):
            dispatch_gdn_decode(_req(0))

    def test_unsupported_dtype_is_rejected(self):
        with self.assertRaises(ValueError):
            dispatch_gdn_decode(_req(8, dtype="fp8"))

    def test_kda_d128_is_admitted(self):
        result = dispatch_gdn_decode(
            _req(
                1,
                gate_kind="kda",
                num_k_heads=32,
                num_v_heads=32,
                head_k_dim=128,
                head_v_dim=128,
            )
        )
        self.assertEqual(result.spec.gate_kind, "kda")
        self.assertEqual((result.spec.head_k_dim, result.spec.head_v_dim), (128, 128))

    def test_kda_non_d128_is_loudly_scoped_out(self):
        for head_k_dim, head_v_dim in ((64, 128), (128, 64)):
            with self.subTest(head_k_dim=head_k_dim, head_v_dim=head_v_dim):
                with self.assertRaises(ValueError) as ctx:
                    dispatch_gdn_decode(
                        _req(
                            1,
                            gate_kind="kda",
                            num_k_heads=32,
                            num_v_heads=32,
                            head_k_dim=head_k_dim,
                            head_v_dim=head_v_dim,
                        )
                    )
                self.assertIn("NOT_YET_IMPLEMENTED", str(ctx.exception))
                self.assertIn("128", str(ctx.exception))

    def test_gdn_d64_remains_supported(self):
        result = dispatch_gdn_decode(_req(1, head_k_dim=64))
        self.assertEqual(result.spec.gate_kind, "gdn")
        self.assertEqual(result.spec.head_k_dim, 64)


class TestSpecIdPin(unittest.TestCase):
    def test_pin_selects_the_exact_registered_tile(self):
        result = dispatch_gdn_decode(
            _req(256, algorithm="warp_tiled", spec_id="nw4_wtk16_bpv8")
        )
        self.assertEqual(result.candidate.spec_id, "nw4_wtk16_bpv8")
        self.assertEqual(_TILE(result.spec), (4, 16, 8))

    def test_every_legal_gdn_pin_is_reachable_at_any_batch(self):
        for candidate in gdn_candidates():
            if candidate.spec_id.startswith("kda_"):
                continue
            req = _req(64, algorithm=candidate.algorithm, spec_id=candidate.spec_id)
            if not candidate.admits(req)[0]:
                continue
            with self.subTest(spec_id=candidate.spec_id):
                got = dispatch_gdn_decode(req)
                self.assertEqual(got.candidate.spec_id, candidate.spec_id)

    def test_a_kda_pin_cannot_serve_gdn(self):
        with self.assertRaises(ValueError):
            dispatch_gdn_decode(
                GdnDecodeRequest(
                    batch=64, arch=ARCH, spec_id="kda_nw4_wtk16_bpv4", gate_kind="gdn"
                )
            )

    def test_algorithm_pin_is_honoured_and_an_unknown_one_is_rejected(self):
        # The `algorithm` pin is a separate selector from `spec_id` above, and
        # until this test nothing exercised it: a tuner re-measuring the table,
        # or anyone bisecting a routing regression, forces the algorithm rather
        # than the tile. The family ships one algorithm today, so the case that
        # would silently rot is the REJECTION -- a pin nobody serves must fail
        # loudly instead of falling through to the tuned default.
        got = dispatch_gdn_decode(_req(64, algorithm="warp_tiled"))
        self.assertEqual(got.candidate.algorithm, "warp_tiled")
        with self.assertRaises(ValueError) as ctx:
            dispatch_gdn_decode(_req(64, algorithm="no_such_algorithm"))
        self.assertIn("algorithm", str(ctx.exception))


class TestRegistryContract(unittest.TestCase):
    def test_the_family_refuses_an_unbuildable_candidate(self):
        """`require_build=True` is what stops a candidate being selectable but
        not compilable -- it fails at registration instead of at launch. The
        sibling family asserts this (tests/dispatch/kda); without the assertion
        the flag could be dropped and nothing would go red."""
        self.assertTrue(GDN_REGISTRY.require_build)
        for candidate in gdn_candidates():
            with self.subTest(candidate=candidate.name):
                self.assertIsNotNone(candidate.build)


class TestDtypeCoverage(unittest.TestCase):
    def test_kernel_and_capability_name_the_same_dtypes(self):
        """The dtype set is declared once by the kernel and re-exported by
        dispatch. This pins the CONTENT as well as the sharing: narrowing the
        tuple would otherwise silently shrink coverage -- every test still
        passes, there are just fewer of them -- which is the quiet direction
        the re-export was meant to prevent."""
        from kernels.gfx950.gdn_decode import GDN_DTYPES

        self.assertEqual(set(GDN_DTYPES), {"bf16", "f16"})
        for candidate in gdn_candidates():
            with self.subTest(candidate=candidate.name):
                self.assertEqual(set(candidate.capability.dtypes), set(GDN_DTYPES))
        for dtype in GDN_DTYPES:
            with self.subTest(dtype=dtype):
                got = dispatch_gdn_decode(_req(16, dtype=dtype))
                self.assertEqual(got.spec.dtype, dtype)

    def test_f32_reaches_the_state_but_not_the_io(self):
        """f32 is a state dtype only. Every spelling of it must reach the
        compiled spec for both gate kinds (the kernel name, and so the cache
        key, carries it), and the I/O dtype must still refuse it."""
        from kernels.gfx950.gdn_decode import STATE_DTYPES

        self.assertEqual(set(STATE_DTYPES), {"bf16", "f16", "f32"})
        for gate_kind in ("gdn", "kda"):
            for spelling in ("f32", "fp32", "float32"):
                with self.subTest(gate_kind=gate_kind, spelling=spelling):
                    got = dispatch_gdn_decode(
                        _req(16, gate_kind=gate_kind, state_dtype=spelling)
                    )
                    self.assertEqual(got.spec.state_dtype, "f32")
                    self.assertIn("stf32", got.spec.kernel_name())
        with self.assertRaises(ValueError):
            dispatch_gdn_decode(_req(16, dtype="float32"))


class TestLaunchGeometry(unittest.TestCase):
    def test_grid_and_block_track_the_selected_spec(self):
        for batch in (1, 16, 64, 256):
            with self.subTest(batch=batch):
                got = dispatch_gdn_decode(_req(batch))
                self.assertEqual(got.grid, gdn_decode_grid(batch, got.spec))
                self.assertEqual(got.block, (got.spec.block_size, 1, 1))

    def test_dtype_aliases_normalize(self):
        a = dispatch_gdn_decode(_req(16, dtype="bfloat16"))
        b = dispatch_gdn_decode(_req(16, dtype="bf16"))
        self.assertEqual(a.spec.kernel_name(), b.spec.kernel_name())


class TestKernelIdentity(unittest.TestCase):
    def test_same_request_gives_a_stable_cache_key(self):
        a = dispatch_gdn_decode(_req(16)).kernel_id
        b = dispatch_gdn_decode(_req(16)).kernel_id
        self.assertEqual(a.spec_hash, b.spec_hash)
        self.assertEqual(a.compile_key, b.compile_key)

    def test_different_tiles_do_not_share_a_cache_key(self):
        seen = set()
        for spec_id in ("nw1_wtk8_bpv1", "nw2_wtk8_bpv2", "nw4_wtk16_bpv8"):
            result = dispatch_gdn_decode(
                _req(16, algorithm="warp_tiled", spec_id=spec_id)
            )
            self.assertNotIn(result.kernel_id.compile_key, seen)
            seen.add(result.kernel_id.compile_key)

    def test_spec_hash_covers_the_tile(self):
        from rocke.dispatch.core import stable_json_hash

        a = dispatch_gdn_decode(
            _req(16, algorithm="warp_tiled", spec_id="nw1_wtk8_bpv1")
        ).spec
        b = dispatch_gdn_decode(
            _req(16, algorithm="warp_tiled", spec_id="nw2_wtk8_bpv2")
        ).spec
        self.assertNotEqual(
            stable_json_hash(asdict(a), n=16), stable_json_hash(asdict(b), n=16)
        )


class TestSweepSpace(unittest.TestCase):
    def test_sweep_space_is_non_empty_and_valid(self):
        specs = gdn_sweep_space(_req(16))
        self.assertTrue(specs)
        for spec in specs:
            ok, why = is_valid_spec(spec, arch=ARCH)
            self.assertTrue(ok, why)

    def test_sweep_space_of_a_bad_request_is_empty(self):
        self.assertEqual(gdn_sweep_space(_req(8, num_v_heads=33)), ())


class TestDispatchResultContract(unittest.TestCase):
    """The result must be sufficient to drive a launch on its own.

    A caller should not need to reach back into the kernel module for the
    signature or the grid; if the result disagrees with the spec it carries,
    kernel arguments would be packed against one layout and the kernel compiled
    against another.
    """

    def test_build_returns_the_kernel_the_spec_names(self):
        for batch in (1, 16, 64, 256):
            with self.subTest(batch=batch):
                result = dispatch_gdn_decode(_req(batch))
                kernel = result.build()
                self.assertEqual(kernel.name, result.spec.kernel_name())

    def test_signature_matches_the_spec(self):
        for batch in (1, 16, 64, 256):
            with self.subTest(batch=batch):
                result = dispatch_gdn_decode(_req(batch))
                self.assertEqual(
                    tuple(result.signature),
                    tuple(gdn_decode_signature(result.spec)),
                )

    def test_compile_key_names_arch_and_abi(self):
        kid = dispatch_gdn_decode(_req(16)).kernel_id
        self.assertIn(ARCH, kid.compile_key)
        self.assertIn("rocke-gdn-decode", kid.compile_key)


if __name__ == "__main__":
    unittest.main()


class TestGateKindWiring(unittest.TestCase):
    """The request carries gate_kind through to the spec, GDN by default."""

    def test_request_defaults_to_the_gdn_gate(self):
        self.assertEqual(_req(1).gate_kind, "gdn")
        self.assertEqual(dispatch_gdn_decode(_req(1)).spec.gate_kind, "gdn")

    def test_kda_request_selects_a_kda_spec(self):
        req = GdnDecodeRequest(batch=4, arch=ARCH, gate_kind="kda")
        spec = dispatch_gdn_decode(req).spec
        self.assertEqual(spec.gate_kind, "kda")
        self.assertIn("kda", spec.kernel_name())

    def test_gate_kind_reaches_the_compile_key(self):
        # Two requests differing only in gate_kind select different kernels, so
        # they must not collapse onto one compile-cache entry.
        gdn = dispatch_gdn_decode(GdnDecodeRequest(batch=4, arch=ARCH))
        kda = dispatch_gdn_decode(GdnDecodeRequest(batch=4, arch=ARCH, gate_kind="kda"))
        self.assertNotEqual(gdn.kernel_id.compile_key, kda.kernel_id.compile_key)

    def test_unknown_gate_kind_is_rejected(self):
        errors = request_errors(GdnDecodeRequest(batch=1, arch=ARCH, gate_kind="mamba"))
        self.assertTrue(any("gate_kind" in e for e in errors), errors)


class TestKdaStaticSelection(unittest.TestCase):
    """KDA auto must use one static default per state width, not a work table.

    The shipped tiles are pinned as literals, not as ``KDA_DEFAULT_TILE`` /
    ``KDA_DEFAULT_TILE_F32``: a test that follows the constant cannot notice
    the default moving. Changing one changes what every gfx950 KDA ``auto``
    user with that state width runs, so it needs KDA measurements
    (``tune.py --gate-kind kda``) in the same change.
    """

    _SHIPPED_TILE = (4, 16, 4)
    _SHIPPED_TILE_F32 = (8, 16, 4)
    # state dtype spelling -> (normalized state dtype, tile, carrying spec id)
    _EXPECTED = {
        "bf16": ("bf16", (4, 16, 4), "kda_nw4_wtk16_bpv4"),
        "f16": ("f16", (4, 16, 4), "kda_nw4_wtk16_bpv4"),
        "f32": ("f32", (8, 16, 4), "kda_nw8_wtk16_bpv4"),
        "fp32": ("f32", (8, 16, 4), "kda_nw8_wtk16_bpv4"),
    }

    def test_default_tiles_are_the_shipped_tiles(self):
        self.assertEqual(KDA_DEFAULT_TILE, self._SHIPPED_TILE)
        self.assertEqual(KDA_DEFAULT_TILE_F32, self._SHIPPED_TILE_F32)

    def test_selection_is_frozen_across_head_counts_and_batches(self):
        # Tensor-parallel sharding changes the local head count, and the old
        # work-keyed selection moved with it; the static defaults must not.
        for num_k_heads, num_v_heads in (
            (4, 4),
            (8, 8),
            (16, 16),
            (32, 32),
            (64, 64),
            (8, 32),
            (4, 8),
            (16, 64),
        ):
            for batch in (1, 4, 5, 8, 32, 33, 64, 128, 129, 256, 4096):
                for state_dtype, (norm, tile, spec_id) in self._EXPECTED.items():
                    with self.subTest(
                        hk=num_k_heads,
                        hv=num_v_heads,
                        batch=batch,
                        state_dtype=state_dtype,
                    ):
                        result = dispatch_gdn_decode(
                            GdnDecodeRequest(
                                batch=batch,
                                arch=ARCH,
                                gate_kind="kda",
                                num_k_heads=num_k_heads,
                                num_v_heads=num_v_heads,
                                state_dtype=state_dtype,
                            )
                        )
                        self.assertEqual(_TILE(result.spec), tile)
                        self.assertEqual(result.candidate.spec_id, spec_id)
                        self.assertEqual(result.spec.state_dtype, norm)
                        self.assertEqual(result.spec.gate_kind, "kda")


def _literal_tile(spec_id: str):
    """Parse ``[kda_]nw{}_wtk{}_bpv{}`` independently of dispatch's formatter."""
    match = re.fullmatch(r"(?:kda_)?nw(\d+)_wtk(\d+)_bpv(\d+)", spec_id)
    assert match, f"unexpected spec id {spec_id!r}"
    return tuple(int(value) for value in match.groups())


class TestKdaRegistry(unittest.TestCase):
    """KDA registers GDN's configured tile space; every legal tile is pinnable."""

    def test_kda_registers_every_configured_tile_once(self):
        kda = [c for c in gdn_candidates() if c.spec_id.startswith("kda_")]
        self.assertEqual(len(kda), len(CONFIGURED_TILES))
        self.assertEqual(
            [_literal_tile(c.spec_id) for c in kda], list(CONFIGURED_TILES)
        )

    def test_every_legal_kda_pin_selects_its_literal_tile(self):
        for state_dtype in ("bf16", "f32"):
            base = GdnDecodeRequest(
                batch=16,
                arch=ARCH,
                gate_kind="kda",
                num_k_heads=16,
                num_v_heads=16,
                state_dtype=state_dtype,
            )
            legal = 0
            for candidate in gdn_candidates():
                if not candidate.spec_id.startswith("kda_"):
                    continue
                req = replace(base, spec_id=candidate.spec_id)
                tile = _literal_tile(candidate.spec_id)
                if not is_valid_spec(make_spec(base, tile), arch=ARCH)[0]:
                    with self.assertRaises(ValueError):
                        dispatch_gdn_decode(req)
                    continue
                legal += 1
                for batch in (1, 256):
                    with self.subTest(
                        spec_id=candidate.spec_id, state=state_dtype, batch=batch
                    ):
                        result = dispatch_gdn_decode(replace(req, batch=batch))
                        self.assertEqual(result.candidate.spec_id, candidate.spec_id)
                        self.assertEqual(_TILE(result.spec), tile)
                        self.assertEqual(result.spec.gate_kind, "kda")
                        self.assertEqual(result.spec.state_dtype, state_dtype)
            self.assertGreater(legal, 0)

    def test_a_gdn_pin_cannot_serve_kda(self):
        with self.assertRaises(ValueError):
            dispatch_gdn_decode(
                GdnDecodeRequest(
                    batch=64, arch=ARCH, spec_id="nw4_wtk16_bpv4", gate_kind="kda"
                )
            )


class TestGdnAndKdaTileNamespaces(unittest.TestCase):
    """The shared registry keeps GDN and KDA selectable identities disjoint."""

    def test_spec_ids_and_names_are_unique_across_gate_kinds(self):
        candidates = gdn_candidates()
        self.assertEqual(len(candidates), 2 * len(CONFIGURED_TILES))
        self.assertEqual(
            len({c.spec_id for c in candidates}), 2 * len(CONFIGURED_TILES)
        )
        self.assertEqual(len({c.name for c in candidates}), 2 * len(CONFIGURED_TILES))


class TestFusedDispatch(unittest.TestCase):
    """fuse_conv / fuse_out_norm requests: BPV=1 tiles only, static fused
    defaults, Hk == Hv for conv, and no change to unfused selection."""

    @staticmethod
    def req(**kw):
        base = dict(
            batch=8,
            arch=ARCH,
            num_k_heads=16,
            num_v_heads=16,
            gate_kind="kda",
            fuse_conv=True,
            fuse_out_norm=True,
        )
        base.update(kw)
        return GdnDecodeRequest(**base)

    @staticmethod
    def tile(spec):
        return (spec.num_warps, spec.warp_threads_k, spec.blocks_per_v_dim)

    def test_auto_is_fused_default(self):
        from dispatch.gdn.gfx950 import FUSED_DEFAULT_TILES

        for gate in ("gdn", "kda"):
            for st in ("bf16", "f32"):
                for batch in (1, 8, 128, 4096):
                    with self.subTest(gate=gate, st=st, batch=batch):
                        s = dispatch_gdn_decode(
                            self.req(gate_kind=gate, state_dtype=st, batch=batch)
                        ).spec
                        self.assertEqual(self.tile(s), FUSED_DEFAULT_TILES[(gate, st)])
                        self.assertTrue(s.fuse_conv and s.fuse_out_norm)

    def test_fused_defaults_are_bpv1_and_configured(self):
        from dispatch.gdn.gfx950 import FUSED_DEFAULT_TILES

        for key, t in FUSED_DEFAULT_TILES.items():
            with self.subTest(key=key):
                self.assertIn(t, CONFIGURED_TILES)
                self.assertEqual(t[2], 1)

    def test_bpv_gt1_pin_rejected(self):
        with self.assertRaises(ValueError):
            dispatch_gdn_decode(self.req(spec_id="kda_nw4_wtk16_bpv4"))

    def test_bpv1_pin_accepted(self):
        s = dispatch_gdn_decode(self.req(spec_id="kda_nw2_wtk16_bpv1")).spec
        self.assertEqual(self.tile(s), (2, 16, 1))

    def test_conv_gqa_rejected(self):
        r = self.req(
            gate_kind="gdn", num_k_heads=8, num_v_heads=16, fuse_out_norm=False
        )
        self.assertTrue(any("fuse_conv" in e for e in request_errors(r)))
        with self.assertRaises(ValueError):
            dispatch_gdn_decode(r)

    def test_norm_gqa_accepted(self):
        s = dispatch_gdn_decode(
            self.req(gate_kind="gdn", num_k_heads=8, num_v_heads=16, fuse_conv=False)
        ).spec
        self.assertTrue(s.fuse_out_norm)
        self.assertFalse(s.fuse_conv)
        self.assertEqual(s.blocks_per_v_dim, 1)

    def test_unfused_auto_unchanged(self):
        s = dispatch_gdn_decode(self.req(fuse_conv=False, fuse_out_norm=False)).spec
        self.assertEqual(self.tile(s), KDA_DEFAULT_TILE)
        g = dispatch_gdn_decode(
            self.req(
                gate_kind="gdn", num_v_heads=32, fuse_conv=False, fuse_out_norm=False
            )
        ).spec
        self.assertEqual(self.tile(g), DEFAULT_TILE)
