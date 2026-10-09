# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests for the optional fused conv1d / gated-RMSNorm GDN/KDA decode modes.

Covers the spec contract (fields, kernel-name tags, admission rules) and the
conditional kernel ABI. On-device numerics live in
``test_gdn_decode_fused_gfx950_numeric.py``.
"""

from __future__ import annotations

import dataclasses as dc
import unittest

from kernels.gfx950.gdn_decode import (
    GdnDecodeSpec,
    build_gdn_decode,
    gdn_decode_signature,
    is_valid_spec,
)

ARCH = "gfx950"


def _fused(**kw) -> GdnDecodeSpec:
    """A BPV=1, Hk == Hv KDA spec; flags off unless ``kw`` turns them on."""
    base = dict(
        num_k_heads=16,
        num_v_heads=16,
        gate_kind="kda",
        num_warps=4,
        warp_threads_k=16,
        blocks_per_v_dim=1,
    )
    base.update(kw)
    return GdnDecodeSpec(**base)


class TestFusedSpec(unittest.TestCase):
    def test_default_name_unchanged(self):
        self.assertEqual(
            GdnDecodeSpec().kernel_name(),
            "rocke_gdn_decode_bf16_kh16_vh32_dk128_dv128_w2k16b8_l2",
        )

    def test_name_tags_only_when_on(self):
        s = _fused()
        self.assertFalse(s.kernel_name().endswith(("_cv", "_rn")))
        self.assertTrue(dc.replace(s, fuse_conv=True).kernel_name().endswith("_l2_cv"))
        self.assertTrue(
            dc.replace(s, fuse_out_norm=True).kernel_name().endswith("_l2_rn")
        )
        self.assertTrue(
            dc.replace(s, fuse_conv=True, fuse_out_norm=True)
            .kernel_name()
            .endswith("_l2_cv_rn")
        )

    def test_fused_requires_bpv1(self):
        for flag in ("fuse_conv", "fuse_out_norm"):
            with self.subTest(flag=flag):
                ok, why = is_valid_spec(
                    _fused(blocks_per_v_dim=4, **{flag: True}), arch=ARCH
                )
                self.assertFalse(ok)
                self.assertIn("blocks_per_v_dim", why)

    def test_fused_rejects_simple(self):
        ok, why = is_valid_spec(_fused(simple=True, fuse_out_norm=True), arch=ARCH)
        self.assertFalse(ok)
        self.assertIn("simple", why)

    def test_conv_requires_equal_heads(self):
        ok, why = is_valid_spec(
            _fused(num_k_heads=8, num_v_heads=16, fuse_conv=True), arch=ARCH
        )
        self.assertFalse(ok)
        self.assertIn("num_k_heads == num_v_heads", why)

    def test_norm_allows_gqa(self):
        ok, why = is_valid_spec(
            _fused(num_k_heads=8, num_v_heads=16, gate_kind="gdn", fuse_out_norm=True),
            arch=ARCH,
        )
        self.assertTrue(ok, why)

    def test_fused_bpv1_valid(self):
        for gate in ("gdn", "kda"):
            with self.subTest(gate=gate):
                ok, why = is_valid_spec(
                    _fused(gate_kind=gate, fuse_conv=True, fuse_out_norm=True),
                    arch=ARCH,
                )
                self.assertTrue(ok, why)


class TestFusedSignature(unittest.TestCase):
    @staticmethod
    def names(spec):
        return [p["name"] for p in gdn_decode_signature(spec)]

    def test_unfused_unchanged(self):
        self.assertEqual(
            self.names(GdnDecodeSpec()),
            [
                "query",
                "key",
                "value",
                "a",
                "b",
                "dt_bias",
                "A_log",
                "read_indices",
                "write_indices",
                "state",
                "out",
                "batch_size",
            ],
        )

    def test_conv_and_norm(self):
        s = _fused(fuse_conv=True, fuse_out_norm=True)
        self.assertEqual(
            self.names(s),
            [
                "mixed_qkv",
                "a",
                "b",
                "dt_bias",
                "A_log",
                "read_indices",
                "write_indices",
                "state",
                "out",
                "conv_state",
                "conv_weight",
                "out_gate",
                "norm_weight",
                "batch_size",
                "qkv_stride",
                "og_stride",
                "norm_eps",
            ],
        )

    def test_norm_only(self):
        n = self.names(_fused(fuse_out_norm=True))
        self.assertEqual(n[:3], ["query", "key", "value"])
        self.assertNotIn("conv_state", n)
        self.assertEqual(n[-3:], ["batch_size", "og_stride", "norm_eps"])


def _scratch_bytes(test: unittest.TestCase, spec: GdnDecodeSpec) -> int:
    """Compile through comgr (no GPU) and return the spilled scratch bytes."""
    import tempfile
    from pathlib import Path

    try:
        from rocke.analysis.isa import analyze_hsaco
        from rocke.helpers.compile import compile_kernel
    except Exception as e:  # pragma: no cover - env-dependent
        test.skipTest(f"comgr toolchain unavailable: {e}")
    try:
        art = compile_kernel(
            build_gdn_decode(spec, arch=ARCH), arch=ARCH, capture_ir_text=False
        )
    except ImportError as e:  # pragma: no cover - env-dependent
        test.skipTest(f"comgr toolchain unavailable: {e}")
    with tempfile.NamedTemporaryFile(suffix=".hsaco") as fh:
        fh.write(bytes(art.hsaco))
        fh.flush()
        try:
            scratch = analyze_hsaco(Path(fh.name)).resources.scratch_bytes
        except (FileNotFoundError, RuntimeError) as e:  # pragma: no cover
            test.skipTest(f"HSACO introspection tool unavailable: {e}")
    if scratch is None:  # pragma: no cover - metadata shape drift
        test.skipTest("could not parse the scratch size from the HSACO")
    return scratch


_FLAGS = ((True, False), (False, True), (True, True))


class TestFusedEmission(unittest.TestCase):
    def test_params_match_signature(self):
        for gate in ("gdn", "kda"):
            for conv, norm in _FLAGS:
                for nw in (1, 4):
                    s = _fused(
                        gate_kind=gate, num_warps=nw, fuse_conv=conv, fuse_out_norm=norm
                    )
                    with self.subTest(name=s.kernel_name()):
                        k = build_gdn_decode(s, arch=ARCH)
                        self.assertEqual(
                            [p.name for p in k.params],
                            [e["name"] for e in gdn_decode_signature(s)],
                        )

    def test_lowers_and_compiles_without_scratch(self):
        """Every fused variant lowers and compiles; 0 scratch except one wave
        with fuse_conv, which owns all DV rows plus their taps and weights and
        is a known register-heavy (legal, never-default) tile -- the fused
        twin of the unfused (1,1,1) exemption."""
        from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

        for gate in ("gdn", "kda"):
            for st in ("bf16", "f32"):
                for conv, norm in _FLAGS:
                    for nw in (1, 2, 4, 8):
                        s = _fused(
                            gate_kind=gate,
                            state_dtype=st,
                            num_warps=nw,
                            fuse_conv=conv,
                            fuse_out_norm=norm,
                        )
                        with self.subTest(name=s.kernel_name()):
                            ll = _lower_kernel_to_llvm_python(
                                build_gdn_decode(s, arch=ARCH),
                                arch=ARCH,
                                llvm_flavor="llvm20",
                            )
                            self.assertIn("define amdgpu_kernel", ll)
                            scratch = _scratch_bytes(self, s)
                            if not (conv and nw == 1):
                                self.assertEqual(scratch, 0)

    def test_unfused_kernel_has_no_lds_or_barrier(self):
        from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

        ll = _lower_kernel_to_llvm_python(
            build_gdn_decode(_fused(), arch=ARCH), arch=ARCH, llvm_flavor="llvm20"
        )
        self.assertNotIn("addrspace(3)", ll)
        self.assertNotIn("s.barrier", ll)


class TestFusedReference(unittest.TestCase):
    """Host-side contract of the fused inputs and the fp32 reference (CPU)."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
        except ImportError:  # pragma: no cover - env-dependent
            raise unittest.SkipTest("torch unavailable")

    @staticmethod
    def spec(**kw):
        return _fused(num_k_heads=2, num_v_heads=2, **kw)

    def test_inputs_present(self):
        from builders.gfx950.gdn.gdn_decode import make_inputs

        inp = make_inputs(
            self.spec(fuse_conv=True, fuse_out_norm=True), 3, device="cpu"
        )
        cd = 2 * 2 * 128 + 2 * 128
        self.assertEqual(tuple(inp["mixed_qkv"].shape), (3, cd))
        self.assertEqual(inp["qkv_stride"], cd)
        self.assertEqual(tuple(inp["conv_state"].shape), (7, cd, 3))
        self.assertEqual(tuple(inp["conv_weight"].shape), (cd, 4))
        self.assertEqual(tuple(inp["out_gate"].shape), (3, 2 * 128))
        self.assertEqual(tuple(inp["norm_weight"].shape), (128,))
        self.assertEqual(inp["og_stride"], 2 * 128)
        self.assertEqual(inp["norm_eps"], 1e-6)

    def test_unfused_inputs_unchanged(self):
        import torch

        from builders.gfx950.gdn.gdn_decode import make_inputs

        plain = make_inputs(self.spec(), 3, device="cpu")
        self.assertNotIn("mixed_qkv", plain)
        self.assertNotIn("out_gate", plain)
        # the norm draws come after every unfused draw: shared inputs match
        normed = make_inputs(self.spec(fuse_out_norm=True), 3, device="cpu")
        for name in ("query", "key", "value", "a", "b", "state"):
            self.assertTrue(torch.equal(plain[name], normed[name]), name)

    def test_conv_shift(self):
        import torch

        from builders.gfx950.gdn.gdn_decode import make_inputs, ref_conv_state_after

        s = self.spec(fuse_conv=True)
        inp = make_inputs(s, 3, device="cpu")
        after = ref_conv_state_after(s, inp)
        taps = inp["conv_state"].float()[inp["read_indices"].long()]
        self.assertTrue(torch.equal(after[..., 0], taps[..., 1]))
        self.assertTrue(torch.equal(after[..., 1], taps[..., 2]))
        self.assertTrue(torch.equal(after[..., 2], inp["mixed_qkv"].float()))

    def test_conv_changes_output(self):
        import torch

        from builders.gfx950.gdn.gdn_decode import make_inputs, ref_fp32

        s = self.spec(fuse_conv=True)
        inp = make_inputs(s, 3, device="cpu")
        o_fused, _ = ref_fp32(s, inp)
        o_plain, _ = ref_fp32(dc.replace(s, fuse_conv=False), inp)
        self.assertFalse(torch.allclose(o_fused, o_plain))

    def test_norm_gives_unit_rms(self):
        import torch

        from builders.gfx950.gdn.gdn_decode import make_inputs, ref_fp32

        s = self.spec(fuse_out_norm=True)
        inp = make_inputs(s, 3, device="cpu")
        inp["out_gate"] = torch.full_like(inp["out_gate"], 30.0)  # sigmoid -> 1
        inp["norm_weight"] = torch.ones_like(inp["norm_weight"])
        # The raw decode output's RMS is ~1e-4..1e-2 here, so eps must be far
        # below mean(o^2) for the normalised RMS to reach 1.
        inp["norm_eps"] = 1e-12
        o, _ = ref_fp32(s, inp)
        rms = o.float().pow(2).mean(-1).sqrt()
        self.assertTrue(torch.allclose(rms, torch.ones_like(rms), atol=1e-3))

    def test_out_error_accepts_bf16_rounding_of_the_reference(self):
        """The fused output is O(1..7): bf16 rounding of the exact reference
        must pass, in every fused mode."""
        from builders.gfx950.gdn.gdn_decode import TOL, make_inputs, out_error, ref_fp32

        for gate in ("gdn", "kda"):
            for conv, norm in _FLAGS:
                s = self.spec(gate_kind=gate, fuse_conv=conv, fuse_out_norm=norm)
                with self.subTest(name=s.kernel_name()):
                    ref, _ = ref_fp32(s, make_inputs(s, 16, device="cpu"))
                    self.assertLess(out_error(s, ref.bfloat16(), ref), TOL)

    def test_out_error_rejects_plausible_norm_bugs(self):
        """The scale-aware metric still catches real defects: a dropped output
        gate and a norm taken over the wrong length."""
        import torch

        from builders.gfx950.gdn.gdn_decode import TOL, make_inputs, out_error, ref_fp32

        s = self.spec(fuse_conv=True, fuse_out_norm=True)
        inp = make_inputs(s, 16, device="cpu")
        ref, _ = ref_fp32(s, inp)
        no_gate = ref / torch.sigmoid(inp["out_gate"].float().reshape(ref.shape))
        self.assertGreater(out_error(s, no_gate, ref), TOL)
        wrong_len = ref * (2.0**0.5)  # rsqrt(sum/(DV/2)) instead of rsqrt(sum/DV)
        self.assertGreater(out_error(s, wrong_len, ref), TOL)

    def test_validation_accepts_fused_inputs(self):
        from builders.gfx950.gdn.gdn_decode import _validate_decode_inputs, make_inputs

        s = self.spec(fuse_conv=True, fuse_out_norm=True)
        _validate_decode_inputs(s, make_inputs(s, 3, device="cpu"), 3)

    def test_validation_rejects_bad_conv_state(self):
        from builders.gfx950.gdn.gdn_decode import _validate_decode_inputs, make_inputs

        s = self.spec(fuse_conv=True)
        inp = make_inputs(s, 3, device="cpu")
        inp["conv_state"] = inp["conv_state"][:, :-1]
        with self.assertRaises(ValueError):
            _validate_decode_inputs(s, inp, 3)


if __name__ == "__main__":
    unittest.main()
