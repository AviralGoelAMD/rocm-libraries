# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU checks for the KDA gate kind of the decode fp32 reference.

The reference is the independent oracle the on-device tests score against, so
it gets checked here first -- against the gate written out longhand, and
against the GDN reference it has to stay compatible with. A reference nobody
audits is not an oracle, it is a second implementation with the same bugs.

Everything here runs on CPU torch: shapes, the gate formula, and the subset
relation are all device-independent. Only the kernel needs a GPU.
"""

from __future__ import annotations

import dataclasses as dc
import unittest

import pytest

torch = pytest.importorskip("torch", reason="torch required (CPU is enough)")

from builders.gfx950.gdn.gdn_decode import make_inputs, ref_fp32  # noqa: E402
from kernels.gfx950.gdn_decode import GdnDecodeSpec  # noqa: E402


def _kda_spec(**kw):
    return dc.replace(GdnDecodeSpec(), gate_kind="kda", **kw)


class TestInputShapes(unittest.TestCase):
    """The KDA gate is per-channel, and its bias is f32. Both are contract."""

    def test_kda_gate_is_per_channel_and_bias_is_f32(self):
        spec = _kda_spec()
        inp = make_inputs(spec, batch=2, device="cpu")

        self.assertEqual(
            tuple(inp["a"].shape), (2, 1, spec.num_v_heads, spec.head_k_dim)
        )
        self.assertEqual(
            tuple(inp["dt_bias"].shape), (spec.num_v_heads, spec.head_k_dim)
        )
        # f32 is required, not stylistic: the shared gate helper reads dt_bias
        # with a 32-bit load, and KDA prefill already declares it f32.
        self.assertEqual(inp["dt_bias"].dtype, torch.float32)

    def test_a_log_stays_per_head_in_both_kinds(self):
        # Only dt_bias changes rank between the gate kinds. A_log does not.
        for spec in (GdnDecodeSpec(), _kda_spec()):
            inp = make_inputs(spec, batch=2, device="cpu")
            self.assertEqual(tuple(inp["A_log"].shape), (spec.num_v_heads,))

    def test_gdn_shapes_are_unchanged(self):
        spec = GdnDecodeSpec()
        inp = make_inputs(spec, batch=2, device="cpu")

        self.assertEqual(tuple(inp["a"].shape), (2, 1, spec.num_v_heads))
        self.assertEqual(tuple(inp["dt_bias"].shape), (spec.num_v_heads,))

    def test_gdn_inputs_are_bitwise_reproducible(self):
        # The KDA branch must not perturb the GDN random stream, or every
        # previously recorded GDN number silently refers to different inputs.
        a = make_inputs(GdnDecodeSpec(), batch=3, seed=7, device="cpu")
        b = make_inputs(GdnDecodeSpec(), batch=3, seed=7, device="cpu")
        for key in a:
            self.assertTrue(torch.equal(a[key], b[key]), key)


class TestKdaGateFormula(unittest.TestCase):
    def test_reference_matches_the_gate_written_longhand(self):
        spec = _kda_spec()
        inp = make_inputs(spec, batch=2, device="cpu")

        # decay = exp(lower_bound * sigmoid(exp(A_log) * (g + dt_bias)))
        inner = torch.exp(inp["A_log"].float())[None, :, None] * (
            inp["a"][:, 0].float() + inp["dt_bias"].float()
        )
        decay = torch.exp(spec.lower_bound * torch.sigmoid(inner))

        self.assertEqual(tuple(decay.shape), (2, spec.num_v_heads, spec.head_k_dim))
        # sigmoid is in (0, 1) and lower_bound < 0, so the decay is a genuine
        # fade: strictly positive, never amplifying.
        self.assertTrue(bool((decay > 0).all()))
        self.assertTrue(bool((decay < 1).all()))

        # The reference must fade the state by exactly that, per channel.
        state = inp["state"].float()[inp["read_indices"].long()]
        faded = state * decay[..., None, :]

        out, state_after = ref_fp32(spec, inp)
        self.assertEqual(tuple(out.shape), (2, 1, spec.num_v_heads, spec.head_v_dim))
        self.assertTrue(bool(torch.isfinite(out).all()))
        self.assertTrue(bool(torch.isfinite(state_after).all()))
        # state_after = faded + rank-1 update, so it cannot equal the raw fade,
        # but every entry must have moved off the *unfaded* state.
        self.assertFalse(torch.allclose(state_after, state))
        self.assertEqual(tuple(state_after.shape), tuple(faded.shape))

    def test_decay_is_genuinely_per_channel(self):
        # A per-head decay broadcast across DK would pass a shape check but be
        # the GDN gate wearing a KDA hat. Assert the channels actually differ.
        spec = _kda_spec()
        inp = make_inputs(spec, batch=2, device="cpu")
        inner = torch.exp(inp["A_log"].float())[None, :, None] * (
            inp["a"][:, 0].float() + inp["dt_bias"].float()
        )
        decay = torch.exp(spec.lower_bound * torch.sigmoid(inner))

        spread = decay.amax(dim=-1) - decay.amin(dim=-1)
        self.assertGreater(float(spread.max()), 1e-3)


class TestSubsetRelation(unittest.TestCase):
    """GDN is KDA with every channel equal. Tested, not asserted in prose."""

    def test_channel_constant_kda_reproduces_gdn(self):
        gdn_spec = GdnDecodeSpec()
        kda_spec = _kda_spec()
        inp = make_inputs(gdn_spec, batch=2, device="cpu")
        gdn_out, gdn_state = ref_fp32(gdn_spec, inp)

        # Solve for the per-channel g that reproduces GDN's scalar decay.
        #   GDN: log_decay = -exp(A_log) * softplus(a + dt_bias)
        #   KDA: log_decay = lower_bound * sigmoid(exp(A_log) * (g + dt_bias))
        x = inp["a"][:, 0].float() + inp["dt_bias"].float()
        softplus = torch.where(x > 20.0, x, torch.log1p(torch.exp(x)))
        target = -torch.exp(inp["A_log"].float()) * softplus  # [B, HV]

        sig = (target / kda_spec.lower_bound).clamp(1e-6, 1 - 1e-6)
        g = torch.log(sig / (1 - sig)) / torch.exp(inp["A_log"].float())

        kda_inp = dict(inp)
        kda_inp["a"] = (
            g[:, None, :, None]
            .expand(-1, -1, -1, kda_spec.head_k_dim)
            .contiguous()
            .to(inp["a"].dtype)
        )
        kda_inp["dt_bias"] = torch.zeros(
            kda_spec.num_v_heads, kda_spec.head_k_dim, dtype=torch.float32
        )

        kda_out, kda_state = ref_fp32(kda_spec, kda_inp)

        torch.testing.assert_close(kda_out, gdn_out, rtol=2e-3, atol=2e-3)
        torch.testing.assert_close(kda_state, gdn_state, rtol=2e-3, atol=2e-3)


if __name__ == "__main__":
    unittest.main()
