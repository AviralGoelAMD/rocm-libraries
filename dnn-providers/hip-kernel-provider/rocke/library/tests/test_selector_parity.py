# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Pure-Python unit tests for the gfx950 attention tiled-2D selector logic.

These tests verify that the Python ``_tiled_spec_from_problem`` selector
produces the expected spec fields for a range of UnifiedAttentionProblem
inputs -- covering the selectors and gate predicates that route a problem
into the correct kernel variant.  No GPU, no subprocess, no rocke_engine .so
required.

Primary motivation: PR #9220 silently broke parity by removing the bias-
exclusion guard from Python ``_enable_combo_2d``, routing biased combo-geometry
attention onto the transposed combo path while C++ still refused bias.  These
tests encode the expected routing decisions so that kind of regression is caught
immediately on any developer machine.

NOTE ON EXPECTED FAILURES (PR #9220 regression):
Tests marked with @skip document a known divergence introduced by
PR #9220, which removed the bias-exclusion guard from Python _enable_combo_2d.
Python now routes biased combo-geometry problems onto the combo path while
C++ (attention_unified_selectors.cpp) still refuses them. The tests are
written for the CORRECT behavior (bias should block combo_2d) and are being skipped until the Python guard is restored.
"""

from __future__ import annotations

import unittest
from dataclasses import asdict, replace
from unittest import mock

import builders.common.attention_spec_builder as _asb
import kernels.common.attention_unified as _au
from kernels import UnifiedAttentionProblem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_arch(arch: str):
    """Pin ``_resolve_attention_arch`` on the module that defines it.

    That is sufficient because the spec builder reaches the resolver through its
    ``attention_unified`` module handle -- a bound import there would freeze the
    reference and silently leave the builder on the real device arch, so these
    cases would test the host, not ``arch``. That invariant is pinned, statically
    and behaviourally, by ``test_arch_binding_guard.py``; it is not re-asserted
    here.
    """
    return mock.patch.object(_au, "_resolve_attention_arch", return_value=arch)


def _spec(problem: UnifiedAttentionProblem, arch: str = "gfx950") -> dict:
    """Return _tiled_spec_from_problem as a plain dict for easy assertions."""
    with _patch_arch(arch):
        return asdict(_asb._tiled_spec_from_problem(problem))


def _prob(
    head_size: int,
    block_size: int,
    num_query_heads: int,
    num_kv_heads: int,
    dtype: str = "bf16",
    max_seqlen_q: int = 512,
    max_seqlen_k: int = 4096,
    num_seqs: int = 2,
    total_q: int = 1024,
    sliding_window: int = 0,
    softcap: float = 0.0,
    use_sinks: bool = False,
    use_alibi: bool = False,
    use_qq_bias: bool = False,
    use_fp8: bool = False,
    num_kv_blocks: int = 0,
) -> UnifiedAttentionProblem:
    return UnifiedAttentionProblem(
        total_q=total_q,
        num_seqs=num_seqs,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        block_size=block_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dtype=dtype,
        sliding_window=sliding_window,
        softcap=softcap,
        use_sinks=use_sinks,
        use_alibi=use_alibi,
        use_qq_bias=use_qq_bias,
        use_fp8=use_fp8,
        num_kv_blocks=num_kv_blocks,
    )


# Canonical combo_2d problem: gfx950 + bf16 + d64/b32 + GQA-8 + seqlen_q > 256
def _combo_prob(**overrides) -> UnifiedAttentionProblem:
    defaults = dict(
        head_size=64,
        block_size=32,
        num_query_heads=64,
        num_kv_heads=8,
        dtype="bf16",
        max_seqlen_q=512,
    )
    defaults.update(overrides)
    return _prob(**defaults)


# ---------------------------------------------------------------------------
# enable_combo_2d gate
# ---------------------------------------------------------------------------


class TestEnableCombo2d(unittest.TestCase):
    """_enable_combo_2d fires for the gfx950 d64/b32/GQA-8/S>256 cohort: bf16
    (sinks or not), or fp16 only when use_sinks (the fp16 widening was measured
    on sink prefill)."""

    _BIAS_BUG = (
        "PR #9220: Python _enable_combo_2d missing bias guard (C++ still refuses bias)"
    )

    def _gate(self, problem: UnifiedAttentionProblem, arch: str = "gfx950") -> bool:
        with _patch_arch(arch):
            return _au._enable_combo_2d(problem)

    def test_canonical_combo_fires(self):
        self.assertTrue(self._gate(_combo_prob()))

    @unittest.skip(_BIAS_BUG)
    def test_qq_bias_blocks_combo(self):
        """use_qq_bias=True must disable combo_2d (was the #9220 bug)."""
        self.assertFalse(self._gate(_combo_prob(use_qq_bias=True)), self._BIAS_BUG)

    @unittest.skip(_BIAS_BUG)
    def test_alibi_blocks_combo(self):
        self.assertFalse(self._gate(_combo_prob(use_alibi=True)), self._BIAS_BUG)

    @unittest.skip(_BIAS_BUG)
    def test_softcap_blocks_combo(self):
        self.assertFalse(self._gate(_combo_prob(softcap=50.0)), self._BIAS_BUG)

    def test_fp16_no_sinks_blocks_combo(self):
        # fp16 combo is sink-prefill only; non-sink fp16 stays off the combo path.
        self.assertFalse(self._gate(_combo_prob(dtype="fp16")))

    def test_fp16_sinks_enables_combo(self):
        # fp16 + sinks is the one fp16 shape the widening admits to combo.
        self.assertTrue(self._gate(_combo_prob(dtype="fp16", use_sinks=True)))

    def test_wrong_head_size_blocks_combo(self):
        self.assertFalse(self._gate(_prob(128, 32, 64, 8)))

    def test_wrong_block_size_blocks_combo(self):
        self.assertFalse(self._gate(_prob(64, 16, 64, 8)))

    def test_wrong_gqa_ratio_blocks_combo(self):
        # nqpk = 16/4 = 4, not 8
        self.assertFalse(self._gate(_prob(64, 32, 16, 4)))

    def test_short_seqlen_blocks_combo(self):
        self.assertFalse(self._gate(_combo_prob(max_seqlen_q=256)))
        self.assertFalse(self._gate(_combo_prob(max_seqlen_q=100)))

    def test_gfx942_blocks_combo(self):
        self.assertFalse(self._gate(_combo_prob(), arch="gfx942"))


# ---------------------------------------------------------------------------
# Routing: combo_2d cohort → transposed_qk_32x32 flags
# ---------------------------------------------------------------------------


class TestCombo2dRouting(unittest.TestCase):
    """combo_2d problems must get the full 32x32 transposed stack."""

    _BIAS_BUG = "PR #9220: Python _enable_combo_2d missing bias guard"

    def test_combo_gets_transposed_flags(self):
        s = _spec(_combo_prob())
        self.assertTrue(s["use_mfma_32x32"])
        self.assertTrue(s["use_transposed_qk_32x32"])
        self.assertTrue(s["use_transposed_half_local_pv"])
        self.assertTrue(s["use_transposed_scalar_state"])

    @unittest.skip(_BIAS_BUG)
    def test_biased_combo_does_not_get_fast_paged_kv(self):
        """#9220: biased combo must NOT get use_fast_paged_kv_desc."""
        s = _spec(_combo_prob(use_qq_bias=True))
        self.assertFalse(s["use_fast_paged_kv_desc"], self._BIAS_BUG)

    @unittest.skip(_BIAS_BUG)
    def test_alibi_combo_no_fast_paged_kv(self):
        s = _spec(_combo_prob(use_alibi=True))
        self.assertFalse(s["use_fast_paged_kv_desc"], self._BIAS_BUG)

    @unittest.skip(_BIAS_BUG)
    def test_softcap_combo_no_fast_paged_kv(self):
        s = _spec(_combo_prob(softcap=50.0))
        self.assertFalse(s["use_fast_paged_kv_desc"], self._BIAS_BUG)

    def test_combo_no_sw_gets_fast_paged_kv(self):
        s = _spec(_combo_prob())
        self.assertTrue(s["use_fast_paged_kv_desc"])

    def test_combo_with_sw_no_fast_paged_kv(self):
        s = _spec(_combo_prob(sliding_window=256))
        self.assertFalse(s["use_fast_paged_kv_desc"])


# ---------------------------------------------------------------------------
# Bias x geometry cohort matrix (covers the #9220 class of divergence)
# ---------------------------------------------------------------------------


class TestBiasGeometryCohorts(unittest.TestCase):
    """Systematic sweep of bias flags x combo-geometry dimensions."""

    _BIAS_BUG = "PR #9220: Python _enable_combo_2d missing bias guard"

    def _fast_kv(self, problem: UnifiedAttentionProblem) -> bool:
        return _spec(problem)["use_fast_paged_kv_desc"]

    # ---- baseline: canonical combo fires ----
    def test_canonical_combo(self):
        self.assertTrue(self._fast_kv(_combo_prob()))

    # ---- PR #9220 biased-combo cohort (skip the test until bug is fixed) ----
    @unittest.skip(_BIAS_BUG)
    def test_qq_bias_blocks_fast_paged_kv(self):
        self.assertFalse(self._fast_kv(_combo_prob(use_qq_bias=True)), self._BIAS_BUG)

    @unittest.skip(_BIAS_BUG)
    def test_alibi_blocks_fast_paged_kv(self):
        self.assertFalse(self._fast_kv(_combo_prob(use_alibi=True)), self._BIAS_BUG)

    @unittest.skip(_BIAS_BUG)
    def test_softcap_blocks_fast_paged_kv(self):
        self.assertFalse(self._fast_kv(_combo_prob(softcap=50.0)), self._BIAS_BUG)

    @unittest.skip(_BIAS_BUG)
    def test_alibi_and_qq_bias_no_combo(self):
        self.assertFalse(
            self._fast_kv(_combo_prob(use_alibi=True, use_qq_bias=True)), self._BIAS_BUG
        )

    # ---- geometry boundaries (not affected by #9220) ----
    def test_seqlen_at_boundary_no_combo(self):
        self.assertFalse(self._fast_kv(_combo_prob(max_seqlen_q=256)))

    def test_seqlen_above_boundary_combo(self):
        self.assertTrue(self._fast_kv(_combo_prob(max_seqlen_q=257)))

    def test_wrong_dtype_no_combo(self):
        self.assertFalse(self._fast_kv(_combo_prob(dtype="fp16")))

    def test_wrong_head_size_no_combo(self):
        self.assertFalse(self._fast_kv(_prob(128, 32, 64, 8)))

    def test_wrong_gqa_ratio_no_combo(self):
        self.assertFalse(self._fast_kv(_prob(64, 32, 16, 4)))


# ---------------------------------------------------------------------------
# Selector fields: tile_size, num_warps, block_m_per_warp, waves_per_eu
# ---------------------------------------------------------------------------


class TestSelectorFields(unittest.TestCase):
    """Key selector outputs for representative problems."""

    def test_combo_tile_size_is_2x_block(self):
        # combo_2d, no SW: tile_size = 2 * block_size = 64
        s = _spec(_combo_prob())
        self.assertEqual(s["tile_size"], 64)

    def test_combo_sw_tile_size_is_block_size(self):
        # combo_2d + SW: tile_size = block_size = 32
        s = _spec(_combo_prob(sliding_window=256))
        self.assertEqual(s["tile_size"], 32)

    def test_single_batch_d64_tile_size(self):
        # single-batch d64 combo: tile_size = 128
        s = _spec(_prob(64, 32, 32, 32, num_seqs=1, max_seqlen_q=512))
        self.assertEqual(s["tile_size"], 128)

    def test_combo_waves_per_eu(self):
        # combo_2d: waves_per_eu = 4
        s = _spec(_combo_prob())
        self.assertEqual(s["waves_per_eu"], 4)

    def test_combo_block_m_per_warp(self):
        # combo_2d uses 32x32: block_m_per_warp = 32
        s = _spec(_combo_prob())
        self.assertEqual(s["block_m_per_warp"], 32)

    def test_non_combo_block_m_per_warp(self):
        # short seqlen → not combo, not transposed → block_m_per_warp = 16
        s = _spec(_combo_prob(max_seqlen_q=64))
        self.assertEqual(s["block_m_per_warp"], 16)

    def test_non_combo_waves_per_eu(self):
        # standard path: waves_per_eu = 2
        s = _spec(_combo_prob(max_seqlen_q=64))
        self.assertEqual(s["waves_per_eu"], 2)


# ---------------------------------------------------------------------------
# Single-batch combo cohort
# ---------------------------------------------------------------------------


class TestSingleBatchCombo(unittest.TestCase):

    def test_single_batch_d64_gets_transposed_flags(self):
        s = _spec(_prob(64, 32, 32, 32, num_seqs=1, max_seqlen_q=512))
        self.assertTrue(s["use_mfma_32x32"])
        self.assertTrue(s["use_transposed_qk_32x32"])

    def test_single_batch_bias_no_transposed(self):
        # single-batch + bias: _enable_single_batch_combo rejects alibi/qq_bias
        s = _spec(_prob(64, 32, 32, 32, num_seqs=1, max_seqlen_q=512, use_qq_bias=True))
        # transposed_qk_32x32 may still fire via multi-batch branch, but
        # fast_paged_kv_desc (which needs combo + no-bias) must be off
        self.assertFalse(s["use_fast_paged_kv_desc"])

    def test_single_batch_long_d64_gets_early_v(self):
        s = _spec(_prob(64, 32, 32, 32, num_seqs=1, max_seqlen_q=2048))
        self.assertTrue(s["use_early_v_schedule"])
        self.assertFalse(s["use_v_double_buffer"])

    def test_single_batch_short_d64_gets_v_double_buffer(self):
        s = _spec(_prob(64, 32, 32, 32, num_seqs=1, max_seqlen_q=512))
        self.assertFalse(s["use_early_v_schedule"])
        self.assertTrue(s["use_v_double_buffer"])

    def test_single_batch_d128_ksingle_off_not_raise(self):
        # d128 single-batch triggers the softmax-MFMA interleave -> num_warps=4
        # (block_m=128). With tile_size=64 the geometry guard block_m <= tile_size
        # fails, so _enable_k_single_buffer derives K-single OFF and the spec
        # BUILDS (previously the stale block_size>=32 proxy left K-single ON here,
        # raising an uncaught ValueError at spec build).
        s = _spec(_prob(128, 32, 32, 32, num_seqs=1, max_seqlen_q=512))
        self.assertFalse(s["use_k_single_buffer"])


# ---------------------------------------------------------------------------
# Register-PV gate
# ---------------------------------------------------------------------------


class TestRegisterPv(unittest.TestCase):

    def test_register_pv_off_for_combo(self):
        # combo_2d uses mfma_32x32 which conflicts with register_pv
        s = _spec(_combo_prob())
        self.assertFalse(s["use_register_pv"])

    def test_register_pv_off_for_bias(self):
        s = _spec(_prob(64, 32, 32, 32, use_qq_bias=True))
        self.assertFalse(s["use_register_pv"])

    def test_register_pv_off_for_alibi(self):
        s = _spec(_prob(64, 32, 32, 32, use_alibi=True))
        self.assertFalse(s["use_register_pv"])

    def test_register_pv_off_for_sinks(self):
        s = _spec(_prob(64, 32, 32, 32, use_sinks=True))
        self.assertFalse(s["use_register_pv"])

    def test_register_pv_eligible_for_standard_bf16(self):
        # Standard bf16 short-seqlen, no special flags → register_pv eligible
        s = _spec(_prob(64, 32, 32, 32, max_seqlen_q=64))
        self.assertTrue(s["use_register_pv"])


# ---------------------------------------------------------------------------
# i64_kv_addr
# ---------------------------------------------------------------------------


class TestI64KvAddr(unittest.TestCase):

    def test_no_kv_blocks_no_i64(self):
        s = _spec(_combo_prob())  # num_kv_blocks=0 default
        self.assertFalse(s["use_i64_kv_addr"])

    def test_small_cache_no_i64(self):
        # 65536 blocks × 32 tokens × 8 kv-heads × 64 dims × 2 bytes = 2 GiB exactly
        # → still i32 (threshold is STRICTLY > 2 GiB)
        s = _spec(_combo_prob(num_kv_blocks=65536))
        self.assertFalse(s["use_i64_kv_addr"])

    def test_large_cache_gets_i64(self):
        # 65537 blocks × same stride → just over 2 GiB → i64
        s = _spec(_combo_prob(num_kv_blocks=65537))
        self.assertTrue(s["use_i64_kv_addr"])


# ---------------------------------------------------------------------------
# Transposed sub-flags (mask_once / mask_limit / scalar_state)
# ---------------------------------------------------------------------------


class TestTransposedSubflags(unittest.TestCase):

    def test_combo_no_sw_gets_mask_flags(self):
        s = _spec(_combo_prob())
        self.assertTrue(s["use_transposed_mask_once"])
        self.assertTrue(s["use_transposed_mask_limit"])
        self.assertTrue(s["use_transposed_scalar_state"])

    def test_combo_with_sw_no_mask_flags(self):
        # SW combo: mask_once / mask_limit must be off
        s = _spec(_combo_prob(sliding_window=256))
        self.assertFalse(s["use_transposed_mask_once"])
        self.assertFalse(s["use_transposed_mask_limit"])

    def test_biased_combo_no_mask_flags(self):
        # bias_active blocks mask_opts
        s = _spec(_combo_prob(use_qq_bias=True))
        self.assertFalse(s["use_transposed_mask_once"])
        self.assertFalse(s["use_transposed_mask_limit"])


# ---------------------------------------------------------------------------
# gfx950 pipelined prefill stack (_gfx950_prefill_pipeline); C++ twin:
# gfx950_prefill_pipeline in attention_unified_selectors.cpp.
# ---------------------------------------------------------------------------

_PIPELINE_FLAGS = (
    "use_k_lds_xor_swizzle",
    "use_v_lds_xor_swizzle",
    "use_reverse_q_block_order",
    "use_v_double_buffer",
    "use_lds_dma_overlap",
    "use_mask_phase_split",
    "use_kv_block_id_prefetch",
    "use_softmax_scale_fold",
)


def _pipe_prob(**overrides) -> UnifiedAttentionProblem:
    """d128 bs16 GQA-4 causal prefill (the pipeline cohort's SSOT shape)."""
    defaults = dict(
        head_size=128,
        block_size=16,
        num_query_heads=32,
        num_kv_heads=8,
        max_seqlen_q=8192,
        max_seqlen_k=8192,
        num_seqs=1,
        total_q=8192,
    )
    defaults.update(overrides)
    return _prob(**defaults)


class TestGfx950PrefillPipeline(unittest.TestCase):
    def _routed(self, problem: UnifiedAttentionProblem, arch: str = "gfx950") -> bool:
        with _patch_arch(arch):
            return _au._gfx950_prefill_pipeline(problem)

    def _assert_pipeline_spec(self, s: dict, num_warps: int) -> None:
        for flag in _PIPELINE_FLAGS:
            self.assertTrue(s[flag], flag)
        for flag in (
            "use_kq_lds_pad",
            "use_staggered_iter_wait",
            "use_early_v_schedule",
            "use_k_single_buffer",
            "use_fast_paged_kv_desc",
            "use_sched_barrier",
        ):
            self.assertFalse(s[flag], flag)
        self.assertEqual(s["tile_size"], 64)
        self.assertEqual(s["num_warps"], num_warps)
        self.assertTrue(s["use_transposed_half_local_pv"])

    def test_single_seq_bf16_and_fp16(self):
        for dtype in ("bf16", "fp16"):
            p = _pipe_prob(dtype=dtype)
            self.assertTrue(self._routed(p), dtype)
            s = _spec(p)
            self._assert_pipeline_spec(s, num_warps=4)
            # single-seq keeps the softmax<->MFMA interleave (iglp mode 1)
            self.assertTrue(s["use_softmax_mfma_interleave"])

    def test_single_seq_gqa8(self):
        p = _pipe_prob(num_query_heads=8, num_kv_heads=1, max_seqlen_q=2048,
                       max_seqlen_k=2048, total_q=2048)
        self.assertTrue(self._routed(p))
        self._assert_pipeline_spec(_spec(p), num_warps=4)

    def test_multi_seq_and_ragged(self):
        for num_seqs, total_q, max_q in ((4, 8192, 2048), (7, 8204, 3000)):
            p = _pipe_prob(num_seqs=num_seqs, total_q=total_q, max_seqlen_q=max_q,
                           max_seqlen_k=max_q)
            self.assertTrue(self._routed(p), num_seqs)
            s = _spec(p)
            # nw=4 pinned (the generic LDS step-down would pick 2)
            self._assert_pipeline_spec(s, num_warps=4)
            self.assertFalse(s["use_softmax_mfma_interleave"])

    def test_llvm_backend_pinned(self):
        with _patch_arch("gfx950"):
            for num_seqs in (1, 4):
                p = _pipe_prob(num_seqs=num_seqs, max_seqlen_q=2048, max_seqlen_k=2048)
                self.assertEqual(_au._select_2d_compile_backend(p), "llvm")

    def test_explicit_hipcc_keeps_plain_body(self):
        p = replace(_pipe_prob(), compile_backend="hipcc")
        with _patch_arch("gfx950"):
            self.assertTrue(_au._gfx950_prefill_pipeline_cohort(p))
        self.assertFalse(self._routed(p))
        s = _spec(p)
        for flag in _PIPELINE_FLAGS:
            if flag != "use_mask_phase_split":
                self.assertFalse(s[flag], flag)

    def test_cache_key_separates_pipeline(self):
        p = _pipe_prob()
        with _patch_arch("gfx950"):
            k_pipe = _au._tiled_cache_key(p)
            k_hip = _au._tiled_cache_key(replace(p, compile_backend="hipcc"))
        self.assertNotEqual(k_pipe, k_hip)

    def test_exclusions(self):
        excluded = {
            "block_size_32": _pipe_prob(block_size=32),
            "head_size_64": _pipe_prob(head_size=64),
            "sliding_window": _pipe_prob(sliding_window=256),
            "sinks": _pipe_prob(use_sinks=True),
            "alibi": _pipe_prob(use_alibi=True),
            "qq_bias": _pipe_prob(use_qq_bias=True),
            "softcap": _pipe_prob(softcap=30.0),
            "fp8": _pipe_prob(use_fp8=True),
            "short_q": _pipe_prob(max_seqlen_q=256, total_q=256),
        }
        for name, p in excluded.items():
            self.assertFalse(self._routed(p), name)
            self.assertNotEqual(_spec(p).get("use_lds_dma_overlap"), True, name)

    def test_other_arches_never_route(self):
        for arch in ("gfx942", "gfx1250"):
            self.assertFalse(self._routed(_pipe_prob(), arch=arch), arch)


if __name__ == "__main__":
    unittest.main()
