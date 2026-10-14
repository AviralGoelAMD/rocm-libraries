# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only contracts for typed per-kernel code-generation policy."""

from __future__ import annotations

import inspect
import unittest

from rocke.core.codegen_policy import (
    CodegenPolicy,
    F32DenormalMode,
    SchedulerStrategy,
    apply_codegen_policy,
    codegen_policy_for_kernel,
)
from rocke.core.ir import F32, I32, IRBuilder, PtrType
from rocke.core.ir_serialize import parse, serialize
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.helpers.autotune import AutotuneConfig
from rocke.helpers.compile import (
    _comgr_options_for_kernel,
    _hipcc_policy_flags,
    compile_kernel_via_hipcc,
)


def _kernel():
    builder = IRBuilder("codegen_policy_test")
    builder.kernel.attrs["max_workgroup_size"] = 64
    builder.ret()
    return builder.kernel


def _fp_kernel():
    """Scalar f32 add/sub/mul/div plus one integer add, all stored."""
    builder = IRBuilder("codegen_policy_fp_test")
    builder.kernel.attrs["max_workgroup_size"] = 64
    out = builder.param("out", PtrType(F32, "global"))
    iout = builder.param("iout", PtrType(I32, "global"))
    x = builder.param("x", F32)
    y = builder.param("y", F32)
    n = builder.param("n", I32)
    s = builder.fadd(x, y)
    d = builder.fsub(s, y)
    m = builder.fmul(d, x)
    q = builder.fdiv(m, y)
    builder.store(q, out)
    builder.store(builder.add(n, n), iout)
    builder.ret()
    return builder.kernel


class TestCodegenPolicy(unittest.TestCase):
    def test_every_supported_scheduler_strategy_is_canonical(self):
        expected = {
            "max-ilp",
            "max-memory-clause",
            "iterative-ilp",
            "iterative-minreg",
            "iterative-maxocc",
        }
        self.assertEqual({strategy.value for strategy in SchedulerStrategy}, expected)
        for strategy in SchedulerStrategy:
            with self.subTest(strategy=strategy.value):
                policy = CodegenPolicy(scheduler_strategy=strategy)
                self.assertEqual(policy.scheduler_strategy, strategy.value)
                self.assertEqual(
                    CodegenPolicy(scheduler_strategy=strategy.value).scheduler_strategy,
                    strategy.value,
                )
                kernel = _kernel()
                apply_codegen_policy(kernel, policy)
                self.assertIn(
                    f'"amdgpu-sched-strategy"="{strategy.value}"',
                    _lower_kernel_to_llvm_python(kernel, arch="gfx950"),
                )

    def test_invalid_scheduler_strategies_are_rejected(self):
        for value in ("", "default", "ITERATIVE-ILP", 1, True, object()):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    CodegenPolicy(scheduler_strategy=value)

    def test_apply_and_remove_policy(self):
        kernel = _kernel()
        apply_codegen_policy(
            kernel, CodegenPolicy(scheduler_strategy="iterative-minreg")
        )
        self.assertEqual(
            codegen_policy_for_kernel(kernel),
            CodegenPolicy(scheduler_strategy="iterative-minreg"),
        )
        apply_codegen_policy(kernel, CodegenPolicy())
        self.assertNotIn("scheduler_strategy", kernel.attrs)

    def test_apply_rejects_untyped_policy(self):
        with self.assertRaises(TypeError):
            apply_codegen_policy(_kernel(), {"scheduler_strategy": "max-ilp"})

    def test_policy_survives_serialization_round_trip(self):
        kernel = _kernel()
        apply_codegen_policy(kernel, CodegenPolicy(scheduler_strategy="max-ilp"))
        reparsed = parse(serialize(kernel))
        self.assertEqual(
            codegen_policy_for_kernel(reparsed), codegen_policy_for_kernel(kernel)
        )
        self.assertEqual(serialize(reparsed), serialize(kernel))

    def test_default_policy_does_not_change_lowered_llvm(self):
        kernel = _kernel()
        before = _lower_kernel_to_llvm_python(kernel, arch="gfx950")
        apply_codegen_policy(kernel, CodegenPolicy())
        after = _lower_kernel_to_llvm_python(kernel, arch="gfx950")
        self.assertEqual(after, before)
        self.assertNotIn("amdgpu-sched-strategy", after)

    def test_scheduler_attribute_has_stable_order(self):
        kernel = _kernel()
        kernel.attrs["waves_per_eu"] = 2
        apply_codegen_policy(
            kernel, CodegenPolicy(scheduler_strategy="iterative-maxocc")
        )
        llvm = _lower_kernel_to_llvm_python(kernel, arch="gfx950")
        attrs = next(
            line for line in llvm.splitlines() if line.startswith("attributes #0")
        )
        self.assertIn(
            '"amdgpu-flat-work-group-size"="64,64" '
            '"amdgpu-sched-strategy"="iterative-maxocc" '
            '"amdgpu-waves-per-eu"="2,2"',
            attrs,
        )

    def test_scheduler_policy_is_not_forwarded_as_a_raw_comgr_flag(self):
        kernel = _kernel()
        apply_codegen_policy(kernel, CodegenPolicy(scheduler_strategy="max-ilp"))
        self.assertEqual(_comgr_options_for_kernel(kernel), ["-O3"])

    def test_hipcc_rejects_a_policy_it_cannot_honor(self):
        kernel = _kernel()
        apply_codegen_policy(kernel, CodegenPolicy(scheduler_strategy="max-ilp"))
        with self.assertRaisesRegex(ValueError, "does not support scheduler_strategy"):
            compile_kernel_via_hipcc(kernel)

    def test_autotune_extra_carries_policy_without_api_change(self):
        policy = CodegenPolicy(scheduler_strategy="iterative-maxocc")
        config = AutotuneConfig(
            spec=object(),
            name="iterative-maxocc",
            extra={"codegen_policy": policy},
        )
        self.assertEqual(
            tuple(inspect.signature(AutotuneConfig).parameters),
            ("spec", "name", "extra"),
        )
        self.assertIs(config.extra["codegen_policy"], policy)

    def test_f32_denormal_mode_is_canonical(self):
        self.assertEqual(
            {mode.value for mode in F32DenormalMode}, {"ieee", "preserve-sign"}
        )
        self.assertIsNone(CodegenPolicy(f32_denormal_mode="ieee").f32_denormal_mode)
        self.assertIsNone(
            CodegenPolicy(f32_denormal_mode=F32DenormalMode.IEEE).f32_denormal_mode
        )
        self.assertEqual(
            CodegenPolicy(
                f32_denormal_mode=F32DenormalMode.PRESERVE_SIGN
            ).f32_denormal_mode,
            "preserve-sign",
        )

    def test_invalid_f32_denormal_modes_are_rejected(self):
        for value in ("", "positive-zero", "PRESERVE-SIGN", "dynamic", 1, True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    CodegenPolicy(f32_denormal_mode=value)

    def test_invalid_fp_contract_is_rejected(self):
        for value in (None, 0, 1, "true", "fast"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    CodegenPolicy(fp_contract=value)

    def test_f32_denormal_mode_emits_function_attribute_in_stable_order(self):
        kernel = _kernel()
        kernel.attrs["waves_per_eu"] = 2
        apply_codegen_policy(
            kernel,
            CodegenPolicy(
                scheduler_strategy="max-ilp", f32_denormal_mode="preserve-sign"
            ),
        )
        llvm = _lower_kernel_to_llvm_python(kernel, arch="gfx950")
        attrs = next(
            line for line in llvm.splitlines() if line.startswith("attributes #0")
        )
        self.assertIn(
            '"amdgpu-sched-strategy"="max-ilp" '
            '"denormal-fp-math-f32"="preserve-sign,preserve-sign" '
            '"amdgpu-waves-per-eu"="2,2"',
            attrs,
        )
        apply_codegen_policy(kernel, CodegenPolicy(f32_denormal_mode="ieee"))
        self.assertNotIn("f32_denormal_mode", kernel.attrs)
        self.assertNotIn(
            "denormal-fp-math",
            _lower_kernel_to_llvm_python(kernel, arch="gfx950"),
        )

    def test_fp_contract_marks_only_float_add_sub_mul(self):
        kernel = _fp_kernel()
        before = _lower_kernel_to_llvm_python(kernel, arch="gfx950")
        self.assertNotIn(" contract ", before)
        apply_codegen_policy(kernel, CodegenPolicy(fp_contract=True))
        self.assertIs(kernel.attrs["fp_contract"], True)
        after = _lower_kernel_to_llvm_python(kernel, arch="gfx950")
        for op in ("fadd", "fsub", "fmul"):
            with self.subTest(op=op):
                self.assertIn(f"= {op} contract float ", after)
                self.assertNotIn(f"= {op} float ", after)
        self.assertIn("= fdiv float ", after)
        self.assertIn("= add nsw i32 ", after)
        self.assertEqual(after.replace(" contract ", " "), before)
        apply_codegen_policy(kernel, CodegenPolicy())
        self.assertNotIn("fp_contract", kernel.attrs)
        self.assertEqual(_lower_kernel_to_llvm_python(kernel, arch="gfx950"), before)

    def test_new_policy_fields_survive_serialization_round_trip(self):
        kernel = _fp_kernel()
        policy = CodegenPolicy(f32_denormal_mode="preserve-sign", fp_contract=True)
        apply_codegen_policy(kernel, policy)
        reparsed = parse(serialize(kernel))
        self.assertEqual(codegen_policy_for_kernel(reparsed), policy)
        self.assertEqual(
            _lower_kernel_to_llvm_python(reparsed, arch="gfx950"),
            _lower_kernel_to_llvm_python(kernel, arch="gfx950"),
        )

    def test_new_policy_fields_are_not_raw_comgr_flags(self):
        kernel = _kernel()
        apply_codegen_policy(
            kernel, CodegenPolicy(f32_denormal_mode="preserve-sign", fp_contract=True)
        )
        self.assertEqual(_comgr_options_for_kernel(kernel), ["-O3"])

    def test_hipcc_maps_f32_flush_to_its_clang_flag(self):
        self.assertEqual(_hipcc_policy_flags(CodegenPolicy()), [])
        self.assertEqual(_hipcc_policy_flags(CodegenPolicy(fp_contract=True)), [])
        self.assertEqual(
            _hipcc_policy_flags(CodegenPolicy(f32_denormal_mode="preserve-sign")),
            ["-fgpu-flush-denormals-to-zero"],
        )


if __name__ == "__main__":
    unittest.main()
