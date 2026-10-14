# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The ``target_features`` kernel attribute.

It lowers verbatim to the LLVM ``"target-features"`` function attribute in both
engines; a kernel without it (or with an empty string) lowers unchanged.
"""

from __future__ import annotations

import unittest

from rocke.core.ir import I32, IRBuilder, PtrType
from rocke.core.ir_serialize import parse, serialize
from rocke.core.lower_llvm import LLVM_FLAVORS, _lower_kernel_to_llvm_python


def _kernel(features=None):
    b = IRBuilder("target_features_attr")
    out = b.param("out", PtrType(I32, "global"))
    b.global_store(out, b.const_i32(0), b.const_i32(7), align=4)
    b.ret()
    if features is not None:
        b.kernel.attrs["target_features"] = features
    return b.kernel


def _fn_attrs(ll: str) -> str:
    return next(line for line in ll.splitlines() if line.startswith("attributes #0"))


class TestTargetFeaturesAttr(unittest.TestCase):
    def test_attr_emitted_on_the_kernel(self):
        for features in ("-packed-fp32-ops", "+wavefrontsize64,-packed-fp32-ops"):
            with self.subTest(features=features):
                attrs = _fn_attrs(_lower_kernel_to_llvm_python(_kernel(features)))
                self.assertIn(
                    f' "target-features"="{features}" norecurse nounwind }}', attrs
                )
                self.assertEqual(attrs.count("target-features"), 1)

    def test_absent_or_empty_attr_changes_nothing(self):
        plain = _lower_kernel_to_llvm_python(_kernel())
        self.assertNotIn("target-features", plain)
        self.assertEqual(plain, _lower_kernel_to_llvm_python(_kernel("")))
        with_attr = _lower_kernel_to_llvm_python(_kernel("-packed-fp32-ops"))
        self.assertEqual(
            plain, with_attr.replace(' "target-features"="-packed-fp32-ops"', "")
        )

    def test_rejects_non_string(self):
        for bad in (1, ["-packed-fp32-ops"]):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(
                    ValueError, "target_features must be a string"
                ):
                    _lower_kernel_to_llvm_python(_kernel(bad))

    def test_attr_survives_serialization(self):
        kernel = parse(serialize(_kernel("-packed-fp32-ops")))
        self.assertEqual(kernel.attrs["target_features"], "-packed-fp32-ops")


class TestTargetFeaturesBothEngines(unittest.TestCase):
    def setUp(self):
        try:
            import rocke_engine  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise unittest.SkipTest(f"rocke_engine not importable: {exc}")

    def _cpp(self, kernel, flavor):
        from rocke.core.backend import _lower_via_cpp_engine

        return _lower_via_cpp_engine(serialize(kernel), "gfx950", flavor)

    def test_engines_agree_byte_for_byte(self):
        for flavor in LLVM_FLAVORS:
            for features in (None, "", "-packed-fp32-ops"):
                with self.subTest(flavor=flavor, features=features):
                    kernel = _kernel(features)
                    self.assertEqual(
                        self._cpp(kernel, flavor),
                        _lower_kernel_to_llvm_python(
                            kernel, llvm_flavor=flavor, arch="gfx950"
                        ),
                    )

    def test_cpp_rejects_non_string(self):
        with self.assertRaisesRegex(Exception, "target_features must be a string"):
            self._cpp(_kernel(3), LLVM_FLAVORS[0])


if __name__ == "__main__":
    unittest.main()
