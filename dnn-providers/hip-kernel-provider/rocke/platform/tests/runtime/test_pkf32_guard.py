# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The gfx950 packed-f32 op_sel guard in ``build_hsaco_from_llvm_ir``.

gfx950 computes ``v_pk_{add,mul}_f32 op_sel:[0,1]`` and ``v_pk_fma_f32`` with
op_sel bit 1 or 2 wrongly when they issue between MFMAs. A gfx950 code object
holding such a form and an MFMA is rebuilt with ``-packed-fp32-ops``; anything
else is returned byte-for-byte as COMGR built it. The compile tests need a
COMGR that can target gfx950 (no GPU).
"""

from __future__ import annotations

import shutil
import struct
import unittest
from unittest import mock

from rocke.analysis.isa import has_matrix_ops, invalid_pk_f32_op_sel
from rocke.core.ir import IRBuilder
from rocke.helpers import compile as compile_module
from rocke.runtime import comgr

_HEAD = """\
target triple = "amdgcn-amd-amdhsa"
declare i32 @llvm.amdgcn.workitem.id.x()
declare <2 x float> @llvm.fma.v2f32(<2 x float>, <2 x float>, <2 x float>)
declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half>, <4 x half>, <4 x float>, i32, i32, i32)
"""

# a * splat(b.hi), a + swap(b) and fma(a, splat(b.hi), .) select src1's high
# half for the low lane: the three invalid forms on a compiler without the fix.
_PACKED = """\
  %pa = getelementptr <2 x float>, ptr addrspace(1) %a, i64 %i
  %pb = getelementptr <2 x float>, ptr addrspace(1) %b, i64 %i
  %va = load <2 x float>, ptr addrspace(1) %pa, align 8
  %vb = load <2 x float>, ptr addrspace(1) %pb, align 8
  %bhi = shufflevector <2 x float> %vb, <2 x float> poison, <2 x i32> <i32 1, i32 1>
  %bsw = shufflevector <2 x float> %vb, <2 x float> poison, <2 x i32> <i32 1, i32 0>
  %m = fmul <2 x float> %va, %bhi
  %s = fadd <2 x float> %va, %bsw
  %f = call <2 x float> @llvm.fma.v2f32(<2 x float> %va, <2 x float> %bhi, <2 x float> %s)
  store <2 x float> %m, ptr addrspace(1) %pa, align 8
  store <2 x float> %s, ptr addrspace(1) %pb, align 8
  %pf = getelementptr <2 x float>, ptr addrspace(1) %pa, i64 4096
  store <2 x float> %f, ptr addrspace(1) %pf, align 8
"""

_MFMA = """\
  %ph = getelementptr <4 x half>, ptr addrspace(1) %h, i64 %i
  %vh = load <4 x half>, ptr addrspace(1) %ph, align 8
  %acc = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> %vh, <4 x half> %vh, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %po = getelementptr <4 x float>, ptr addrspace(1) %o, i64 %i
  store <4 x float> %acc, ptr addrspace(1) %po, align 16
"""


def _ir(name: str, *parts: str) -> str:
    return (
        _HEAD
        + f"define amdgpu_kernel void @{name}(ptr addrspace(1) %a, ptr addrspace(1) %b, "
        "ptr addrspace(1) %h, ptr addrspace(1) %o) #0 {\n"
        "  %tid = call i32 @llvm.amdgcn.workitem.id.x()\n"
        "  %i = zext i32 %tid to i64\n" + "".join(parts) + "  ret void\n}\n\n"
        'attributes #0 = { "amdgpu-flat-work-group-size"="64,256" nounwind }\n'
    )


# The same three forms next to an MFMA, as HIP source for the hipcc path.
_HIP = """\
#include <hip/hip_runtime.h>
typedef float f2 __attribute__((ext_vector_type(2)));
typedef _Float16 h4 __attribute__((ext_vector_type(4)));
typedef float f4 __attribute__((ext_vector_type(4)));

extern "C" __global__ void pkf32_guard_hip(const f2* a, const f2* b, const h4* h,
                                           f2* out, f4* out2) {
  int i = threadIdx.x;
  f2 va = a[i], vb = b[i];
  f4 acc = __builtin_amdgcn_mfma_f32_16x16x16f16(h[i], h[i], f4{0, 0, 0, 0}, 0, 0, 0);
  f2 bhi = vb.yy;
  f2 s = va + vb.yx;
  out[i] = va * bhi;
  out[i + 1024] = s;
  out[i + 2048] = __builtin_elementwise_fma(va, bhi, s);
  out2[i] = acc;
}
"""


class TestInvalidPkF32Classifier(unittest.TestCase):
    def test_forms(self):
        cases = {
            "v_pk_mul_f32 v[14:15], v[4:5], v[6:7] op_sel:[0,1]": True,
            "v_pk_mul_f32 v[2:3], v[4:5], v[6:7] op_sel:[0,1] op_sel_hi:[1,0]": True,
            "v_pk_add_f32 v[16:17], v[4:5], v[6:7] op_sel:[0,1] op_sel_hi:[1,0]": True,
            # src0 == src1: the horizontal add computes correctly.
            "v_pk_add_f32 v[0:1], v[2:3], v[2:3] op_sel:[0,1] op_sel_hi:[1,0]": False,
            "v_pk_mul_f32 v[0:1], v[2:3], v[4:5] op_sel:[1,0] op_sel_hi:[0,1]": False,
            "v_pk_mul_f32 v[0:1], v[2:3], v[4:5]": False,
            "v_pk_mul_f32 v[0:1], v[2:3], v[4:5] op_sel_hi:[1,0]": False,
            "v_pk_fma_f32 v[4:5], v[4:5], v[6:7], v[16:17] op_sel:[0,1,0]": True,
            "v_pk_fma_f32 v[4:5], v[4:5], v[6:7], v[16:17] op_sel:[0,0,1]": True,
            "v_pk_fma_f32 v[4:5], v[4:5], v[6:7], v[16:17] op_sel:[1,0,0]": False,
            "v_pk_fma_f32 v[4:5], v[4:5], v[6:7], v[16:17] op_sel_hi:[0,1,1]": False,
        }
        for line, invalid in cases.items():
            with self.subTest(line=line):
                # The disassembler appends the encoding as a // comment.
                text = f"\t{line}// 000000001650: D3B1500E 18020D04\n"
                self.assertEqual(bool(invalid_pk_f32_op_sel(text)), invalid)

    def test_matrix_ops(self):
        self.assertTrue(
            has_matrix_ops("\tv_mfma_f32_16x16x16_f16 a[0:3], v[0:1], v[0:1], 0\n")
        )
        self.assertTrue(has_matrix_ops("\tv_mfma_scale_f32_16x16x128_f8f6f4 a[0:3]\n"))
        self.assertTrue(has_matrix_ops("\tv_smfmac_f32_16x16x64_f16 v[0:3]\n"))
        self.assertFalse(has_matrix_ops("\tv_pk_mul_f32 v[0:1], v[2:3], v[4:5]\n"))


class TestWithTargetFeature(unittest.TestCase):
    def test_adds_to_every_defined_function(self):
        ir = _ir("k", _PACKED)
        out = comgr._with_target_feature(ir, "-packed-fp32-ops")
        self.assertIn(
            'attributes #0 = { "amdgpu-flat-work-group-size"="64,256" nounwind '
            '"target-features"="-packed-fp32-ops" }',
            out,
        )
        self.assertEqual(out.replace(' "target-features"="-packed-fp32-ops"', ""), ir)

    def test_extends_existing_features(self):
        ir = _ir("k", _PACKED).replace(
            "nounwind", '"target-features"="+xnack" nounwind'
        )
        out = comgr._with_target_feature(ir, "-packed-fp32-ops")
        self.assertIn('"target-features"="+xnack,-packed-fp32-ops" nounwind', out)

    def test_refuses_a_function_without_a_group(self):
        ir = _ir("k", _PACKED).replace(") #0 {", ") {")
        with self.assertRaisesRegex(comgr.ComgrError, "no attribute group"):
            comgr._with_target_feature(ir, "-packed-fp32-ops")


class TestCodeObjects(unittest.TestCase):
    def test_bare_elf_and_bundle(self):
        elf = b"\x7fELF" + b"\x00" * 12
        self.assertEqual(comgr._amdgpu_code_objects(elf), [elf])
        entries = [
            (b"host-x86_64-unknown-linux-gnu-", b""),
            (b"hipv4-amdgcn-amd-amdhsa--gfx950", elf),
        ]
        head = (
            len(comgr._OFFLOAD_BUNDLE_MAGIC) + 8 + sum(24 + len(i) for i, _ in entries)
        )
        table, payload = b"", b""
        for ident, body in entries:
            table += (
                struct.pack("<QQQ", head + len(payload), len(body), len(ident)) + ident
            )
            payload += body
        blob = (
            comgr._OFFLOAD_BUNDLE_MAGIC
            + struct.pack("<Q", len(entries))
            + table
            + payload
        )
        self.assertEqual(comgr._amdgpu_code_objects(blob), [elf])

    def test_unknown_format_is_refused(self):
        with self.assertRaisesRegex(comgr.ComgrError, "cannot inspect"):
            comgr._amdgpu_code_objects(b"CCOB compressed")


class TestGuardCompiles(unittest.TestCase):
    ISA = "amdgcn-amd-amdhsa--gfx950"

    @classmethod
    def setUpClass(cls):
        info = comgr.loaded_compiler_info()
        if info is None or info.llvm_version is None:
            raise unittest.SkipTest("no queryable COMGR")
        cls.needs_guard = comgr._compiler_needs_pk_f32_guard()

    def raw(self, ir, isa=ISA):
        return comgr._compile_llvm_ir(ir, isa, ["-O3"])[0]

    def test_probe_matches_the_compiler(self):
        """The probe says 'guard' exactly when this compiler emits the bad form."""
        text = comgr.disassemble_hsaco(self.raw(comgr._PKF32_PROBE_IR))
        self.assertEqual(self.needs_guard, bool(invalid_pk_f32_op_sel(text)))

    def test_byte_prefilter_never_misses_an_invalid_form(self):
        """The encoding scan that skips disassembly agrees with the ISA text."""
        for parts in ((_MFMA, _PACKED), (_PACKED,), (_MFMA,)):
            raw = self.raw(_ir("pkf32_prefilter", *parts))
            bad = invalid_pk_f32_op_sel(comgr.disassemble_hsaco(raw))
            with self.subTest(parts=len(parts), bad=len(bad)):
                self.assertEqual(comgr._may_hold_invalid_pk_f32(raw), bool(bad))

    def test_invalid_form_next_to_mfma_is_rebuilt(self):
        ir = _ir("pkf32_guard_mfma", _MFMA, _PACKED)
        before = comgr.disassemble_hsaco(self.raw(ir))
        self.assertTrue(has_matrix_ops(before))
        hsaco, timings = comgr.build_hsaco_from_llvm_ir(ir, isa=self.ISA)
        after = comgr.disassemble_hsaco(hsaco)
        self.assertEqual(invalid_pk_f32_op_sel(after), [])
        self.assertTrue(has_matrix_ops(after))
        if not self.needs_guard:
            self.assertEqual(invalid_pk_f32_op_sel(before), [])
            return
        self.assertTrue(invalid_pk_f32_op_sel(before))
        self.assertNotIn("v_pk_", after.split("Disassembly", 1)[-1])
        self.assertIn("pkf32_guard_mfma", comgr.pk_f32_guard_rebuilds())
        self.assertGreater(timings.pkf32_guard, 0.0)

    def test_kernel_without_mfma_is_untouched(self):
        ir = _ir("pkf32_guard_nomfma", _PACKED)
        raw = self.raw(ir)
        if self.needs_guard:
            self.assertTrue(invalid_pk_f32_op_sel(comgr.disassemble_hsaco(raw)))
        self.assertEqual(comgr.build_hsaco_from_llvm_ir(ir, isa=self.ISA)[0], raw)
        self.assertNotIn("pkf32_guard_nomfma", comgr.pk_f32_guard_rebuilds())

    def test_mfma_kernel_without_invalid_form_is_untouched(self):
        ir = _ir("pkf32_guard_clean", _MFMA)
        self.assertEqual(
            comgr.build_hsaco_from_llvm_ir(ir, isa=self.ISA)[0], self.raw(ir)
        )
        self.assertNotIn("pkf32_guard_clean", comgr.pk_f32_guard_rebuilds())

    def test_other_targets_are_untouched(self):
        isa = "amdgcn-amd-amdhsa--gfx942"
        ir = _ir("pkf32_guard_gfx942", _MFMA, _PACKED)
        self.assertEqual(
            comgr.build_hsaco_from_llvm_ir(ir, isa=isa)[0], self.raw(ir, isa)
        )
        self.assertNotIn("pkf32_guard_gfx942", comgr.pk_f32_guard_rebuilds())


class TestGuardHipcc(unittest.TestCase):
    """compile_kernel_via_hipcc runs the same guard on hipcc's offload bundle."""

    def setUp(self):
        if shutil.which("hipcc") is None:
            raise unittest.SkipTest("hipcc not on PATH")
        if comgr.loaded_compiler_info() is None:
            raise unittest.SkipTest("no COMGR to disassemble with")

    def isa_text(self, blob):
        return "".join(
            comgr.disassemble_hsaco(elf) for elf in comgr._amdgpu_code_objects(blob)
        )

    def test_invalid_form_next_to_mfma_is_rebuilt(self):
        b = IRBuilder("pkf32_guard_hip")
        b.ret()
        built = []
        real_run = compile_module.subprocess.run

        def run(args, **kwargs):
            proc = real_run(args, **kwargs)
            with open(args[args.index("-o") + 1], "rb") as f:
                built.append(f.read())
            return proc

        with (
            mock.patch.object(compile_module, "lower_kernel_to_hip", return_value=_HIP),
            mock.patch.object(compile_module.subprocess, "run", side_effect=run),
        ):
            artifact = compile_module.compile_kernel_via_hipcc(b.kernel, arch="gfx950")
        first = self.isa_text(built[0])
        after = self.isa_text(artifact.hsaco)
        self.assertTrue(has_matrix_ops(after))
        self.assertEqual(invalid_pk_f32_op_sel(after), [])
        if not invalid_pk_f32_op_sel(first):  # a hipcc with the LLVM fix
            self.assertEqual(len(built), 1)
            return
        self.assertEqual(len(built), 2)
        self.assertIn("pkf32_guard_hip", comgr.pk_f32_guard_rebuilds())


if __name__ == "__main__":
    unittest.main()
