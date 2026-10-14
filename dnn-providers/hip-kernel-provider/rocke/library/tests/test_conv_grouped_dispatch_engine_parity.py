# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The gfx950 grouped-conv dispatcher's kernels lower to the same bytes in
both engines (no GPU).

For each request the dispatcher is asked for its spec, and every kernel it
would launch -- the pre-pass, then the main kernel -- is lowered once by the
Python engine and once by the C++ builder through the ``rocke_engine``
binding. This pins the routes that land on the direct kernels (depthwise
tiled forward, depthwise streaming dgrad, small-cpg grouped forward, grouped
dgrad through the weight transpose), the merged depthwise wgrad and the
small-C stem channel pad, as dispatched rather than as hand-written configs.
"""

from __future__ import annotations

import unittest


def _engine():
    """The ``rocke_engine`` binding, or skip the calling test."""
    from rocke.core.backend import BackendError, _import_engine

    try:
        return _import_engine()
    except BackendError as e:
        raise unittest.SkipTest(f"C++ engine not importable: {e}") from None


def _req(**kw):
    from dispatch.grouped_convolution import ConvGroupedRequest

    d = dict(
        N=2, C=144, K=144, Hi=56, Wi=56, Y=3, X=3, arch="gfx950", G=144,
        stride_h=1, stride_w=1, pad_h=1, pad_w=1, dtype="bf16", direction="fwd",
    )  # fmt: skip
    d.update(kw)
    return ConvGroupedRequest(**d)


def _dw(c, hw, k, s, n, direction, dtype="bf16"):
    return _req(
        N=n, C=c, K=c, G=c, Hi=hw, Wi=hw, Y=k, X=k, stride_h=s, stride_w=s,
        pad_h=k // 2, pad_w=k // 2, direction=direction, dtype=dtype,
    )  # fmt: skip


def _grouped(g, c, hw, n, direction, dtype="bf16"):
    return _req(N=n, C=c, K=c, G=g, Hi=hw, Wi=hw, direction=direction, dtype=dtype)


def _stem(c, k, ks, s, hw, n, dtype="bf16"):
    return _req(
        N=n, C=c, K=k, G=1, Hi=hw, Wi=hw, Y=ks, X=ks, stride_h=s, stride_w=s,
        pad_h=ks // 2, pad_w=ks // 2, dtype=dtype,
    )  # fmt: skip


# (label, request, the direct kind / prepass the dispatcher must pick)
_CASES = [
    ("dw fwd 3x3 s2", _dw(144, 56, 3, 2, 32, "fwd"), "dw_tiled"),
    ("dw fwd 7x7 n1 fp16", _dw(96, 56, 7, 1, 1, "fwd", "fp16"), "dw_tiled"),
    ("dw fwd 5x5 s2", _dw(240, 28, 5, 2, 32, "fwd"), "dw_tiled"),
    ("dw dgrad 3x3 s2 n32", _dw(144, 56, 3, 2, 32, "dgrad"), "dw_dgrad_stream"),
    ("dw dgrad 3x3 n1", _dw(144, 56, 3, 1, 1, "dgrad"), "dw_dgrad_stream"),
    ("dw dgrad 7x7 n1 fp16", _dw(96, 56, 7, 1, 1, "dgrad", "fp16"), "dw_dgrad_stream"),
    ("dw dgrad 7x7 7sq n1", _dw(768, 7, 7, 1, 1, "dgrad"), "dw_dgrad_stream"),
    ("dw dgrad 5x5 s2 n8", _dw(240, 28, 5, 2, 8, "dgrad"), "dw_dgrad_stream"),
    ("grouped fwd cpg4", _grouped(32, 128, 28, 32, "fwd"), "grouped"),
    ("grouped fwd cpg16 fp16", _grouped(16, 256, 56, 8, "fwd", "fp16"), "grouped"),
    ("grouped dgrad cpg4", _grouped(32, 128, 56, 32, "dgrad"), "grouped_dgrad"),
    ("grouped dgrad cpg16", _grouped(32, 512, 14, 32, "dgrad"), "grouped_dgrad"),
    ("dw wgrad 3x3 s2 n32", _dw(144, 56, 3, 2, 32, "wgrad"), "wgrad"),
    ("dw wgrad 3x3 14sq n32", _dw(480, 14, 3, 1, 32, "wgrad"), "wgrad"),
    ("dw wgrad 7x7 n1", _dw(96, 56, 7, 1, 1, "wgrad"), "wgrad"),
    ("dw wgrad 5x5 s2 n32 fp16", _dw(240, 28, 5, 2, 32, "wgrad", "fp16"), "wgrad"),
    ("stem C3 7x7 s2", _stem(3, 64, 7, 2, 224, 32), "channel_pad"),
    ("stem C1 7x7 s2 fp16", _stem(1, 64, 7, 2, 224, 32, "fp16"), "channel_pad"),
]

# Dispatcher direct_kind -> conv_direct_grouped binding kind.
_BINDING_KIND = {
    "dw_tiled": "dw_tiled",
    "dw_dgrad_stream": "dw_dgrad_stream",
    "grouped": "generic",
    "grouped_dgrad": "generic",
}


def _python_ll(kernel):
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    return _lower_kernel_to_llvm_python(kernel, arch="gfx950")


def _kernels(req, spec):
    """``[(what, python .ll thunk, engine .ll thunk)]`` for every launch."""
    from dispatch.grouped_convolution import _problem, conv_grouped_prepass
    from kernels.common import conv_direct_grouped as cdg
    from kernels.common.conv_implicit_gemm import build_implicit_gemm_conv
    from kernels.common.conv_implicit_gemm_wgrad import (
        build_implicit_gemm_conv_wgrad,
    )
    from rocke.core import backend as be

    eng = _engine()
    out = []
    pre = conv_grouped_prepass(req, spec)
    if pre is not None:
        if pre.kind == "channel_pad":
            sd = be.conv_channel_pad_spec_to_dict(pre.kernel_spec)
            fn = eng.conv_channel_pad_lower_llvm
        else:
            sd = be.conv_direct_grouped_spec_to_dict(pre.kernel_spec, "transpose")
            fn = eng.conv_direct_grouped_lower_llvm
        out.append(
            (
                pre.kind,
                lambda pre=pre: _python_ll(pre.build("gfx950")),
                lambda fn=fn, sd=sd: fn(sd, "gfx950"),
            )
        )
    if spec.direct_kind:
        kind = _BINDING_KIND[spec.direct_kind]
        ks = spec.to_direct_spec(req)
        build = getattr(cdg, be._CONV_DIRECT_GROUPED_BUILDERS[kind])
        sd = be.conv_direct_grouped_spec_to_dict(ks, kind)
        py_fn = lambda: _python_ll(build(ks, arch="gfx950"))  # noqa: E731
        eng_fn = lambda: eng.conv_direct_grouped_lower_llvm(sd, "gfx950")  # noqa: E731
    elif req.direction == "wgrad":
        from test_conv_wgrad_streamk import _spec_dict

        ws = spec.to_wgrad_spec(_problem(req))
        py_fn = lambda: _python_ll(build_implicit_gemm_conv_wgrad(ws, arch="gfx950"))  # noqa: E731
        eng_fn = lambda: eng.conv_wgrad_lower_llvm(_spec_dict(ws), "gfx950")  # noqa: E731
    else:
        fs = spec.to_fwd_spec(_problem(req))
        sd = be.conv_implicit_gemm_spec_to_dict(fs)
        py_fn = lambda: _python_ll(build_implicit_gemm_conv(fs, arch="gfx950"))  # noqa: E731
        eng_fn = lambda: eng.conv_implicit_gemm_lower_llvm(sd, "gfx950")  # noqa: E731
    out.append((spec.direct_kind or f"igemm {req.direction}", py_fn, eng_fn))
    return out


class TestDispatchedKernelsEngineParity(unittest.TestCase):
    def test_every_dispatched_kernel_is_byte_identical(self):
        from dispatch.grouped_convolution import dispatch_conv_grouped

        _engine()
        for label, req, expect in _CASES:
            spec = dispatch_conv_grouped(req).spec
            with self.subTest(label):
                if expect == "wgrad":
                    self.assertEqual(spec.direct_kind, "")
                    self.assertGreater(spec.group_merge, 1)
                elif expect == "channel_pad":
                    self.assertEqual(spec.prepass, "channel_pad")
                else:
                    self.assertEqual(spec.direct_kind, expect)
                for what, py_fn, eng_fn in _kernels(req, spec):
                    with self.subTest(label=label, kernel=what):
                        self.assertEqual(eng_fn(), py_fn())


if __name__ == "__main__":
    unittest.main(verbosity=2)
