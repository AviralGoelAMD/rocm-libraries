# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Channel zero-pad pre-pass for small-channel (stem) forward convolutions.

A forward implicit GEMM reads its A operand (the NHWC activation) one
``C``-contiguous run per filter tap. When ``C`` is odd -- the RGB stem, ``C=3``
-- that run cannot be loaded as a vector, so every A element costs its own
scalar load plus a full coordinate decode, and the conv is address-bound.

This kernel copies the activation ``X[rows, c_in]`` and the filter
``W[rows, c_in]`` (both NHWC/KYXC, so the channel is the innermost axis of a
flat row) into a workspace with the channel axis zero-padded to ``c_pad``.
The forward conv then runs unchanged on the ``C = c_pad`` problem: the extra
channels are zero in both operands, so they contribute nothing to the sum,
and the A rows are now ``c_pad * 2`` bytes -- a single vector load per tap.

Both tensors are padded in one launch: rows ``[0, x_rows)`` of the flat
thread index copy the activation, rows ``[x_rows, x_rows + w_rows)`` the
filter. Each thread owns one row: ``c_in`` scalar loads, one ``c_pad``-wide
vector store.

Workspace layout (elements of ``dtype``)::

    [0, x_rows * c_pad)                       padded activation
    [x_rows * c_pad, (x_rows + w_rows) * c_pad)  padded filter

The activation segment is ``x_rows * c_pad * 2`` bytes, a multiple of 16 for
``c_pad == 8``, so the filter segment stays 16-byte aligned.

Kernel signature::

    X      : ptr<dtype>, readonly  -- activation [x_rows, c_in]
    W      : ptr<dtype>, readonly  -- filter     [w_rows, c_in]
    Xp     : ptr<dtype>, writeonly -- workspace activation segment [x_rows, c_pad]
    Wp     : ptr<dtype>, writeonly -- workspace filter segment     [w_rows, c_pad]
    x_rows : i32
    w_rows : i32

Grid: ``(ceil((x_rows + w_rows) / block_size), 1, 1)``; block ``(block_size, 1, 1)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from rocke.core.ir import I32, IRBuilder, KernelDef, PtrType
from rocke.helpers.io import io_ir_type, load_scalar
from rocke.helpers.spec import SignatureBuilder, kernel_name_join


@dataclass(frozen=True)
class ChannelPadSpec:
    """One channel zero-pad kernel: ``c_in`` -> ``c_pad`` innermost elements."""

    c_in: int
    c_pad: int
    dtype: str = "bf16"
    block_size: int = 256
    name: str = "rocke_conv_channel_pad"

    def kernel_name(self) -> str:
        return kernel_name_join(
            self.name, f"c{self.c_in}to{self.c_pad}", self.dtype, f"b{self.block_size}"
        )


def is_valid_channel_pad_spec(spec: ChannelPadSpec) -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for ``spec``."""
    if spec.dtype not in ("fp16", "bf16"):
        return False, f"unsupported dtype {spec.dtype!r}; fp16 or bf16 only"
    if spec.c_pad not in (2, 4, 8):
        return False, f"c_pad must be 2, 4 or 8 (one vector store), got {spec.c_pad}"
    if not 1 <= spec.c_in < spec.c_pad:
        return False, f"need 1 <= c_in < c_pad, got c_in={spec.c_in} c_pad={spec.c_pad}"
    if spec.block_size <= 0 or spec.block_size % 64:
        return False, f"block_size must be a positive multiple of 64, got {spec.block_size}"
    return True, "ok"


def build_conv_channel_pad(spec: ChannelPadSpec, *, arch: str = "gfx950") -> KernelDef:
    """Build the IR for one channel zero-pad kernel (arch-neutral)."""
    del arch  # no arch-specific instruction is emitted
    ok, why = is_valid_channel_pad_spec(spec)
    if not ok:
        raise ValueError(f"invalid channel-pad spec: {why}")
    elem = io_ir_type(spec.dtype)
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block_size
    X = b.param("X", PtrType(elem, "global"), noalias=True, readonly=True, align=2)
    W = b.param("W", PtrType(elem, "global"), noalias=True, readonly=True, align=2)
    Xp = b.param("Xp", PtrType(elem, "global"), noalias=True, writeonly=True, align=16)
    Wp = b.param("Wp", PtrType(elem, "global"), noalias=True, writeonly=True, align=16)
    x_rows = b.param("x_rows", I32)
    w_rows = b.param("w_rows", I32)

    row = b.add(b.mul(b.block_id_x(), b.const_i32(spec.block_size)), b.thread_id_x())
    zero = b.cast_f32_to(b.const_f32(0.0), elem)
    c_in = b.const_i32(spec.c_in)
    c_pad = b.const_i32(spec.c_pad)

    def copy_row(src, dst, r) -> None:
        base = b.mul(r, c_in)
        vals = [
            load_scalar(
                b, src, base if c == 0 else b.add(base, b.const_i32(c)), dtype=spec.dtype
            )
            for c in range(spec.c_in)
        ]
        vals += [zero] * (spec.c_pad - spec.c_in)
        b.global_store_vN(
            dst, b.mul(r, c_pad), b.vec_pack(vals, elem), spec.c_pad, align=2 * spec.c_pad
        )

    with b.scf_if(b.cmp_lt(row, x_rows)):
        copy_row(X, Xp, row)
    w_row = b.sub(row, x_rows)
    in_w = b.land(b.cmp_ge(row, x_rows), b.cmp_lt(w_row, w_rows))
    with b.scf_if(in_w):
        copy_row(W, Wp, w_row)
    return b.kernel


def channel_pad_grid(spec: ChannelPadSpec, x_rows: int, w_rows: int) -> Tuple[int, int, int]:
    """Launch grid for ``x_rows + w_rows`` rows."""
    total = x_rows + w_rows
    return ((total + spec.block_size - 1) // spec.block_size, 1, 1)


def channel_pad_signature(spec: ChannelPadSpec) -> list:
    """Kernel signature list for :class:`KernelLauncher`."""
    return (
        SignatureBuilder()
        .ptr("X", spec.dtype)
        .ptr("W", spec.dtype)
        .ptr("Xp", spec.dtype)
        .ptr("Wp", spec.dtype)
        .scalar("x_rows", "i32")
        .scalar("w_rows", "i32")
        .build()
    )
