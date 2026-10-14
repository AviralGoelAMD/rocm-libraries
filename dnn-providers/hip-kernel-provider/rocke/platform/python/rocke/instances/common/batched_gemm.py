# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Batched GEMM instance builder (CK Tile ``16_batched_gemm`` parity).

This is the DSL-side counterpart of CK Tile's
``example/ck_tile/16_batched_gemm``. It re-uses the universal GEMM
kernel body verbatim, in batched mode:

    grid = (N_tiles, M_tiles, batch_count)
    kernel reads block_id_z as batch_idx
    A_base = A + batch_idx * stride_a * elem_bytes   (64-bit pointer math)
    B_base = B + batch_idx * stride_b * elem_bytes
    C_base = C + batch_idx * stride_c * elem_bytes
    per-tile indices are i32 element offsets from those bases

The strides are passed as additional ``i32`` kernel arguments (read as
unsigned) so we can support irregular per-batch layouts (e.g. a 3D
``(B, M, K)`` torch tensor where ``stride_a = M*K`` packed, or one with
extra row padding).

Addressing: the batch slab is folded into each base pointer in 64-bit, so
the whole tensor may exceed 2**31 elements. Only the within-slab index
(``row * K + col`` for A/B, ``row * N + col`` for C) is i32: each single
batch slab must stay below 2**31 elements. :func:`check_batched_strides_fit`
validates the stride arguments of a concrete launch.

The MFMA / LDS body is the same as ``build_universal_gemm`` (which is
already battle-tested by the GEMM bake-off). This makes batched GEMM
inherit all of universal_gemm's perf knobs:
``pipeline in {mem, compv3, compv4}``, ``epilogue in {default,
cshuffle}``, warp grids up to 4x4, MFMA atoms 16x16x{16,32} and
32x32x{8,16}.

The per-batch slab offsets are CTA-uniform and are folded into the base
pointers once per CTA (scalar registers), mirroring CK Tile's
``batched_gemm_kernel.hpp:215-230``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from ...core.ir import KernelDef
from ...helpers.spec import WarpTileBlockSizeMixin
from .gemm_universal import (
    DataSpec,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
    is_valid_spec as is_valid_gemm_spec,
    mono_data_spec,
)


@dataclass(frozen=True)
class BatchedGemmSpec(WarpTileBlockSizeMixin):
    """One batched GEMM kernel instance.

    ``batch_size`` is informational (it's an upper bound on the
    ``block_id_z`` dimension; the actual launch grid passes the real
    batch count). It's used by the dispatcher to skip configs that
    won't fit, but the kernel itself doesn't bake it in.
    """

    name: str
    tile: TileSpec
    trait: TraitSpec = field(default_factory=TraitSpec)
    wave_size: int = 64
    block_size: int = 0
    batch_size: int = 0
    dtype: str = "fp16"

    def __post_init__(self) -> None:
        self._init_block_size()

    def _data_spec(self) -> DataSpec:
        return mono_data_spec(self.dtype)

    def to_universal_spec(self) -> UniversalGemmSpec:
        return UniversalGemmSpec(
            name=self.name,
            tile=self.tile,
            trait=self.trait,
            data=self._data_spec(),
            wave_size=self.wave_size,
            block_size=self.block_size,
            batched=True,
        )

    def kernel_name(self) -> str:
        return self.to_universal_spec().kernel_name()


def is_valid_spec(spec: BatchedGemmSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    return is_valid_gemm_spec(spec.to_universal_spec(), arch=arch)


# Largest within-slab element index the kernel's i32 tile arithmetic can hold.
_I32_MAX = (1 << 31) - 1
# The stride arguments are i32 registers zero-extended to i64 in the kernel.
_U32_MAX = (1 << 32) - 1


def check_batched_strides_fit(
    M: int, N: int, K: int, stride_a: int, stride_b: int, stride_c: int
) -> None:
    """Runtime precondition for a batched-GEMM launch.

    The batch slab ``block_id_z * stride_X`` is folded into each base pointer
    in 64-bit, so the batch count is unbounded. Two limits remain:

    * each stride is passed as a 32-bit argument (zero-extended), so it must
      lie in ``[0, 2**32 - 1]``;
    * the within-slab index is i32, so one slab (``M*K`` for A, ``N*K`` for B,
      ``M*N`` for C) must stay ``<= 2**31 - 1`` elements.

    Raises ``ValueError`` on violation so callers fail loudly instead of
    reading or writing the wrong rows.
    """
    for name, s in (
        ("stride_a", stride_a),
        ("stride_b", stride_b),
        ("stride_c", stride_c),
    ):
        if not 0 <= s <= _U32_MAX:
            raise ValueError(f"batched GEMM {name}={s} is outside [0, 2**32-1]")
    for name, elems in (("A", M * K), ("B", N * K), ("C", M * N)):
        if elems > _I32_MAX:
            raise ValueError(
                f"batched GEMM {name} slab of {elems} elements exceeds the i32 "
                f"within-slab index range (2**31-1)"
            )


def build_batched_gemm(spec: BatchedGemmSpec, arch: str = "gfx950") -> KernelDef:
    """Build the IR for one batched GEMM instance.

    Kernel signature:
      ``(A: ptr, B: ptr, C: ptr, M: i32, N: i32, K: i32,
         stride_a: i32, stride_b: i32, stride_c: i32)``

    Grid layout: ``(ceil_div(N, tile_n), ceil_div(M, tile_m), batch)``.
    Block layout: ``(block_size, 1, 1)``.

    ``arch`` selects the target GPU (``"gfx942"`` / ``"gfx950"``). The
    spec is validated against that target's MFMA catalog + LDS cap so a
    warp-tile atom that exists only on gfx950 (e.g. the wide
    ``16x16x32`` f16 atom) is rejected with a structured Python error
    *before* comgr is reached (a gfx950-only atom on gfx942 otherwise
    hard-crashes comgr with ``LLVM ERROR: Cannot select intrinsic``).
    """
    universal = spec.to_universal_spec()
    ok, why = is_valid_gemm_spec(universal, arch=arch)
    if not ok:
        raise ValueError(f"invalid batched_gemm spec for {arch}: {why}")
    return build_universal_gemm(universal, arch=arch)


def build_persistent_batched_gemm(
    spec: BatchedGemmSpec, arch: str = "gfx950"
) -> KernelDef:
    """Persistent grouped-GEMM dispatch (P64).

    Reads ``counts`` / ``offsets`` from device and dispatches all
    ``E`` GEMMs in one persistent kernel via
    :func:`rocke.helpers.persistent.persistent_tile_for_each` —
    eliminates the ``counts.cpu() + torch.cuda.synchronize()`` D→H
    roundtrip in the dynamic MoE forward path. Unblocks HIP-graph
    capture of the dynamic path.

    Reference: CK Tile ``streamk_common.hpp:249-287``
    (``StreamKDispatch``); FlyDSL
    ``mixed_moe_gemm_2stage.py:3215-3252``.

    Minimum-viable: builds the persistent universal-GEMM kernel
    (``trait.persistent=True``); the dynamic ``counts`` /
    ``offsets`` lookup is wired into the host-side launcher (which
    already knows the batch count). The persistent loop body uses
    :class:`rocke.helpers.persistent.persistent_tile_for_each` with
    P35's race-free counter init so this kernel is safe on
    single-wave CTAs at any ``max_iters``.
    """
    universal = spec.to_universal_spec()
    persistent_trait = TraitSpec(
        pipeline=universal.trait.pipeline,
        scheduler=universal.trait.scheduler,
        epilogue=universal.trait.epilogue,
        pad_m=universal.trait.pad_m,
        pad_n=universal.trait.pad_n,
        pad_k=universal.trait.pad_k,
        persistent=True,
        chiplet_swizzle=universal.trait.chiplet_swizzle,
        chiplet_wgm=universal.trait.chiplet_wgm,
        chiplet_num_xcds=universal.trait.chiplet_num_xcds,
        chiplet_chunk_size=universal.trait.chiplet_chunk_size,
        waves_per_eu=universal.trait.waves_per_eu,
    )
    persistent_universal = UniversalGemmSpec(
        name=universal.name + "_persistent",
        tile=universal.tile,
        trait=persistent_trait,
        data=universal.data,
        wave_size=universal.wave_size,
        block_size=universal.block_size,
        batched=universal.batched,
    )
    ok, why = is_valid_gemm_spec(persistent_universal, arch=arch)
    if not ok:
        raise ValueError(f"invalid persistent_batched_gemm spec for {arch}: {why}")
    return build_universal_gemm(persistent_universal, arch=arch)


def batched_gemm_signature(spec: BatchedGemmSpec):
    from ...helpers.spec import SignatureBuilder

    ptr_dt = spec.dtype if spec.dtype in ("f16", "fp16", "bf16") else "f16"
    sig = (
        SignatureBuilder()
        .ptr("A", ptr_dt)
        .ptr("B", ptr_dt)
        .ptr("C", ptr_dt)
        .scalar("M", "i32")
        .scalar("N", "i32")
        .scalar("K", "i32")
        .scalar("stride_a", "i32")
        .scalar("stride_b", "i32")
        .scalar("stride_c", "i32")
    )
    if spec.trait.active_tile_skip:
        sig = sig.ptr("SortedTokenIds", "i32").scalar("slot_size", "i32")
    return sig.build()


def batched_gemm_grid(
    batch: int, m: int, n: int, spec: BatchedGemmSpec
) -> Tuple[int, int, int]:
    from ...helpers.spec import ceil_div_grid

    t = spec.tile
    return ceil_div_grid((n, t.tile_n), (m, t.tile_m), (batch, 1))
