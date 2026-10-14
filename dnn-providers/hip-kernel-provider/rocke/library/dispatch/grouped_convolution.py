# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Grouped convolution dispatcher (forward, backward-weight, backward-data).

Covers ``implicit_gemm_conv`` (forward NHWC × KYXC → NHWK),
``implicit_gemm_conv_wgrad`` (dY × X → dW weight gradient) and
``implicit_gemm_conv_dgrad`` (dY × W → dX input gradient), sharing a single
``ConvGroupedRequest`` so callers can dispatch all three directions from the
same shape description.

Coverage is not uniform across the three. Forward and wgrad each have gfx942,
gfx950 and gfx1250 candidates; dgrad has gfx950 only, and pins split_k=1.

SCOPE -- what this dispatcher decides
-------------------------------------
Each candidate commits to a fixed set of tile / warp / pipeline / epilogue
parameters.  All values are hard-coded for now (sweep results TBD).

The epilogue is derived from ``vec_size_c``, which is computed by the same
static heuristic the kernel builder uses
(``ImplicitGemmConvSpec.default_vector_sizes``):

    vec_size_c = largest power-of-two factor of K (fp16/bf16) or K (fp32)
    epilogue   = "cshuffle"  if vec_size_c > 1
    epilogue   = "default"   otherwise

Three arch families are supported, each with its own hard-coded tile config:

* **gfx942** — wave64, MFMA 16×16×16 atom (largest fp16/bf16 atom available):
      tile 64×64×64, warp 2×2, atom 16×16×16, pipeline ``mem``
* **gfx950** — wave64, MFMA 32×32×16 atom:
      tile 64×64×64, warp 2×2, atom 32×32×16, pipeline ``mem``
* **gfx1250** (wave32, WMMA) — WMMA 16×16×32 atom:
      tile 32×32×32, warp 2×2, atom 16×16×32, pipeline ``mem``
  (gfx1250 restricts WMMA conv to pipeline=``mem``, groups=1, no cshuffle
  override, and wgrad is not yet supported)

DEFERRED -- per-problem divisibility and MMA atom selection
------------------------------------------------------------
``is_valid_spec`` / ``is_valid_wgrad_spec`` perform the arch-aware MMA-atom
and LDS-budget checks at the instance level.  Those checks are delegated to
the existing validators rather than duplicated here.


GEOMETRY SELECTION -- approaches for replacing hard-coded tiles
---------------------------------------------------------------
The candidates below use a single fixed tile configuration per arch family.
This section documents the planned approaches for replacing those constants
with data-driven or learned selection.

A key constraint: conv has more independent dimensions than GEMM.  Two
problems with the same implicit-GEMM shape ``(M, N_gemm, K_gemm)`` can have
very different performance characteristics if they differ in how those dims
are composed — e.g. a large ``Y*X`` vs a large ``C`` in ``K_gemm`` changes
the address-computation pattern and L1/L2 reuse completely.  Any selection
strategy must either expose these raw dims as features or bucket them
conservatively enough that the composition differences are irrelevant.

**Approach A — offline sweep + lookup table (simplest)**

Run ``benchmark_implicit_gemm_conv.py`` on a representative set of shapes
(e.g. ``bench_cases_conv.json``) for each target arch and dtype.  Record the
best ``(tile_m, tile_n, tile_k, warp_m, warp_n, warp_tile_mn, pipeline,
epilogue)`` per shape.  At dispatch time, find the nearest entry in the table
by bucketing the implicit-GEMM dims to the nearest power-of-two:

    M_bucket  = prev_power_of_2(N * Ho * Wo)
    N_bucket  = prev_power_of_2(K)
    K_bucket  = prev_power_of_2(Y * X * C)
    key       = (arch, dtype, M_bucket, N_bucket, K_bucket)

Limitations: bucketing loses the Y*X vs C composition information; a new
arch or dtype requires a fresh sweep; the table must be re-measured whenever
kernel codegen changes.

**Approach B — analytic heuristics**

Choose tiles based on arithmetic intensity and expected occupancy without
measuring.  Example rules:

* If ``M * N_gemm`` is small (< 4096) the 2D grid is too thin to saturate
  the device — prefer a larger tile in the larger dim to reduce wave count.
* If ``K_gemm`` is small (< 64) there is little K-loop work; use a smaller
  ``tile_k`` and the simpler ``mem`` pipeline.
* If ``K % 8 == 0`` use cshuffle with ``vec_size_c = 8`` for wide stores.
* For wgrad, if ``K_wg = N * Ho * Wo`` is large relative to the ``M * N_wg``
  tile area, increase ``split_k`` to saturate the device.

Limitations: heuristics require careful calibration; they are brittle on
edge-cases and need re-tuning when new micro-arch behaviours are found.

**Approach C — online sweep + persistent cache (recommended next step)**

``sweep_space(req)`` already returns all valid ``ConvGroupedSpec`` instances
for a request.  A thin caching layer around ``dispatch_conv_grouped`` can:

1. Hash the request with ``stable_json_hash(req.normalized())`` (already
   computed as ``KernelId.request_hash``).
2. Look up the hash in a JSON/SQLite cache on disk.
3. On a cache miss: compile and time all specs from ``sweep_space``, write
   the winner to the cache, return it.
4. On a cache hit: return the cached spec directly (pure CPU, no GPU needed).

This is equivalent to MIOpen's find-mode and CK's profiler cache, adapted to
the rocke dispatch contract.  The cache file path can be controlled by an env
var (e.g. ``ROCKE_CONV_CACHE``).  The ``Ranker`` hook on
``dispatch_conv_grouped`` provides the injection point: a ``CachingRanker``
can implement steps 2–4 without touching the candidate code.

**Approach D — ML heuristic (long-term)**

Train a small model (gradient-boosted tree or a 2–3 layer MLP) to predict
the best tile config directly from problem features.  The training data comes
from approach A/C sweep results.

Feature vector (all scalar, normalised to log scale):

    [M, N_gemm, K_gemm,       -- implicit-GEMM dims
     N, Ho, Wo,               -- spatial decomposition of M
     K,                       -- = N_gemm
     Y, X, C,                 -- decomposition of K_gemm
     G,                       -- groups
     sH, sW, pH, pW, dH, dW,  -- conv geometry
     dtype_id,                -- 0=fp16, 1=bf16, 2=fp32
     arch_id]                 -- 0=gfx942, 1=gfx950, 2=gfx1250

Target: one-hot over the discrete config space
``{tile_m, tile_n, tile_k, warp_m, warp_n, warp_tile_mn, pipeline}``.
Epilogue is always derived from ``vec_size_c`` (not predicted).

The model inference is a few hundred microseconds on CPU — negligible vs
kernel compile time, and can be cached by ``request_hash`` like approach C.

Integration: ship the trained model weights as a small binary blob alongside
the dispatcher; load lazily on first call.  A ``MLRanker`` implementing the
``Ranker`` protocol re-orders candidates by predicted TFLOPS before
``CandidateRegistry.select`` picks the first supported one.  Fallback to the
hard-coded defaults if the model is absent or predicts an invalid config.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Optional, Sequence, Tuple

from rocke.core.arch import ArchTarget
from kernels.common.conv_implicit_gemm import (
    ConvDataSpec,
    ConvProblem,
    ImplicitGemmConvSpec,
    is_valid_spec as _fwd_is_valid_spec,
)
from kernels.common.conv_implicit_gemm_wgrad import (
    WgradConvSpec,
    _DEFAULT_WS_REPLICAS as _WGRAD_WS_REPLICAS,
    _GROUP_MERGE_DEGREES,
    is_valid_wgrad_spec as _wgrad_is_valid_spec,
    wgrad_atomic_epilogue_available as _wgrad_atomic_epilogue_available,
)
from kernels.common.conv_implicit_gemm_dgrad import (
    DgradConvSpec,
    is_valid_dgrad_spec as _dgrad_is_valid_spec,
)
from kernels.common.conv_channel_pad import (
    ChannelPadSpec,
    build_conv_channel_pad,
    channel_pad_grid,
    channel_pad_signature,
)
from kernels.common.conv_direct_grouped import (
    DirectConvProblem,
    DirectConvSpec,
    DirectDepthwiseDgradStreamSpec,
    DirectDepthwiseTiledSpec,
    DirectTransposeWeightsDgradSpec,
    build_direct_transpose_weights_dgrad,
    direct_dgrad_workspace_bytes,
    direct_launch_geometry,
    is_valid_depthwise_dgrad_stream_spec,
    is_valid_depthwise_tiled_spec,
    is_valid_spec as _direct_is_valid_spec,
    make_dgrad_fprop_spec,
)
from rocke.dispatch.core import (
    Capability,
    CandidateRegistry,
    DispatchResult,
    KernelCandidate,
    KernelId,
    OperatorRequest,
    Ranker,
    stable_json_hash,
    selector_matches,
)

# ---------------------------------------------------------------------------
# Family / ABI constants
# ---------------------------------------------------------------------------

_FAMILY_FWD = "conv_implicit_gemm"
_FAMILY_WGRAD = "conv_implicit_gemm_wgrad"
_FAMILY_DGRAD = "conv_implicit_gemm_dgrad"

CONV_GROUPED_ABI_VERSION = "hipkg-conv-grouped/v1"

# ---------------------------------------------------------------------------
# Hard-coded tile parameters (to be replaced by sweep-derived tuning tables)
# ---------------------------------------------------------------------------

# gfx950 — wave64, MFMA 32x32x16 (gfx942 lacks this fp16 atom)
_GFX950_TILE_M = 64
_GFX950_TILE_N = 64
_GFX950_TILE_K = 64
_GFX950_WARP_M = 2
_GFX950_WARP_N = 2
_GFX950_WARP_TILE_MN = 32
_GFX950_WARP_TILE_K = 16

# gfx942 — wave64, MFMA 16x16x16 (largest fp16/bf16 atom available)
_GFX942_TILE_M = 64
_GFX942_TILE_N = 64
_GFX942_TILE_K = 64
_GFX942_WARP_M = 2
_GFX942_WARP_N = 2
_GFX942_WARP_TILE_MN = 16
_GFX942_WARP_TILE_K = 16

# gfx950 wgrad, scalar-B variant — wave64, MFMA 16x16x32, 8 waves per CTA.
#
# Taken when the X (B-operand) free-axis load cannot vectorise at all, i.e.
# ``cpg`` is odd so ``vec_b == 1`` -- the small-channel "stem" convs (C=1, C=3).
# There every B element costs its own full coordinate decode (unmerge k_wg into
# (n, ho, wo) with two magic divides, embed to (h, w), bounds-check), so the
# wgrad hot loop is address arithmetic, not memory or MFMA.  The cost per thread
# scales with the B elements it owns, ``tile_n * tile_k / block_size``, so the
# lever is to halve the B tile (the padded ``tile_n=64`` is mostly waste when
# ``wg_N = Y*X*cpg`` is small anyway) and double the threads.  The 16-wide atom
# is what makes 8 waves fit a 64x32 tile; the 32-wide one caps it at 2x2.
_GFX950_WGRAD_SCALARB_TILE_M = 64
_GFX950_WGRAD_SCALARB_TILE_N = 32
_GFX950_WGRAD_SCALARB_TILE_K = 64
_GFX950_WGRAD_SCALARB_WARP_M = 4
_GFX950_WGRAD_SCALARB_WARP_N = 2
_GFX950_WGRAD_SCALARB_WARP_TILE_MN = 16
_GFX950_WGRAD_SCALARB_WARP_TILE_K = 32
# The MFMA-clustering schedule. It helps this address-bound tile and does
# nothing on the vectorised one, which is why it rides with the variant rather
# than replacing _PIPELINE.
_GFX950_WGRAD_SCALARB_PIPELINE = "compv3"

# There is deliberately no tuned constant for depthwise wgrad group merging:
# :func:`_wgrad_merge_degree` is the kernel's own admissibility rule and nothing
# more. See its docstring for why no knob survived the fit.

# gfx1250 — wave32, WMMA 16x16x32; pipeline must be "mem", groups=1 only
_GFX1250_TILE_M = 32
_GFX1250_TILE_N = 32
_GFX1250_TILE_K = 32
_GFX1250_WARP_M = 2
_GFX1250_WARP_N = 2
_GFX1250_WARP_TILE_MN = 16
_GFX1250_WARP_TILE_K = 32

_PIPELINE = "mem"

# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConvGroupedRequest(OperatorRequest):
    """Normalized grouped convolution request (fwd or wgrad, NHWC)."""

    N: int
    C: int
    K: int
    Hi: int
    Wi: int
    Y: int
    X: int
    arch: str
    G: int = 1
    stride_h: int = 1
    stride_w: int = 1
    pad_h: int = 0
    pad_w: int = 0
    dilation_h: int = 1
    dilation_w: int = 1
    dtype: str = "fp16"
    layout: str = "NHWC"
    # "fwd" | "wgrad"
    direction: str = "fwd"
    # 3D depth fields — all must be set together or all left as None (2D)
    Di: Optional[int] = None
    Z: Optional[int] = None
    stride_d: Optional[int] = None
    pad_d: Optional[int] = None
    dilation_d: Optional[int] = None
    # optional vec_size_c override; None = let the candidate decide
    vec_size_c: Optional[int] = None
    op: str = "conv_grouped"
    algorithm: str = "auto"
    spec_id: str = "auto"

    def normalized(self) -> dict:
        d = asdict(self)
        d["dtype"] = d["dtype"].lower()
        d["layout"] = d["layout"].upper()
        d["direction"] = d["direction"].lower()
        return d

    def dims(self) -> dict[str, int]:
        d = {
            name: int(getattr(self, name))
            for name in (
                "N",
                "C",
                "K",
                "Hi",
                "Wi",
                "Y",
                "X",
                "G",
                "stride_h",
                "stride_w",
                "pad_h",
                "pad_w",
                "dilation_h",
                "dilation_w",
            )
        }
        if self.Di is not None:
            for name in ("Di", "Z", "stride_d", "pad_d", "dilation_d"):
                v = getattr(self, name)
                if v is not None:
                    d[name] = int(v)
        try:
            p = _problem(self)
            d.update(Ho=int(p.Ho), Wo=int(p.Wo))
            if p.is_3d:
                d["Do"] = int(p.Do)
        except Exception:
            pass
        return d


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _problem(req: ConvGroupedRequest) -> ConvProblem:
    return ConvProblem(
        N=int(req.N),
        Hi=int(req.Hi),
        Wi=int(req.Wi),
        C=int(req.C),
        K=int(req.K),
        Y=int(req.Y),
        X=int(req.X),
        sH=int(req.stride_h),
        sW=int(req.stride_w),
        pH=int(req.pad_h),
        pW=int(req.pad_w),
        dH=int(req.dilation_h),
        dW=int(req.dilation_w),
        groups=int(req.G),
        Di=int(req.Di) if req.Di is not None else None,
        Z=int(req.Z) if req.Z is not None else None,
        sD=int(req.stride_d) if req.stride_d is not None else None,
        pD=int(req.pad_d) if req.pad_d is not None else None,
        dD=int(req.dilation_d) if req.dilation_d is not None else None,
    )


def _direct_problem(req: ConvGroupedRequest) -> Optional[DirectConvProblem]:
    """The request as a direct-conv problem, or None when the direct kernels
    cannot express it: they are 2-D with one stride, one padding and no
    dilation for both spatial axes."""
    if (
        req.Di is not None
        or req.stride_h != req.stride_w
        or req.pad_h != req.pad_w
        or req.dilation_h != 1
        or req.dilation_w != 1
        or req.C % req.G
        or req.K % req.G
    ):
        return None
    return DirectConvProblem(
        N=int(req.N),
        H=int(req.Hi),
        W=int(req.Wi),
        groups=int(req.G),
        cpg=int(req.C) // int(req.G),
        kpg=int(req.K) // int(req.G),
        KH=int(req.Y),
        KW=int(req.X),
        PAD=int(req.pad_h),
        stride=int(req.stride_h),
        dtype=req.dtype.lower(),
    )


def _request_errors(req: OperatorRequest) -> list[str]:
    if not isinstance(req, ConvGroupedRequest):
        return [f"expected ConvGroupedRequest, got {type(req).__name__}"]
    errors: list[str] = []
    if req.op != "conv_grouped":
        errors.append(f"unsupported op {req.op!r}")
    if req.direction not in ("fwd", "wgrad", "dgrad"):
        errors.append(
            f"direction must be 'fwd', 'wgrad' or 'dgrad', got {req.direction!r}"
        )
    for field_name in ("N", "C", "K", "Hi", "Wi", "Y", "X"):
        if int(getattr(req, field_name)) <= 0:
            errors.append(f"{field_name} must be positive")
    if int(req.G) <= 0:
        errors.append("G (groups) must be positive")
    if req.dtype.lower() not in ("fp16", "bf16"):
        errors.append(f"unsupported dtype {req.dtype!r}; fp16 or bf16 only")
    if req.layout.upper() != "NHWC":
        errors.append(f"unsupported layout {req.layout!r}; NHWC only")
    try:
        ArchTarget.from_gfx(req.arch)
    except KeyError as e:
        errors.append(str(e))
    _3d_fields = {
        "Di": req.Di,
        "Z": req.Z,
        "stride_d": req.stride_d,
        "pad_d": req.pad_d,
        "dilation_d": req.dilation_d,
    }
    _set_3d = {k for k, v in _3d_fields.items() if v is not None}
    if _set_3d and _set_3d != set(_3d_fields):
        missing = sorted(set(_3d_fields) - _set_3d)
        errors.append(f"3D fields must all be set or all be None; missing: {missing}")
    if errors:
        return errors
    p = _problem(req)
    if p.Ho <= 0 or p.Wo <= 0:
        errors.append(
            f"degenerate output spatial dims Ho={p.Ho} Wo={p.Wo} "
            "(filter larger than padded input)"
        )
    if p.is_3d and p.Do <= 0:
        errors.append(
            f"degenerate output depth Do={p.Do} "
            "(filter larger than padded input depth)"
        )
    return errors


def _vec_size_c(req: ConvGroupedRequest) -> int:
    """Compute vec_size_c the same way the kernel builder does.

    Forward pass (D=NHWK, last dim K):  uses ImplicitGemmConvSpec, returns _vec(K).
    Wgrad       (D=KYXC, last dim C):   uses WgradConvSpec,          returns _vec(C).
    Dgrad       (D=NHWC, last dim C):   uses DgradConvSpec,          returns _vec(C).

    Each direction has its own ``default_vector_sizes``; they are not
    interchangeable. Dgrad's takes the *per-group* runs ``(cpg, kpg)`` rather
    than ``(C, K)``, so falling through to the forward formula here would size
    the store vector off the wrong extent on a grouped problem.
    """
    if req.vec_size_c is not None:
        return req.vec_size_c
    if req.direction == "wgrad":
        _va, _vb, vc = WgradConvSpec.default_vector_sizes(
            req.C, req.K, req.dtype.lower(), split_k=1
        )
        return vc
    if req.direction == "dgrad":
        p = _problem(req)
        _va, _vb, vc = DgradConvSpec.default_vector_sizes(
            p.cpg, p.kpg, req.dtype.lower()
        )
        return vc
    _va, _vb, vc = ImplicitGemmConvSpec.default_vector_sizes(
        req.C, req.K, req.dtype.lower()
    )
    return vc


def _epilogue_for(req: ConvGroupedRequest) -> str:
    """Derive epilogue from vec_size_c: cshuffle when >1, default otherwise."""
    return "cshuffle" if _vec_size_c(req) > 1 else "default"


def _wgrad_grouped_overrides(req: ConvGroupedRequest) -> Tuple[str, int]:
    """(epilogue, split_k) for a wgrad spec.

    Grouping is orthogonal to the epilogue and split-K: every epilogue (direct,
    cshuffle, and the split-K atomic path) threads the per-group fold
    ``k_out += group*kpg`` into the dW store, and when split_k>1 the group rides
    block_id_z alongside the K-slice (z = groups*split_k).  So grouped uses the
    same vec-derived epilogue and auto split-K formula (split_k=-1) as ungrouped.
    (WMMA grouped is handled by its own candidate, which forces
    default@split_k=1 -- WMMA has neither a cshuffle nor a split-K path.)
    """
    return _epilogue_for(req), -1


def _data_spec(req: ConvGroupedRequest) -> ConvDataSpec:
    return ConvDataSpec(
        dtype_a=req.dtype.lower(),
        dtype_b=req.dtype.lower(),
        dtype_d=req.dtype.lower(),
    )


def _is_gfx942(req: ConvGroupedRequest) -> bool:
    return req.arch == "gfx942"


def _is_gfx950(req: ConvGroupedRequest) -> bool:
    return req.arch == "gfx950"


def _is_gfx1250(req: ConvGroupedRequest) -> bool:
    return req.arch == "gfx1250"


# ---------------------------------------------------------------------------
# Spec type returned by select_spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConvGroupedSpec:
    """Selected spec for a grouped conv candidate (fwd or wgrad)."""

    direction: str  # "fwd" | "wgrad" | "dgrad"
    tile_m: int
    tile_n: int
    tile_k: int
    warp_m: int
    warp_n: int
    warp_tile_mn: int
    warp_tile_k: int
    pipeline: str
    epilogue: str
    dtype: str
    arch: str
    split_k: int = 1  # wgrad only
    lds_k_outer: bool = False  # wgrad only
    group_merge: int = 1  # fwd and wgrad, depthwise only
    name: str = "rocke_conv_grouped"
    # Direct (non-GEMM) kernel families: "" selects the implicit-GEMM kernels
    # the tile/warp/atom fields above describe. Otherwise one of
    # _DIRECT_KINDS, with ``direct_tile`` holding that family's tile knobs
    # (see :meth:`to_direct_spec`); the GEMM fields are then unused.
    direct_kind: str = ""
    direct_tile: Tuple[int, ...] = ()
    # fwd implicit GEMM only: when non-zero, the conv runs on the problem with
    # the input-channel axis zero-padded to ``pad_c`` (a C=3 stem padded to 8),
    # fed by the channel-pad pre-pass. See :func:`conv_grouped_prepass`.
    pad_c: int = 0

    @property
    def prepass(self) -> str:
        """The pre-pass this spec needs before its main kernel, or "".

        * ``"channel_pad"`` -- copy X and W into a workspace with the channel
          axis zero-padded to ``pad_c``; the implicit GEMM reads the workspace.
        * ``"weight_transpose"`` -- the grouped dgrad direct kind: transpose W
          into a workspace; the direct forward kernel reads it as its filter.
        """
        if self.pad_c:
            return "channel_pad"
        if self.direct_kind == "grouped_dgrad":
            return "weight_transpose"
        return ""

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        if self.direct_kind:
            return kernel_name_join(
                self.name,
                self.direction,
                self.dtype,
                self.direct_kind,
                "t" + "x".join(str(v) for v in self.direct_tile),
            )
        parts = [
            self.direction,
            self.dtype,
            f"tile{self.tile_m}x{self.tile_n}x{self.tile_k}",
            f"warp{self.warp_m}x{self.warp_n}",
            f"atom{self.warp_tile_mn}x{self.warp_tile_mn}x{self.warp_tile_k}",
            self.pipeline,
            self.epilogue,
        ]
        if self.direction in ("wgrad", "dgrad") and self.split_k != 1:
            parts.append(f"spk{self.split_k}")
        # These two change the emitted body, so they have to reach the name --
        # this is the layer whose names key the host-side compile cache, and the
        # instance-level WgradConvSpec.kernel_name() already tags both.
        #   lds_k_outer: different LDS tile shape and a transpose-read operand
        #     fetch rather than a transpose-on-store.
        #   group_merge: a different GEMM entirely -- Gm groups folded into one
        #     tile, with a diagonal mask in the epilogue.
        if self.direction in ("wgrad", "dgrad") and self.lds_k_outer:
            parts.append("kouter")
        if self.direction in ("fwd", "wgrad") and self.group_merge > 1:
            parts.append(f"gm{self.group_merge}")
        # A different problem (C = pad_c) and a different launch contract (a
        # workspace and a pre-pass), so it has to key the cache separately.
        if self.pad_c:
            parts.append(f"padc{self.pad_c}")
        return kernel_name_join(self.name, *parts)

    def to_fwd_spec(self, problem: "ConvProblem") -> "ImplicitGemmConvSpec":
        """Build an ImplicitGemmConvSpec from this dispatcher spec and a ConvProblem.

        ``problem`` is the request's problem; a ``pad_c`` spec builds the kernel
        for the channel-padded one, which is what its main kernel computes.
        """
        assert self.direction == "fwd", "to_fwd_spec is only valid for fwd specs"
        target = ArchTarget.from_gfx(self.arch)
        if self.pad_c:
            problem = replace(problem, C=self.pad_c)
        return ImplicitGemmConvSpec(
            problem=problem,
            name=self.name,
            data=ConvDataSpec(
                dtype_a=self.dtype,
                dtype_b=self.dtype,
                dtype_d=self.dtype,
            ),
            tile_m=self.tile_m,
            tile_n=self.tile_n,
            tile_k=self.tile_k,
            warp_m=self.warp_m,
            warp_n=self.warp_n,
            warp_tile_m=self.warp_tile_mn,
            warp_tile_n=self.warp_tile_mn,
            warp_tile_k=self.warp_tile_k,
            wave_size=target.wave_size,
            pipeline=self.pipeline,
            epilogue=self.epilogue,
            groups=problem.groups,
            group_merge=self.group_merge,
        )

    def to_wgrad_spec(self, problem: "ConvProblem") -> "WgradConvSpec":
        """Build a WgradConvSpec with a resolved split_k from this dispatcher spec.

        split_k=-1 (auto) is resolved here via the CK formula so the returned
        spec always has a concrete split_k >= 1, ready for grid calculation and
        launch without re-running the formula in the caller.
        """
        assert self.direction == "wgrad", "to_wgrad_spec is only valid for wgrad specs"
        target = ArchTarget.from_gfx(self.arch)
        resolved_split_k, two_stage, _ = _resolve_wgrad_split_k(self, problem)
        return WgradConvSpec(
            problem=problem,
            name=self.name,
            lds_k_outer=self.lds_k_outer,
            data=ConvDataSpec(
                dtype_a=self.dtype,
                dtype_b=self.dtype,
                dtype_d=self.dtype,
            ),
            tile_m=self.tile_m,
            tile_n=self.tile_n,
            tile_k=self.tile_k,
            warp_m=self.warp_m,
            warp_n=self.warp_n,
            warp_tile_m=self.warp_tile_mn,
            warp_tile_n=self.warp_tile_mn,
            warp_tile_k=self.warp_tile_k,
            wave_size=target.wave_size,
            pipeline=self.pipeline,
            # two_stage writes one f32 per element via workspace store; cshuffle
            # is not used and would produce an invalid spec (validator rejects it).
            epilogue="default" if two_stage else self.epilogue,
            split_k=resolved_split_k,
            two_stage=two_stage,
            group_merge=self.group_merge,
            # Pin the replica count explicitly rather than inheriting the
            # dataclass default: _resolve_wgrad_split_k caps the scratch against
            # the i32 ws_bytes ABI using this same constant, and a cap computed
            # over a different R than the spec carries bounds nothing.
            ws_replicas=_WGRAD_WS_REPLICAS,
        )

    def to_dgrad_spec(self, problem: "ConvProblem") -> "DgradConvSpec":
        """Build a DgradConvSpec from this dispatcher spec and a ConvProblem.

        No split-K auto-resolution counterpart to :meth:`to_wgrad_spec`: dgrad's
        reduction is ``Y*X*K``, which is not the lopsided axis wgrad's ``N*Ho*Wo``
        is, so the CK formula that rescues wgrad's grid does not apply and the
        candidate pins a concrete degree instead.
        """
        assert self.direction == "dgrad", "to_dgrad_spec is only valid for dgrad specs"
        target = ArchTarget.from_gfx(self.arch)
        return DgradConvSpec(
            problem=problem,
            name=self.name,
            lds_k_outer=self.lds_k_outer,
            data=ConvDataSpec(
                dtype_a=self.dtype,
                dtype_b=self.dtype,
                dtype_d=self.dtype,
            ),
            tile_m=self.tile_m,
            tile_n=self.tile_n,
            tile_k=self.tile_k,
            warp_m=self.warp_m,
            warp_n=self.warp_n,
            warp_tile_m=self.warp_tile_mn,
            warp_tile_n=self.warp_tile_mn,
            warp_tile_k=self.warp_tile_k,
            wave_size=target.wave_size,
            pipeline=self.pipeline,
            epilogue=self.epilogue,
            split_k=self.split_k,
        )

    def to_direct_spec(self, req: "ConvGroupedRequest"):
        """The direct-conv kernel spec for a ``direct_kind`` dispatcher spec."""
        if not self.direct_kind:
            raise ValueError("to_direct_spec is only valid for direct specs")
        p = _direct_problem(req)
        t = self.direct_tile
        if self.direct_kind == "dw_tiled":
            return DirectDepthwiseTiledSpec(
                problem=p,
                name=self.name,
                block_w=t[0],
                block_h=t[1],
                block_waves=t[2],
                unroll_rows=True,
            )
        if self.direct_kind == "dw_dgrad_stream":
            return DirectDepthwiseDgradStreamSpec(
                problem=p, name=self.name, block_w=t[0], block_waves=t[1]
            )
        if self.direct_kind == "grouped":
            return DirectConvSpec(
                problem=p,
                name=self.name,
                block_q=t[0],
                block_groups=t[1],
                double_buffer=bool(t[2]),
            )
        if self.direct_kind == "grouped_dgrad":
            # dgrad as the forward direct kernel over (dY, W^T): the spec
            # describes the transposed problem, and the weight-transpose
            # pre-pass supplies W^T in the workspace.
            return replace(
                make_dgrad_fprop_spec(
                    p, block_q=t[0], block_groups=t[1], block_h=t[2]
                ),
                name=self.name,
            )
        raise ValueError(f"unknown direct_kind {self.direct_kind!r}")


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------


def _fwd_grid(spec: ConvGroupedSpec, req: OperatorRequest) -> Tuple[int, int, int]:
    assert isinstance(req, ConvGroupedRequest)
    p = _problem(req)
    # Launched over the MERGED dims: under group_merge the tile covers
    # N_gemm * Gm columns and groups / Gm groups exist (the same computation as
    # ImplicitGemmConvSpec.grid_N_gemm / grid_groups). Identical at Gm == 1.
    merge = max(1, int(spec.group_merge))
    gm = (p.M + spec.tile_m - 1) // spec.tile_m
    gn = (p.N_gemm * merge + spec.tile_n - 1) // spec.tile_n
    # grid_order "NM": x=n-tiles, y=m-tiles — mirrors the fwd conv manifest
    return (gn, gm, p.groups // merge)


# ws_bytes is i32 in the kernel ABI, so a two-stage workspace above this would
# overflow the argument.
_MAX_WGRAD_WS_BYTES = (1 << 31) - 4  # 2 GiB - 4

# hipDeviceAttributeMaxGridDimZ. Unlike x, the z extent is a 16-bit field in the
# dispatch packet on every arch rocke targets, so this is an architectural limit
# rather than a per-device one and is safe as a constant. Grouped wgrad puts
# groups*split_k on z, which is the only rocke launch that can reach it.
_MAX_GRID_DIM_Z = 65535


def _resolve_wgrad_split_k(
    spec: ConvGroupedSpec, p: "ConvProblem"
) -> Tuple[int, bool, int]:
    """Resolve wgrad split_k and two-stage eligibility; returns
    ``(split_k, two_stage, requested_split_k)``.

    Honors a concrete split-K fixed by the candidate (WMMA forces 1 -- it has
    no split-K path); otherwise auto-resolves via the CK formula on the
    per-group GEMM dims (== full dims when groups==1). Grouped rides the group
    on block_id_z alongside the K-slice (z = groups*split_k), so grouped
    split-K is valid on MFMA.

    The result is then held under the scratch cap: when it bites, split_k
    falls back to 1 (a plain store, no scratch needed). Shared by
    ``to_wgrad_spec`` and ``_wgrad_grid`` so the spec and the launch grid can
    never disagree on split_k.

    Two further constraints, both of which only bite once ``groups`` is large
    (they are no-ops at groups == 1):

    * **gridDim.z.** The group and the K-slice share the z axis, so the launch
      needs ``groups * split_k`` blocks there. The CK formula sizes split_k
      from the *per-group* GEMM and never sees the groups factor, so on a
      depthwise problem (groups == C, per-group GEMM 1 x Y*X) it happily asks
      for a degree that overflows the z limit and the launch fails with
      ``hipErrorInvalidValue``. Clamp here, in the one resolver both the spec
      and the grid go through, so they cannot desynchronise.
    * **Atomic availability.** The packed ``<2 x dtype>`` atomic the split-K
      epilogue emits for a 16-bit dW needs an even dW row length
      ``wg_N = Y*X*cpg``. When that fails there is no 16-bit atomic to fall
      back on (gfx9 has no scalar 16-bit atomic add), so the choice is
      split_k=1 or the two-stage f32-workspace path. Promote to two-stage and
      keep the parallelism -- collapsing the reduction axis of a wgrad whose
      K_wg is N*Ho*Wo is the expensive way out.
    """
    spatial = (p.Z if p.is_3d else 1) * p.Y * p.X
    wg_M = p.K // p.groups
    wg_N = spatial * (p.C // p.groups)
    # Group merging folds Gm groups into one GEMM: the tile covers Gm times the
    # per-group dims, and there are Gm times fewer of them on the grid. Both
    # the CK formula and the gridDim.z clamp below are sized off the *launched*
    # geometry, so feed them the merged numbers -- sizing a degree off the
    # unmerged grid is how the pre-merge formula came to ask for a z extent the
    # launch could not express.
    gm = max(1, int(spec.group_merge))
    grid_groups = max(1, p.groups // gm)
    if spec.split_k != -1:
        requested = spec.split_k
    else:
        from rocke.helpers.split_k import select_split_k_wgrad

        requested = select_split_k_wgrad(
            wg_M=wg_M * gm,
            wg_N=wg_N * gm,
            wg_K=p.N * p.Ho * p.Wo * (p.Do if p.is_3d else 1),
            tile_m=spec.tile_m,
            tile_n=spec.tile_n,
            tile_k=spec.tile_k,
            arch=spec.arch,
            groups=grid_groups,
            block_size=_block(spec)[0],
        ).split_k
    # The helper already keeps groups*split_k inside the z limit on the auto
    # path; this clamp still has to run for an explicitly requested split_k.
    split_k = max(1, min(requested, _MAX_GRID_DIM_Z // grid_groups))
    # When the packed 16-bit atomic cannot represent this problem, the f32
    # scratch path is the only way to keep split_k > 1. Delegate rather than
    # re-deriving the rule: a local copy is exactly how the dispatcher came to
    # hand the builder a spec the builder then rejected.
    # vector_size_c is None here by construction: to_wgrad_spec never sets it,
    # so the epilogue derives the store width from the channel dims.
    _atomic_ok, _ = _wgrad_atomic_epilogue_available(p, spec.dtype.lower(), None)
    # A merged tile cannot use the packed-atomic split-K epilogue at all: that
    # epilogue has no way to drop an off-diagonal group pair, so it would
    # accumulate garbage into a live dW element. Route any merged split_k > 1
    # through two-stage, whose scratch atomic carries the diagonal mask. This
    # mirrors the kernel-side gate in ``wgrad_group_merge_available``; keeping
    # the dispatcher's answer inside it is what stops select() from handing the
    # builder a spec the builder then rejects.
    #
    # As written today the ``gm > 1`` term is subsumed and never flips the
    # result: dispatch admits only fp16/bf16, merging is depthwise-only, and a
    # depthwise cpg == 1 forces a store-vector width of 1, so _atomic_ok is
    # already False on every shape that can merge. It is kept because it is the
    # term that stays true if any of those three premises moves -- an fp32 dW,
    # or merging extended past depthwise -- and in that world it is the only
    # thing between a merged tile and a silently wrong gradient. Deriving the
    # invariant from the merge degree directly is also simply what the rule
    # means. ``test_merged_split_k_never_takes_the_packed_atomic`` asserts the
    # subsumption explicitly, so it reports when it lifts.
    two_stage = ((not _atomic_ok) or gm > 1) and split_k > 1
    # The scratch carries no split_k factor, but it does carry the replica
    # factor -- R copies of dW per group. Use the same R this module hands the
    # spec in to_wgrad_spec, or the cap bounds an allocation nobody makes.
    #
    # Deliberately *unmerged*: this mirrors wgrad_two_stage_workspace_nbytes,
    # which sizes off WgradConvSpec.wg_M/wg_N -- the true per-group dims, kept
    # separate from grid_M/grid_N for exactly this reason. The scratch is R
    # copies of dW, and merging does not change how much dW there is; it only
    # changes how many CTAs write it. Scaling these by gm would overstate the
    # allocation by gm**2 and start refusing two-stage on problems that fit.
    ws_bytes = p.groups * _WGRAD_WS_REPLICAS * wg_M * wg_N * 4
    if two_stage and ws_bytes > _MAX_WGRAD_WS_BYTES:
        two_stage = False
        split_k = 1
    return split_k, two_stage, requested


def _wgrad_grid(spec: ConvGroupedSpec, req: OperatorRequest) -> Tuple[int, int, int]:
    assert isinstance(req, ConvGroupedRequest)
    p = _problem(req)
    spatial = (p.Z if p.is_3d else 1) * p.Y * p.X
    # Per-group GEMM dims. For the ungrouped path G==1 these reduce to wg_M=K,
    # wg_N=spatial*C.
    kpg = p.K // p.groups
    cpg = p.C // p.groups
    # Group merging folds Gm groups into one GEMM, so the tile covers Gm times
    # the per-group dims and the grid carries Gm times fewer groups. Both
    # factors have to land, and together: scaling the dims without shrinking z
    # would launch Gm times the work, and shrinking z without scaling the dims
    # would leave (Gm-1)/Gm of the merged GEMM uncomputed.
    gm = max(1, int(spec.group_merge))
    wg_M = kpg * gm  # merged output channels
    wg_N = spatial * cpg * gm  # merged filter spatial × input channel
    gx = (wg_N + spec.tile_n - 1) // spec.tile_n
    gy = (wg_M + spec.tile_m - 1) // spec.tile_m
    split_k, _two_stage, _requested = _resolve_wgrad_split_k(spec, p)
    # The merged-group index rides on block_id_z alongside the K-slice: z =
    # (groups/Gm)*split_k, decoded in-kernel as group = z // split_k, slice =
    # z % split_k. For G==1 this reduces to (gx, gy, split_k); for split_k==1
    # to (gx, gy, groups/Gm); for Gm==1 to the unmerged groups*split_k.
    return (gx, gy, (p.groups // gm) * split_k)


def _dgrad_grid(spec: ConvGroupedSpec, req: OperatorRequest) -> Tuple[int, int, int]:
    """``(flat_tiles, groups, split_k)`` for the unified tilde-decomposed kernel.

    Unlike fwd/wgrad the M-tile count is not a closed form over the problem
    dims: stride > 1 splits the convolution into ``y_tilde * x_tilde``
    independent sub-GEMMs of differing sizes, and the kernel binary-searches a
    packed record buffer to find its own. The flat tile count is therefore the
    cumulative ``block_end`` of the last sub-GEMM, which the instance already
    computes -- derive it there rather than re-deriving the decomposition here.
    The sub-GEMM geometry is channel-independent, so the count is per-group and
    the group rides ``blockIdx.y``.
    """
    assert isinstance(req, ConvGroupedRequest)
    p = _problem(req)
    sub_gemms = spec.to_dgrad_spec(p).compute_sub_gemms()
    flat_tiles = sub_gemms[-1].block_end
    return (flat_tiles, max(int(p.groups), 1), max(int(spec.split_k), 1))


def _block(spec: ConvGroupedSpec) -> Tuple[int, int, int]:
    # wave_size is baked into block_size via warp_m * warp_n * wave_size.
    # For gfx942/gfx950 wave64: 2*2*64=256; for gfx1250 wave32: 2*2*32=128.
    target = ArchTarget.from_gfx(spec.arch)
    block_size = spec.warp_m * spec.warp_n * target.wave_size
    return (block_size, 1, 1)


# ---------------------------------------------------------------------------
# Forward depthwise group merging: the largest admissible degree up to this
# cap. Unmerged, a depthwise implicit GEMM has N_gemm = kpg = 1, so a 64-wide
# tile is 1/64 dense and gridDim.z is one CTA column per group. Merging Gm
# groups fills Gm tile columns and divides the group axis of the grid by Gm.
# Past 16 the merged reduction (Y*X*Gm, with a 1/Gm-dense diagonal B tile)
# grows faster than the density gains, and 32/64 measured no better than 16
# on any depthwise shape and clearly worse on the 7x7 ones.
_GFX950_FWD_MAX_GROUP_MERGE = 16


def _fwd_merge_degree(req: "ConvGroupedRequest", spec: ImplicitGemmConvSpec) -> int:
    """``group_merge`` for a gfx950 forward request: the largest degree up to
    :data:`_GFX950_FWD_MAX_GROUP_MERGE` that the kernel's own gate admits.

    Probes :func:`fwd_group_merge_available` (through the instance validator)
    rather than re-deriving it, so the choice cannot drift from the gate: it
    rejects non-depthwise, pointwise, 3-D and non-dividing degrees itself.
    """
    p = _problem(req)
    if p.cpg != 1 or p.kpg != 1 or p.groups <= 1:
        return 1
    for gm in sorted(_GROUP_MERGE_DEGREES, reverse=True):
        if gm > _GFX950_FWD_MAX_GROUP_MERGE:
            continue
        ok, _why = _fwd_is_valid_spec(replace(spec, group_merge=gm), arch=req.arch)
        if ok:
            return gm
    return 1


# Small-channel (stem) forward convs: the channel count a C-odd, C<8 input is
# zero-padded to before the implicit GEMM runs. An odd C makes every A-operand
# tap a scalar load with its own coordinate decode; padded to 8 the tap is one
# 16-byte load and the conv runs the dense C=8 kernel. The pad pre-pass copies
# X (and the tiny W) once per call into a workspace.
_GFX950_STEM_PAD_C = 8
# For C=1 the pad pre-pass (a second launch over an 8x longer reduction) only
# pays once the conv has more than this many output pixels (N*Ho*Wo, i.e. 768
# 64-row implicit-GEMM tiles); up to it the unpadded kernel runs alone.
_GFX950_STEM_PAD_C1_MAX_UNPADDED_ROWS = 49152


def _stem_pad_c(req: "ConvGroupedRequest") -> int:
    """``pad_c`` for a gfx950 forward request: :data:`_GFX950_STEM_PAD_C` for
    an ungrouped 2-D conv with an odd input-channel count below it (C=1 only
    above :data:`_GFX950_STEM_PAD_C1_MAX_UNPADDED_ROWS` output pixels), else 0."""
    if (
        req.direction != "fwd"
        or int(req.G) != 1
        or req.Di is not None
        or int(req.C) >= _GFX950_STEM_PAD_C
        or int(req.C) % 2 == 0
    ):
        return 0
    if int(req.C) == 1:
        p = _problem(req)
        if p.N * p.Ho * p.Wo <= _GFX950_STEM_PAD_C1_MAX_UNPADDED_ROWS:
            return 0
    return _GFX950_STEM_PAD_C


# ---------------------------------------------------------------------------
# gfx950 forward candidate (wave64, MFMA 32×32×16)
# ---------------------------------------------------------------------------


def _make_gfx950_fwd_candidate() -> KernelCandidate:
    """Forward conv for gfx950: 64×64×64, 2×2, 32×32×16 MFMA (gfx942 lacks this atom)."""
    name = "implicit_gemm_conv"
    spec_id = "igemm_conv_fwd_64x64"
    algorithm = "implicit_gemm_fwd"

    def _tile(req: ConvGroupedRequest):
        return (
            _GFX950_TILE_M,
            _GFX950_TILE_N,
            _GFX950_TILE_K,
            _GFX950_WARP_M,
            _GFX950_WARP_N,
            _GFX950_WARP_TILE_MN,
            _GFX950_WARP_TILE_K,
        )

    def _build_unmerged_spec(req: ConvGroupedRequest) -> ImplicitGemmConvSpec:
        tm, tn, tk, wm, wn, wtmn, wtk = _tile(req)
        p = _problem(req)
        pad_c = _stem_pad_c(req)
        return ImplicitGemmConvSpec(
            problem=replace(p, C=pad_c) if pad_c else p,
            name=name,
            data=_data_spec(req),
            tile_m=tm,
            tile_n=tn,
            tile_k=tk,
            warp_m=wm,
            warp_n=wn,
            warp_tile_m=wtmn,
            warp_tile_n=wtmn,
            warp_tile_k=wtk,
            wave_size=ArchTarget.from_gfx(req.arch).wave_size,
            pipeline=_PIPELINE,
            epilogue=_epilogue_for(req),
            groups=int(req.G),
        )

    def _group_merge(req: ConvGroupedRequest) -> int:
        return _fwd_merge_degree(req, _build_unmerged_spec(req))

    def _build_instance_spec(req: ConvGroupedRequest) -> ImplicitGemmConvSpec:
        spec = _build_unmerged_spec(req)
        gm = _fwd_merge_degree(req, spec)
        return spec if gm == 1 else replace(spec, group_merge=gm)

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, ConvGroupedRequest)
        if not _is_gfx950(req):
            return False, f"gfx950 candidate requires arch=gfx950 (got {req.arch!r})"
        if req.direction != "fwd":
            return False, f"candidate handles 'fwd', got direction={req.direction!r}"
        ok, why = selector_matches(req, candidate)
        if not ok:
            return False, why
        ok, why = _fwd_is_valid_spec(_build_instance_spec(req), arch=req.arch)
        if not ok:
            return False, why
        return True, "ok"

    def select(req: OperatorRequest) -> ConvGroupedSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, ConvGroupedRequest)
        tm, tn, tk, wm, wn, wtmn, wtk = _tile(req)
        return ConvGroupedSpec(
            direction="fwd",
            tile_m=tm,
            tile_n=tn,
            tile_k=tk,
            warp_m=wm,
            warp_n=wn,
            warp_tile_mn=wtmn,
            warp_tile_k=wtk,
            pipeline=_PIPELINE,
            epilogue=_epilogue_for(req),
            dtype=req.dtype.lower(),
            arch=req.arch,
            group_merge=_group_merge(req),
            name=name,
            pad_c=_stem_pad_c(req),
        )

    candidate = KernelCandidate(
        name=name,
        family=_FAMILY_FWD,
        algorithm=algorithm,
        spec_id=spec_id,
        abi_version=CONV_GROUPED_ABI_VERSION,
        priority=10,
        capability=Capability(
            arches=("gfx950",),
            dtypes=("fp16", "bf16"),
            layouts=("NHWC",),
        ),
        _supports=support,
        select_spec=select,
        signature=lambda _spec: (),
        grid=_fwd_grid,
        block=_block,
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
    )
    return candidate


# ---------------------------------------------------------------------------
# gfx1250 forward candidate (wave32, WMMA 16×16×32)
# ---------------------------------------------------------------------------


def _make_gfx1250_fwd_candidate() -> KernelCandidate:
    """Forward conv for gfx1250: 32×32×32, 2×2, 16×16×32 WMMA, groups=1 only.

    gfx1250 WMMA conv is restricted to pipeline=``mem``, no cshuffle epilogue
    override (``is_valid_spec`` gates cshuffle on WMMA; epilogue is derived
    from vec_size_c as usual), and groups=1.
    """
    name = "implicit_gemm_conv_gfx1250"
    spec_id = "igemm_conv_fwd_gfx1250_32x32"
    algorithm = "implicit_gemm_fwd_gfx1250"

    def _tile(req: ConvGroupedRequest):
        return (
            _GFX1250_TILE_M,
            _GFX1250_TILE_N,
            _GFX1250_TILE_K,
            _GFX1250_WARP_M,
            _GFX1250_WARP_N,
            _GFX1250_WARP_TILE_MN,
            _GFX1250_WARP_TILE_K,
        )

    def _build_instance_spec(req: ConvGroupedRequest) -> ImplicitGemmConvSpec:
        tm, tn, tk, wm, wn, wtmn, wtk = _tile(req)
        return ImplicitGemmConvSpec(
            problem=_problem(req),
            name=name,
            data=_data_spec(req),
            tile_m=tm,
            tile_n=tn,
            tile_k=tk,
            warp_m=wm,
            warp_n=wn,
            warp_tile_m=wtmn,
            warp_tile_n=wtmn,
            warp_tile_k=wtk,
            wave_size=32,
            pipeline=_PIPELINE,
            epilogue=_epilogue_for(req),
            groups=int(req.G),
        )

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, ConvGroupedRequest)
        if not _is_gfx1250(req):
            return False, f"gfx1250 candidate requires arch=gfx1250 (got {req.arch!r})"
        if req.direction != "fwd":
            return False, f"candidate handles 'fwd', got direction={req.direction!r}"
        if int(req.G) != 1:
            return False, "WMMA conv on gfx1250 supports only groups=1"
        ok, why = selector_matches(req, candidate)
        if not ok:
            return False, why
        ok, why = _fwd_is_valid_spec(_build_instance_spec(req), arch=req.arch)
        if not ok:
            return False, why
        return True, "ok"

    def select(req: OperatorRequest) -> ConvGroupedSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, ConvGroupedRequest)
        tm, tn, tk, wm, wn, wtmn, wtk = _tile(req)
        return ConvGroupedSpec(
            direction="fwd",
            tile_m=tm,
            tile_n=tn,
            tile_k=tk,
            warp_m=wm,
            warp_n=wn,
            warp_tile_mn=wtmn,
            warp_tile_k=wtk,
            pipeline=_PIPELINE,
            epilogue=_epilogue_for(req),
            dtype=req.dtype.lower(),
            arch=req.arch,
            name=name,
        )

    candidate = KernelCandidate(
        name=name,
        family=_FAMILY_FWD,
        algorithm=algorithm,
        spec_id=spec_id,
        abi_version=CONV_GROUPED_ABI_VERSION,
        priority=10,
        capability=Capability(
            arches=("gfx1250",),
            dtypes=("fp16", "bf16"),
            layouts=("NHWC",),
        ),
        _supports=support,
        select_spec=select,
        signature=lambda _spec: (),
        grid=_fwd_grid,
        block=_block,
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
    )
    return candidate


# ---------------------------------------------------------------------------
# gfx942 forward candidate (wave64, MFMA 16×16×16)
# ---------------------------------------------------------------------------


def _make_gfx942_fwd_candidate() -> KernelCandidate:
    """Forward conv for gfx942: 64×64×64, 2×2, 16×16×16 MFMA."""
    name = "implicit_gemm_conv_gfx942"
    spec_id = "igemm_conv_fwd_gfx942_64x64"
    algorithm = "implicit_gemm_fwd_gfx942"

    def _tile(req: ConvGroupedRequest):
        return (
            _GFX942_TILE_M,
            _GFX942_TILE_N,
            _GFX942_TILE_K,
            _GFX942_WARP_M,
            _GFX942_WARP_N,
            _GFX942_WARP_TILE_MN,
            _GFX942_WARP_TILE_K,
        )

    def _build_instance_spec(req: ConvGroupedRequest) -> ImplicitGemmConvSpec:
        tm, tn, tk, wm, wn, wtmn, wtk = _tile(req)
        return ImplicitGemmConvSpec(
            problem=_problem(req),
            name=name,
            data=_data_spec(req),
            tile_m=tm,
            tile_n=tn,
            tile_k=tk,
            warp_m=wm,
            warp_n=wn,
            warp_tile_m=wtmn,
            warp_tile_n=wtmn,
            warp_tile_k=wtk,
            wave_size=ArchTarget.from_gfx(req.arch).wave_size,
            pipeline=_PIPELINE,
            epilogue=_epilogue_for(req),
            groups=int(req.G),
        )

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, ConvGroupedRequest)
        if not _is_gfx942(req):
            return False, f"gfx942 candidate requires arch=gfx942 (got {req.arch!r})"
        if req.direction != "fwd":
            return False, f"candidate handles 'fwd', got direction={req.direction!r}"
        ok, why = selector_matches(req, candidate)
        if not ok:
            return False, why
        ok, why = _fwd_is_valid_spec(_build_instance_spec(req), arch=req.arch)
        if not ok:
            return False, why
        return True, "ok"

    def select(req: OperatorRequest) -> ConvGroupedSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, ConvGroupedRequest)
        tm, tn, tk, wm, wn, wtmn, wtk = _tile(req)
        return ConvGroupedSpec(
            direction="fwd",
            tile_m=tm,
            tile_n=tn,
            tile_k=tk,
            warp_m=wm,
            warp_n=wn,
            warp_tile_mn=wtmn,
            warp_tile_k=wtk,
            pipeline=_PIPELINE,
            epilogue=_epilogue_for(req),
            dtype=req.dtype.lower(),
            arch=req.arch,
            name=name,
        )

    candidate = KernelCandidate(
        name=name,
        family=_FAMILY_FWD,
        algorithm=algorithm,
        spec_id=spec_id,
        abi_version=CONV_GROUPED_ABI_VERSION,
        priority=10,
        capability=Capability(
            arches=("gfx942",),
            dtypes=("fp16", "bf16"),
            layouts=("NHWC",),
        ),
        _supports=support,
        select_spec=select,
        signature=lambda _spec: (),
        grid=_fwd_grid,
        block=_block,
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
    )
    return candidate


# ---------------------------------------------------------------------------
# gfx942 wgrad candidate (wave64, MFMA 16×16×16)
# ---------------------------------------------------------------------------


def _make_gfx942_wgrad_candidate() -> KernelCandidate:
    """Backward-weight conv for gfx942: 64×64×64, 2×2, 16×16×16 MFMA.

    Epilogue derived from vec_size_c (cshuffle when >1, default otherwise).
    Split-K forwarded from request (1=disabled, -1=auto CK formula, >1=fixed).
    """
    name = "implicit_gemm_conv_wgrad_gfx942"
    spec_id = "igemm_conv_wgrad_gfx942_64x64"
    algorithm = "implicit_gemm_wgrad_gfx942"

    def _tile(req: ConvGroupedRequest):
        return (
            _GFX942_TILE_M,
            _GFX942_TILE_N,
            _GFX942_TILE_K,
            _GFX942_WARP_M,
            _GFX942_WARP_N,
            _GFX942_WARP_TILE_MN,
            _GFX942_WARP_TILE_K,
        )

    def _build_instance_spec(req: ConvGroupedRequest) -> WgradConvSpec:
        tm, tn, tk, wm, wn, wtmn, wtk = _tile(req)
        _ep, _sk = _wgrad_grouped_overrides(req)
        return WgradConvSpec(
            problem=_problem(req),
            name=name,
            data=_data_spec(req),
            tile_m=tm,
            tile_n=tn,
            tile_k=tk,
            warp_m=wm,
            warp_n=wn,
            warp_tile_m=wtmn,
            warp_tile_n=wtmn,
            warp_tile_k=wtk,
            wave_size=ArchTarget.from_gfx(req.arch).wave_size,
            pipeline=_PIPELINE,
            epilogue=_ep,
            split_k=_sk,
        )

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, ConvGroupedRequest)
        if not _is_gfx942(req):
            return False, f"gfx942 candidate requires arch=gfx942 (got {req.arch!r})"
        if req.direction != "wgrad":
            return False, f"candidate handles 'wgrad', got direction={req.direction!r}"
        ok, why = selector_matches(req, candidate)
        if not ok:
            return False, why
        ok, why = _wgrad_is_valid_spec(_build_instance_spec(req), arch=req.arch)
        if not ok:
            return False, why
        return True, "ok"

    def select(req: OperatorRequest) -> ConvGroupedSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, ConvGroupedRequest)
        tm, tn, tk, wm, wn, wtmn, wtk = _tile(req)
        _ep, _sk = _wgrad_grouped_overrides(req)
        return ConvGroupedSpec(
            direction="wgrad",
            tile_m=tm,
            tile_n=tn,
            tile_k=tk,
            warp_m=wm,
            warp_n=wn,
            warp_tile_mn=wtmn,
            warp_tile_k=wtk,
            pipeline=_PIPELINE,
            epilogue=_ep,
            dtype=req.dtype.lower(),
            arch=req.arch,
            split_k=_sk,
            name=name,
        )

    candidate = KernelCandidate(
        name=name,
        family=_FAMILY_WGRAD,
        algorithm=algorithm,
        spec_id=spec_id,
        abi_version=CONV_GROUPED_ABI_VERSION,
        priority=10,
        capability=Capability(
            arches=("gfx942",),
            dtypes=("fp16", "bf16"),
            layouts=("NHWC",),
        ),
        _supports=support,
        select_spec=select,
        signature=lambda _spec: (),
        grid=_wgrad_grid,
        block=_block,
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
    )
    return candidate


# ---------------------------------------------------------------------------
# gfx950 wgrad candidate (wave64, MFMA 32×32×16)
# ---------------------------------------------------------------------------


def _wgrad_lds_k_outer(req: "ConvGroupedRequest", warp_tile_mn: int) -> bool:
    """Whether a wgrad candidate should use the K-outer LDS layout.

    Used by both the gfx950 (wave64 MFMA) and gfx1250 (wave32 WMMA) candidates;
    the wave size is resolved from ``req.arch`` below rather than assumed, so
    the one gate covers both regimes.

    wgrad's stride-1 global axis is the GEMM *free* axis, so an M-outer LDS tile
    forces a transpose on store: one ``ds_write_b16`` per element, all of them
    bank-conflicting because the row stride is a multiple of the LDS bank period.
    The K-outer layout stores the tile in global order (one wide store) and lets
    ``ds_read_b64_tr_b16`` transpose on the read side instead.

    K-outer lowers the LDS instruction count and removes the row-stride bank
    conflicts the M-outer transpose-on-store incurs. See the ``lds_k_outer``
    field docs on ``WgradConvSpec``.

    Gated to exactly what the transpose-read lane mapping is validated for:
    16-bit A/B operands, and either wave64 MFMA on gfx950 (``ds_read_b64_tr_b16``
    does not exist on gfx942) with a 16- or 32-wide atom edge, or wave32 WMMA on
    gfx1250 (``ds_load_tr16_b128``) with the 16-wide edge.

    Delegates so dispatch and the sweep driver cannot drift: this used to be a
    second copy of the gate that compared the module constant against a tuple
    (constant-true regardless of the request) and never checked wave_size.
    """
    return WgradConvSpec.default_lds_k_outer(
        arch=req.arch,
        dtype_a=req.dtype.lower(),
        dtype_b=req.dtype.lower(),
        warp_tile_m=warp_tile_mn,
        warp_tile_n=warp_tile_mn,
        wave_size=ArchTarget.from_gfx(req.arch).wave_size,
    )


def _wgrad_b_is_scalar(req: "ConvGroupedRequest") -> bool:
    """Whether the X (B-operand) free-axis load degenerates to one element.

    ``vec_b`` is the widest power-of-two width that divides ``cpg`` (X is NHWC,
    so the free-axis run is the channel dim), which is 1 exactly when ``cpg``
    is odd. Asks :meth:`WgradConvSpec.default_vector_sizes` rather than testing
    parity directly so the two cannot drift apart if the width ladder changes.

    Depthwise (``cpg == kpg == 1`` with real groups) is excluded even though it
    is the extreme of the scalar-B case. There the CTA count comes from the
    group axis, not from the tile, so the "fewer B elements per thread" argument
    the variant rests on does not drive it -- and measurement agreed: the
    narrow tile was a win on some depthwise shapes and a loss on others, with
    no predictor separating them.

    The predictor turned out not to be the tile at all but the merge degree:
    see :func:`_wgrad_merge_degree`, which fixes the depthwise scalar run by
    packing several groups into one tile at the shipped width, rather than by
    narrowing the tile to cut per-thread B work. So depthwise keeps the shipped
    geometry and changes only how many groups share it.
    """
    p = _problem(req)
    cpg = p.C // max(int(p.groups), 1)
    if cpg == 1 and p.kpg == 1 and p.groups > 1:
        return False
    _va, vec_b, _vc = WgradConvSpec.default_vector_sizes(cpg, p.kpg, req.dtype.lower())
    return vec_b == 1


# Depthwise wgrad tile ladder: the merge degree a wider tile buys must reach
# _WGRAD_DW_TARGET_MERGE (or at least _WGRAD_DW_MIN_WIDE_MERGE on the widest
# rung) before the tile is widened at all; otherwise the shipped 64-wide tile
# and its own degree stand.
_WGRAD_DW_TILE_N_LADDER = (64, 128, 256)
_WGRAD_DW_TARGET_MERGE = 8
_WGRAD_DW_MIN_WIDE_MERGE = 4
# Below this many reduction rows (N*Ho*Wo) a depthwise wgrad that already
# merges groups at the shipped tile keeps it: the per-CTA reduction is too
# short to repay the wider tile, and halving the CTA count on an already
# small grid costs more than the wider loads save (batch-1 late-stage maps,
# e.g. 14x14). A filter that cannot merge at all at the shipped tile (7x7)
# still widens: there the wide tile is what turns scalar loads into vectors.
_WGRAD_DW_SMALL_REDUCTION = 1024


def _wgrad_merge_degree_at(req: "ConvGroupedRequest", tile_n: int) -> int:
    """The largest admissible ``group_merge`` at a ``tile_n``-wide tile.

    Mirrors the three request-dependent clauses of
    :func:`kernels.common.conv_implicit_gemm_wgrad.wgrad_group_merge_available`
    -- the degree ladder, group divisibility, and ``grid_N <= tile_n``. The
    remaining clauses of the gate (depthwise, wave64, split-K/two-stage) are
    properties of the candidate rather than of the request and are checked by
    the caller. ``grid_M <= tile_m`` is not re-derived: depthwise is
    ``kpg == 1`` so ``grid_M == Gm``, and the ladder already stops at
    64 == ``tile_m``.

    This re-derives rather than probing the gate, because the gate takes a
    built :class:`WgradConvSpec` and building one per degree to answer a
    closed-form question is a lot of work to learn ``spatial * Gm <= tile_n``.
    The cost is that the two can drift, which
    ``TestWgradMergeDegree.test_every_choice_is_admissible`` is there to catch:
    it runs the choice back through the real gate across a grid of shapes, so a
    clause that moves in the kernel and not here fails there.
    """
    p = _problem(req)
    groups = max(int(p.groups), 1)
    # Merging is depthwise-only in the kernel, and pointless with one group.
    if p.cpg != 1 or p.kpg != 1 or groups <= 1:
        return 1
    spatial = p.Y * p.X * ((p.Z or 1) if p.is_3d else 1)
    best = 1
    for gm in _GROUP_MERGE_DEGREES:
        if gm <= groups and groups % gm == 0 and spatial * gm <= tile_n:
            best = gm
    return best


def _wgrad_dw_tile_n(req: "ConvGroupedRequest") -> int:
    """``tile_n`` for a gfx950 wgrad request.

    Depthwise wgrad is the degenerate case of the wgrad GEMM. ``wg_M = kpg = 1``
    and ``wg_N = Y*X*cpg = Y*X``, so the tile covers a handful of filter taps
    and the GEMM free axis is one element per group -- every dY/X load is
    scalar. Merging ``Gm`` groups into one tile makes that run ``Gm`` elements
    long (``Gm`` *is* the load vector width), and ``spatial * Gm <= tile_n``
    bounds it. At the shipped 64-wide tile a 3x3 filter stops at ``Gm = 4``
    and a 7x7 one cannot merge at all (``49 * 2 > 64``).

    So the tile is widened -- only for depthwise, and only as far as needed:
    the narrowest rung of :data:`_WGRAD_DW_TILE_N_LADDER` that fits
    :data:`_WGRAD_DW_TARGET_MERGE` groups, else the widest rung if that still
    fits :data:`_WGRAD_DW_MIN_WIDE_MERGE` groups. A wide tile at a low degree
    is mostly padding, which is how an unconditional ladder regresses; a shape
    whose divisibility or filter leaves the wide rung below that floor keeps
    the shipped tile and its degree. Everything that is not depthwise keeps the
    shipped tile.
    """
    p = _problem(req)
    if p.cpg != 1 or p.kpg != 1 or p.groups <= 1:
        return _GFX950_TILE_N
    if (
        p.N * p.Ho * p.Wo < _WGRAD_DW_SMALL_REDUCTION
        and _wgrad_merge_degree_at(req, _GFX950_TILE_N) > 1
    ):
        return _GFX950_TILE_N
    for tile_n in _WGRAD_DW_TILE_N_LADDER:
        if _wgrad_merge_degree_at(req, tile_n) >= _WGRAD_DW_TARGET_MERGE:
            return tile_n
    widest = _WGRAD_DW_TILE_N_LADDER[-1]
    if _wgrad_merge_degree_at(req, widest) >= _WGRAD_DW_MIN_WIDE_MERGE:
        return widest
    return _GFX950_TILE_N


def _wgrad_merge_degree(req: "ConvGroupedRequest") -> int:
    """``group_merge`` for a gfx950 wgrad request: as much as the gate allows
    at the tile :func:`_wgrad_dw_tile_n` picks."""
    return _wgrad_merge_degree_at(req, _wgrad_dw_tile_n(req))


def _make_gfx950_wgrad_candidate() -> KernelCandidate:
    """Backward-weight conv for gfx950: 64×64×64, 2×2, 32×32×16 MFMA.

    Epilogue derived from vec_size_c (cshuffle when >1, default otherwise).
    Split-K forwarded from request (1=disabled, -1=auto CK formula, >1=fixed).
    gfx1250 wgrad is handled by ``_make_gfx1250_wgrad_candidate`` (WMMA 16x16x32).

    Two geometries, selected per request by :func:`_wgrad_b_is_scalar`: the
    64×64 default, and the 64×32 / 8-wave variant for the odd-``cpg`` shapes
    whose B load cannot vectorise (see the ``_GFX950_WGRAD_SCALARB_*``
    constants). One candidate rather than two because the choice is a pure
    function of the request with no overlap -- a second candidate would need
    the same predicate in its ``support`` anyway, and priorities to break a tie
    that cannot occur.
    """
    name = "implicit_gemm_conv_wgrad"
    spec_id = "igemm_conv_wgrad_64x64"
    algorithm = "implicit_gemm_wgrad"

    def _tile(req: ConvGroupedRequest):
        if _wgrad_b_is_scalar(req):
            return (
                _GFX950_WGRAD_SCALARB_TILE_M,
                _GFX950_WGRAD_SCALARB_TILE_N,
                _GFX950_WGRAD_SCALARB_TILE_K,
                _GFX950_WGRAD_SCALARB_WARP_M,
                _GFX950_WGRAD_SCALARB_WARP_N,
                _GFX950_WGRAD_SCALARB_WARP_TILE_MN,
                _GFX950_WGRAD_SCALARB_WARP_TILE_K,
            )
        return (
            _GFX950_TILE_M,
            _wgrad_dw_tile_n(req),
            _GFX950_TILE_K,
            _GFX950_WARP_M,
            _GFX950_WARP_N,
            _GFX950_WARP_TILE_MN,
            _GFX950_WARP_TILE_K,
        )

    def _pipeline(req: ConvGroupedRequest) -> str:
        return _GFX950_WGRAD_SCALARB_PIPELINE if _wgrad_b_is_scalar(req) else _PIPELINE

    def _group_merge(req: ConvGroupedRequest) -> int:
        # The scalar-B variant pins its own narrower tile, which the degree
        # rule's tile bound was not derived against; rather than re-derive it
        # for a second geometry, that path keeps the unmerged behaviour it was
        # tuned with. The two do not overlap in practice -- _wgrad_b_is_scalar
        # excludes depthwise, and merging is depthwise-only -- so this is
        # belt-and-braces, not a policy choice.
        if _wgrad_b_is_scalar(req):
            return 1
        return _wgrad_merge_degree(req)

    def _build_instance_spec(req: ConvGroupedRequest) -> WgradConvSpec:
        tm, tn, tk, wm, wn, wtmn, wtk = _tile(req)
        _ep, _sk = _wgrad_grouped_overrides(req)
        return WgradConvSpec(
            group_merge=_group_merge(req),
            problem=_problem(req),
            name=name,
            lds_k_outer=_wgrad_lds_k_outer(req, wtmn),
            data=_data_spec(req),
            tile_m=tm,
            tile_n=tn,
            tile_k=tk,
            warp_m=wm,
            warp_n=wn,
            warp_tile_m=wtmn,
            warp_tile_n=wtmn,
            warp_tile_k=wtk,
            wave_size=ArchTarget.from_gfx(req.arch).wave_size,
            pipeline=_pipeline(req),
            epilogue=_ep,
            split_k=_sk,
        )

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, ConvGroupedRequest)
        if not _is_gfx950(req):
            return False, f"gfx950 candidate requires arch=gfx950 (got {req.arch!r})"
        if req.direction != "wgrad":
            return False, f"candidate handles 'wgrad', got direction={req.direction!r}"
        ok, why = selector_matches(req, candidate)
        if not ok:
            return False, why
        ok, why = _wgrad_is_valid_spec(_build_instance_spec(req), arch=req.arch)
        if not ok:
            return False, why
        return True, "ok"

    def select(req: OperatorRequest) -> ConvGroupedSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, ConvGroupedRequest)
        tm, tn, tk, wm, wn, wtmn, wtk = _tile(req)
        _ep, _sk = _wgrad_grouped_overrides(req)
        return ConvGroupedSpec(
            direction="wgrad",
            tile_m=tm,
            tile_n=tn,
            tile_k=tk,
            warp_m=wm,
            warp_n=wn,
            warp_tile_mn=wtmn,
            warp_tile_k=wtk,
            pipeline=_pipeline(req),
            epilogue=_ep,
            lds_k_outer=_wgrad_lds_k_outer(req, wtmn),
            dtype=req.dtype.lower(),
            arch=req.arch,
            split_k=_sk,
            group_merge=_group_merge(req),
            name=name,
        )

    candidate = KernelCandidate(
        name=name,
        family=_FAMILY_WGRAD,
        algorithm=algorithm,
        spec_id=spec_id,
        abi_version=CONV_GROUPED_ABI_VERSION,
        priority=10,
        capability=Capability(
            arches=("gfx950",),
            dtypes=("fp16", "bf16"),
            layouts=("NHWC",),
        ),
        _supports=support,
        select_spec=select,
        signature=lambda _spec: (),
        grid=_wgrad_grid,
        block=_block,
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
    )
    return candidate


def _dgrad_lds_k_outer(req: "ConvGroupedRequest", warp_tile_mn: int) -> bool:
    """Whether the gfx950 dgrad candidate should use the K-outer LDS layout.

    Delegates to the instance predicate for the same reason the wgrad helper
    does -- a second copy of the gate is how the two drift apart.

    Note the asymmetry with :func:`_wgrad_lds_k_outer`: dgrad flips only the B
    tile, so the predicate takes only the B-side dtype and warp tile, and it
    additionally keys on ``cpg``. The saving is proportional to the B load
    width, which collapses to 1 on an odd channel run -- there ``axis_b`` is
    already ``"col"``, there is no transpose-on-store to remove, and K-outer
    would only add the read-side cost.
    """
    p = _problem(req)
    return DgradConvSpec.default_lds_k_outer(
        arch=req.arch,
        dtype_b=req.dtype.lower(),
        warp_tile_n=warp_tile_mn,
        cpg=p.C // max(int(p.groups), 1),
        wave_size=ArchTarget.from_gfx(req.arch).wave_size,
        pipeline=_PIPELINE,
    )


def _make_gfx950_dgrad_candidate() -> KernelCandidate:
    """Backward-data conv for gfx950: 64x64x64, 2x2, 32x32x16 MFMA.

    Pins ``epilogue="default"`` and ``split_k=1``. dgrad already dispatches its
    epilogue internally on ``needs_atomic`` (stride > 1 gives more than one
    sub-GEMM, which forces the atomic store regardless of split-K), so the
    epilogue knob does not carry the same meaning it does for wgrad. Split-K is
    left at 1 rather than auto-resolved: the CK formula wgrad uses keys on its
    lopsided ``N*Ho*Wo`` reduction, and dgrad's ``Y*X*K`` is not that shape.

    No gfx942 or gfx1250 counterpart yet. gfx942 lacks the 32x32x16 atom, and
    the gfx1250 wave32 path is exercised through the sweep driver but has no
    dispatch-level dual-engine test of its own.
    """
    name = "implicit_gemm_conv_dgrad"
    spec_id = "igemm_conv_dgrad_64x64"
    algorithm = "implicit_gemm_dgrad"

    def _tile(req: ConvGroupedRequest):
        return (
            _GFX950_TILE_M,
            _GFX950_TILE_N,
            _GFX950_TILE_K,
            _GFX950_WARP_M,
            _GFX950_WARP_N,
            _GFX950_WARP_TILE_MN,
            _GFX950_WARP_TILE_K,
        )

    def _build_instance_spec(req: ConvGroupedRequest) -> DgradConvSpec:
        tm, tn, tk, wm, wn, wtmn, wtk = _tile(req)
        return DgradConvSpec(
            problem=_problem(req),
            name=name,
            lds_k_outer=_dgrad_lds_k_outer(req, wtmn),
            data=_data_spec(req),
            tile_m=tm,
            tile_n=tn,
            tile_k=tk,
            warp_m=wm,
            warp_n=wn,
            warp_tile_m=wtmn,
            warp_tile_n=wtmn,
            warp_tile_k=wtk,
            wave_size=ArchTarget.from_gfx(req.arch).wave_size,
            pipeline=_PIPELINE,
            epilogue=_epilogue_for(req),
            split_k=1,
        )

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, ConvGroupedRequest)
        if not _is_gfx950(req):
            return False, f"gfx950 candidate requires arch=gfx950 (got {req.arch!r})"
        if req.direction != "dgrad":
            return False, f"candidate handles 'dgrad', got direction={req.direction!r}"
        ok, why = selector_matches(req, candidate)
        if not ok:
            return False, why
        ok, why = _dgrad_is_valid_spec(_build_instance_spec(req), arch=req.arch)
        if not ok:
            return False, why
        return True, "ok"

    def select(req: OperatorRequest) -> ConvGroupedSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, ConvGroupedRequest)
        tm, tn, tk, wm, wn, wtmn, wtk = _tile(req)
        return ConvGroupedSpec(
            direction="dgrad",
            tile_m=tm,
            tile_n=tn,
            tile_k=tk,
            warp_m=wm,
            warp_n=wn,
            warp_tile_mn=wtmn,
            warp_tile_k=wtk,
            pipeline=_PIPELINE,
            epilogue=_epilogue_for(req),
            lds_k_outer=_dgrad_lds_k_outer(req, wtmn),
            dtype=req.dtype.lower(),
            arch=req.arch,
            split_k=1,
            name=name,
        )

    candidate = KernelCandidate(
        name=name,
        family=_FAMILY_DGRAD,
        algorithm=algorithm,
        spec_id=spec_id,
        abi_version=CONV_GROUPED_ABI_VERSION,
        priority=10,
        capability=Capability(
            arches=("gfx950",),
            dtypes=("fp16", "bf16"),
            layouts=("NHWC",),
        ),
        _supports=support,
        select_spec=select,
        signature=lambda _spec: (),
        grid=_dgrad_grid,
        block=_block,
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
    )
    return candidate


def _make_gfx1250_wgrad_candidate() -> KernelCandidate:
    """Backward-weight conv for gfx1250: 32x32x32, 2x2, 16x16x32 WMMA (wave32).

    gfx1250's only fp16/bf16 atom is 16x16x32 (there is no 16x16x16). WMMA wgrad
    requires split_k=1 and the direct-store ('default') epilogue, so both are
    forced here regardless of the request. Grouped convolution is supported and
    runs grid-per-group (the kernel is validated dual-engine by
    test_gfx1250_grouped_wgrad_dual_engine). Group merging (``group_merge > 1``)
    is MFMA-only and additionally needs split_k > 1 on the two-stage path, so it
    is unreachable here on both counts: is_valid_wgrad_spec rejects it for
    wave32 and for the split_k=1 this candidate forces.
    """
    name = "implicit_gemm_conv_wgrad_gfx1250"
    spec_id = "igemm_conv_wgrad_gfx1250_32x32"
    algorithm = "implicit_gemm_wgrad_gfx1250"

    def _tile(req: ConvGroupedRequest):
        return (
            _GFX1250_TILE_M,
            _GFX1250_TILE_N,
            _GFX1250_TILE_K,
            _GFX1250_WARP_M,
            _GFX1250_WARP_N,
            _GFX1250_WARP_TILE_MN,
            _GFX1250_WARP_TILE_K,
        )

    def _build_instance_spec(req: ConvGroupedRequest) -> WgradConvSpec:
        tm, tn, tk, wm, wn, wtmn, wtk = _tile(req)
        return WgradConvSpec(
            problem=_problem(req),
            name=name,
            data=_data_spec(req),
            tile_m=tm,
            tile_n=tn,
            tile_k=tk,
            warp_m=wm,
            warp_n=wn,
            warp_tile_m=wtmn,
            warp_tile_n=wtmn,
            warp_tile_k=wtk,
            wave_size=32,
            pipeline=_PIPELINE,
            epilogue="default",
            split_k=1,
            lds_k_outer=_wgrad_lds_k_outer(req, wtmn),
        )

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, ConvGroupedRequest)
        if not _is_gfx1250(req):
            return False, f"gfx1250 candidate requires arch=gfx1250 (got {req.arch!r})"
        if req.direction != "wgrad":
            return False, f"candidate handles 'wgrad', got direction={req.direction!r}"
        ok, why = selector_matches(req, candidate)
        if not ok:
            return False, why
        ok, why = _wgrad_is_valid_spec(_build_instance_spec(req), arch=req.arch)
        if not ok:
            return False, why
        return True, "ok"

    def select(req: OperatorRequest) -> ConvGroupedSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, ConvGroupedRequest)
        tm, tn, tk, wm, wn, wtmn, wtk = _tile(req)
        return ConvGroupedSpec(
            direction="wgrad",
            tile_m=tm,
            tile_n=tn,
            tile_k=tk,
            warp_m=wm,
            warp_n=wn,
            warp_tile_mn=wtmn,
            warp_tile_k=wtk,
            pipeline=_PIPELINE,
            epilogue="default",
            dtype=req.dtype.lower(),
            arch=req.arch,
            split_k=1,
            lds_k_outer=_wgrad_lds_k_outer(req, wtmn),
            name=name,
        )

    candidate = KernelCandidate(
        name=name,
        family=_FAMILY_WGRAD,
        algorithm=algorithm,
        spec_id=spec_id,
        abi_version=CONV_GROUPED_ABI_VERSION,
        priority=10,
        capability=Capability(
            arches=("gfx1250",),
            dtypes=("fp16", "bf16"),
            layouts=("NHWC",),
        ),
        _supports=support,
        select_spec=select,
        signature=lambda _spec: (),
        grid=_wgrad_grid,
        block=_block,
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
    )
    return candidate


# ---------------------------------------------------------------------------
# gfx950 direct (non-GEMM) candidates
# ---------------------------------------------------------------------------
#
# Depthwise and small-cpg grouped convolutions are a poor fit for the
# implicit GEMM: their per-group GEMM is a handful of columns wide, so a 64x64
# MFMA tile is mostly padding and the grid is one thin CTA column per group.
# The direct kernels in kernels/common/conv_direct_grouped.py stream NHWC rows
# with no GEMM padding at all. They take precedence (lower priority value) over
# the implicit-GEMM candidate wherever they admit the request; everything else
# still reaches the implicit GEMM.

_DIRECT_PRIORITY = 5
_DIRECT_KINDS = ("dw_tiled", "dw_dgrad_stream", "grouped", "grouped_dgrad")


def _direct_block(spec: ConvGroupedSpec) -> Tuple[int, int, int]:
    """Threads per CTA of a direct spec: the last ``direct_tile`` knob is the
    wave count (depthwise) or the groups-per-block wave count (grouped)."""
    grouped = spec.direct_kind in ("grouped", "grouped_dgrad")
    return (spec.direct_tile[1 if grouped else -1] * 64, 1, 1)


def _direct_grid(spec: ConvGroupedSpec, req: OperatorRequest) -> Tuple[int, int, int]:
    assert isinstance(req, ConvGroupedRequest)
    return direct_launch_geometry(spec.to_direct_spec(req))[0]


# Depthwise tiled forward: CTAs (one wave each) the launch should reach before
# a taller row tile is worth its longer per-CTA filter-row chain.
_DW_TILED_MIN_CTAS = 1024


def _dw_tiled_tile(p: DirectConvProblem) -> Tuple[int, int, int]:
    """(block_w, block_h, block_waves) for the output-stationary depthwise
    forward kernel. One wave per CTA and a 7-wide output strip when the width
    tiles by 7 (the 56/28/14/7 maps of the common backbones), else 8.

    Rows per CTA: a tall tile amortises the input halo over more output rows,
    but each CTA walks its rows one after another, so on a small grid the
    launch is as slow as one CTA's row chain. The row tile is the tallest of
    7 (larger filters and large stride-1 maps) or 4, then 2, then 1, that
    still launches :data:`_DW_TILED_MIN_CTAS` CTAs; 1 when none does.
    """
    block_w = 7 if p.Wo % 7 == 0 else 8
    tallest = 7 if (p.KH >= 5 or (p.stride == 1 and p.Ho >= 56)) else 4
    strips = -(-p.Wo // block_w) * -(-p.groups // 64) * p.N
    for block_h in sorted({tallest, 4, 2}, reverse=True):
        if strips * -(-p.Ho // block_h) >= _DW_TILED_MIN_CTAS:
            return (block_w, block_h, 1)
    return (block_w, 1, 1)


def _grouped_direct_tile(p: DirectConvProblem) -> Tuple[int, int, int]:
    """(block_q, block_groups, double_buffer) for the generic grouped kernel."""
    return (16, 4 if p.cpg <= 8 else 2, 1)


# The grouped direct routes (and grouped dgrad's extra weight-transpose launch)
# carry a per-call floor the implicit GEMM does not, so they only pay once the
# implicit GEMM would launch this many 64-row output tiles across its groups.
_GROUPED_DIRECT_MIN_IGEMM_TILES = 2048


def _grouped_igemm_tiles(p: DirectConvProblem) -> int:
    """Output tiles the implicit GEMM launches for ``p``: one per 64 output
    pixels per group (its N tile covers cpg <= 16 in one)."""
    return -(-(p.N * p.Ho * p.Wo) // 64) * p.groups


# Depthwise dgrad CTA-count thresholds (CTAs at the default 4-wide strip).
# Below one CTA per CU the launch is latency-bound and a 1-wide strip, with
# four times the CTAs, wins; up to _DW_DGRAD_WIDE_CTAS a 2-wide strip helps
# when the map is wide next to the filter, so the extra halo stays cheap.
_DW_DGRAD_TINY_CTAS = 256
_DW_DGRAD_WIDE_CTAS = 1024


def _dw_dgrad_tile(p: DirectConvProblem) -> Tuple[int, int]:
    """(block_w, block_waves) for the streaming depthwise dgrad kernel.

    The kernel streams every dY row inside one CTA, so its parallelism is
    only ``ceil(W / block_w) * ceil(C / 64) * N`` CTAs of one wave each.
    Counters on the 4-wide strip show low occupancy and an idle memory unit,
    i.e. too few waves in flight rather than too little bandwidth; the strip
    width is the knob that adds CTAs.
    """
    ctas4 = -(-p.W // 4) * -(-p.groups // 64) * p.N
    if ctas4 < _DW_DGRAD_TINY_CTAS:
        return (1, 1)
    if ctas4 < _DW_DGRAD_WIDE_CTAS and p.W >= 8 * p.KW:
        return (2, 1)
    return (4, 1)


def _grouped_dgrad_tile(p: DirectConvProblem) -> Tuple[int, int, int]:
    """(block_q, block_groups, block_h) for grouped dgrad as the forward direct
    kernel over (dY, W^T): 8-row H tiles, a 32-wide Q strip and 4 groups per
    CTA up to cpg 8, a 16-wide strip and 2 groups at cpg 16."""
    return (32, 4, 8) if p.cpg <= 8 else (16, 2, 8)


def _make_gfx950_direct_candidate(kind: str) -> KernelCandidate:
    """A direct-conv candidate for gfx950: ``kind`` is one of _DIRECT_KINDS.

    * ``dw_tiled`` -- depthwise forward, :class:`DirectDepthwiseTiledSpec`.
    * ``dw_dgrad_stream`` -- depthwise backward-data,
      :class:`DirectDepthwiseDgradStreamSpec` (the implicit-GEMM dgrad
      candidate rejects depthwise outright).
    * ``grouped`` -- grouped forward with ``cpg == kpg`` in 4..16,
      :class:`DirectConvSpec`. Wider groups stay on the implicit GEMM, whose
      tile is dense enough there.
    * ``grouped_dgrad`` -- grouped backward-data for the same ``cpg`` range at
      stride 1: dX is the forward conv of dY with the transposed filter, so it
      runs :class:`DirectConvSpec` on the transposed problem
      (:func:`make_dgrad_fprop_spec`) after a weight-transpose pre-pass (see
      :func:`conv_grouped_prepass`). Stride > 1 stays on the implicit GEMM.
    """
    direction = "dgrad" if kind in ("dw_dgrad_stream", "grouped_dgrad") else "fwd"
    name = f"direct_conv_{kind}"
    spec_id = f"direct_{kind}"
    algorithm = f"direct_{kind}"

    def _tile(p: DirectConvProblem) -> Tuple[int, ...]:
        if kind == "dw_tiled":
            return _dw_tiled_tile(p)
        if kind == "dw_dgrad_stream":
            return _dw_dgrad_tile(p)
        if kind == "grouped_dgrad":
            return _grouped_dgrad_tile(p)
        return _grouped_direct_tile(p)

    def _spec(req: ConvGroupedRequest, p: DirectConvProblem) -> ConvGroupedSpec:
        return ConvGroupedSpec(
            direction=direction,
            tile_m=0,
            tile_n=0,
            tile_k=0,
            warp_m=0,
            warp_n=0,
            warp_tile_mn=0,
            warp_tile_k=0,
            pipeline="direct",
            epilogue="direct",
            dtype=req.dtype.lower(),
            arch=req.arch,
            name=name,
            direct_kind=kind,
            direct_tile=_tile(p),
        )

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, ConvGroupedRequest)
        if not _is_gfx950(req):
            return False, f"gfx950 candidate requires arch=gfx950 (got {req.arch!r})"
        if req.direction != direction:
            return False, f"candidate handles {direction!r}, got {req.direction!r}"
        ok, why = selector_matches(req, candidate)
        if not ok:
            return False, why
        p = _direct_problem(req)
        if p is None:
            return False, "direct kernels need 2-D, square stride/pad, no dilation"
        if p.groups <= 1:
            return False, "direct grouped kernels need groups > 1"
        if kind in ("dw_tiled", "dw_dgrad_stream"):
            if p.cpg != 1 or p.kpg != 1:
                return False, f"{kind} is depthwise only (cpg=kpg=1)"
        elif p.cpg != p.kpg or p.cpg % 4 or not 4 <= p.cpg <= 16:
            return False, f"{kind} needs cpg == kpg in 4..16 (multiple of 4)"
        if kind == "grouped_dgrad" and (p.stride != 1 or p.PAD > p.KH - 1):
            return False, "grouped_dgrad needs stride 1 and pad <= KH - 1"
        if kind in ("grouped", "grouped_dgrad"):
            tiles = _grouped_igemm_tiles(p)
            if tiles < _GROUPED_DIRECT_MIN_IGEMM_TILES:
                return False, (
                    f"{kind} needs at least {_GROUPED_DIRECT_MIN_IGEMM_TILES} "
                    f"implicit-GEMM output tiles to pay off (got {tiles})"
                )
        inst = _spec(req, p).to_direct_spec(req)
        if kind == "dw_tiled":
            return is_valid_depthwise_tiled_spec(inst, arch=req.arch)
        if kind == "dw_dgrad_stream":
            return is_valid_depthwise_dgrad_stream_spec(inst, arch=req.arch)
        return _direct_is_valid_spec(inst, arch=req.arch)

    def select(req: OperatorRequest) -> ConvGroupedSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, ConvGroupedRequest)
        return _spec(req, _direct_problem(req))

    candidate = KernelCandidate(
        name=name,
        family=_FAMILY_DGRAD if direction == "dgrad" else _FAMILY_FWD,
        algorithm=algorithm,
        spec_id=spec_id,
        abi_version=CONV_GROUPED_ABI_VERSION,
        priority=_DIRECT_PRIORITY,
        capability=Capability(
            arches=("gfx950",),
            dtypes=("fp16", "bf16"),
            layouts=("NHWC",),
        ),
        _supports=support,
        select_spec=select,
        signature=lambda _spec: (),
        grid=_direct_grid,
        block=_direct_block,
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
    )
    return candidate


# ---------------------------------------------------------------------------
# Registries — one per direction (different family strings)
# ---------------------------------------------------------------------------

_CONV_DIM_VOCABULARY = (
    "N",
    "C",
    "K",
    "Hi",
    "Wi",
    "Y",
    "X",
    "G",
    "stride_h",
    "stride_w",
    "pad_h",
    "pad_w",
    "dilation_h",
    "dilation_w",
    "Ho",
    "Wo",
    # 3D depth dims (None when 2D)
    "Di",
    "Z",
    "stride_d",
    "pad_d",
    "dilation_d",
    "Do",
)

CONV_DGRAD_REGISTRY = CandidateRegistry(
    _FAMILY_DGRAD, dim_vocabulary=_CONV_DIM_VOCABULARY
)
CONV_DGRAD_REGISTRY.register(_make_gfx950_dgrad_candidate())
CONV_DGRAD_REGISTRY.register(_make_gfx950_direct_candidate("dw_dgrad_stream"))
CONV_DGRAD_REGISTRY.register(_make_gfx950_direct_candidate("grouped_dgrad"))

CONV_FWD_REGISTRY = CandidateRegistry(_FAMILY_FWD, dim_vocabulary=_CONV_DIM_VOCABULARY)
CONV_FWD_REGISTRY.register(_make_gfx942_fwd_candidate())
CONV_FWD_REGISTRY.register(_make_gfx950_fwd_candidate())
CONV_FWD_REGISTRY.register(_make_gfx950_direct_candidate("dw_tiled"))
CONV_FWD_REGISTRY.register(_make_gfx950_direct_candidate("grouped"))
CONV_FWD_REGISTRY.register(_make_gfx1250_fwd_candidate())

CONV_WGRAD_REGISTRY = CandidateRegistry(
    _FAMILY_WGRAD, dim_vocabulary=_CONV_DIM_VOCABULARY
)
CONV_WGRAD_REGISTRY.register(_make_gfx942_wgrad_candidate())
CONV_WGRAD_REGISTRY.register(_make_gfx950_wgrad_candidate())
CONV_WGRAD_REGISTRY.register(_make_gfx1250_wgrad_candidate())


def _registry_for(req: ConvGroupedRequest) -> CandidateRegistry:
    if req.direction == "wgrad":
        return CONV_WGRAD_REGISTRY
    if req.direction == "dgrad":
        return CONV_DGRAD_REGISTRY
    return CONV_FWD_REGISTRY


# ---------------------------------------------------------------------------
# KernelId
# ---------------------------------------------------------------------------


def _kernel_id(
    req: ConvGroupedRequest, candidate: KernelCandidate, spec: ConvGroupedSpec
) -> KernelId:
    request_hash = stable_json_hash(req.normalized(), n=16)
    spec_hash = stable_json_hash(asdict(spec), n=16)
    return KernelId(
        op=f"conv_{req.direction}",
        family=candidate.family,
        candidate=candidate.name,
        algorithm=candidate.algorithm,
        spec_id=candidate.spec_id,
        arch=req.arch,
        abi_version=candidate.abi_version,
        request_hash=request_hash,
        spec_hash=spec_hash,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def conv_grouped_candidates(direction: str = "fwd") -> Tuple[KernelCandidate, ...]:
    if direction == "wgrad":
        return CONV_WGRAD_REGISTRY.candidates()
    if direction == "dgrad":
        return CONV_DGRAD_REGISTRY.candidates()
    return CONV_FWD_REGISTRY.candidates()


def registered_conv_grouped_combos(
    req: OperatorRequest,
) -> Tuple[Tuple[KernelCandidate, ConvGroupedSpec], ...]:
    """Every registered grouped-conv candidate that can launch ``req``.

    Probes opt-in variants and expands each candidate's ``sweep_space``.
    Production :func:`dispatch_conv_grouped` is unchanged.
    """
    if _request_errors(req):
        return ()
    assert isinstance(req, ConvGroupedRequest)
    return _registry_for(req).combos(req)


def conv_grouped_sweep_space(req: OperatorRequest) -> Sequence[ConvGroupedSpec]:
    if _request_errors(req):
        return ()
    assert isinstance(req, ConvGroupedRequest)
    return _registry_for(req).sweep_space(req, spec_key=lambda spec: spec.kernel_name())


def dispatch_conv_grouped_all(
    req: ConvGroupedRequest,
) -> Tuple[DispatchResult, ...]:
    """Every eligible grouped-conv kernel for ``req``, including opt-in variants."""
    if _request_errors(req):
        return ()
    return _registry_for(req).dispatch_all(req, kernel_id=_kernel_id)


def _signature_for(req: ConvGroupedRequest, spec: ConvGroupedSpec) -> tuple:
    """Launch signature for a dispatched kernel, including its variant extras.

    The trailing args are not cosmetic: dgrad always carries the tilde record
    buffer, wgrad always carries ``ks``/``ks_count``, and a two-stage wgrad
    also carries the workspace pair. Kernargs pack positionally, so omitting
    one shifts every argument after it.
    """
    from kernels.common.conv_abi import conv_args_signature, conv_direct_args_signature

    if spec.direct_kind:
        # grouped_dgrad's main kernel is the forward direct kernel run on the
        # transposed problem, so it takes the forward kernarg block.
        if req.direction == "fwd" or spec.direct_kind == "grouped_dgrad":
            return tuple(conv_direct_args_signature(req.dtype))
        return tuple(conv_direct_args_signature(req.dtype, direction=req.direction))

    p = _problem(req)
    dtype = req.dtype
    if req.direction == "wgrad":
        # Two-stage is decided by the resolver, not carried on the dispatcher
        # spec: it is what to_wgrad_spec builds the kernel from, so asking
        # anything else would drop ws_ptr/ws_bytes from a two-stage ABI.
        _split_k, two_stage, _requested = _resolve_wgrad_split_k(spec, p)
        return tuple(
            conv_args_signature(
                dtype,
                direction="wgrad",
                is_3d=p.is_3d,
                two_stage=two_stage,
            )
        )
    if req.direction == "dgrad":
        return tuple(conv_args_signature(dtype, direction="dgrad", is_3d=p.is_3d))
    return tuple(conv_args_signature(dtype, is_3d=p.is_3d))


def launch_values_for(
    req: ConvGroupedRequest,
    spec: ConvGroupedSpec,
    *,
    A_ptr: int,
    B_ptr: int,
    D_ptr: int,
    A_bytes: int,
    B_bytes: int,
    D_bytes: int,
    ws_ptr: Optional[int] = None,
    ws_bytes: Optional[int] = None,
    sub_gemm_buf: Optional[int] = None,
    num_sub_gemms: Optional[int] = None,
) -> dict:
    """Compute the full AOT ``values`` dict for a dispatch request.

    Operand roles by direction:
      fwd   -- A = activation, B = filter,  D = output
      wgrad -- A = dY,         B = X,       D = dW
      dgrad -- A = dY,         B = W,       D = dX

    The tile size comes from ``spec`` because the chiplet-swizzle path decodes
    the workgroup id against ``p_num_pid_m/n``, which are tile-count values.

    A spec with a :attr:`ConvGroupedSpec.prepass` reads (part of) its operands
    from the pre-pass workspace, so ``ws_ptr``/``ws_bytes`` are required and
    the values point the main kernel there; ``A``/``B``/``D`` stay the caller's
    tensors (see :func:`conv_grouped_prepass`).
    """
    from kernels.common.conv_args import ConvArgs

    if spec.prepass:
        pre = conv_grouped_prepass(req, spec)
        if ws_ptr is None or ws_bytes is None or ws_bytes < pre.workspace_bytes:
            raise ValueError(
                f"{spec.kernel_name()} reads the {spec.prepass} pre-pass workspace: "
                f"pass ws_ptr and ws_bytes >= {pre.workspace_bytes}"
            )
    if spec.direct_kind == "grouped_dgrad":
        # dX = conv(dY, W^T) on the transposed problem; W^T is the workspace.
        fp = spec.to_direct_spec(req).problem
        return ConvArgs.from_problem(fp).to_launch_values(
            A_ptr, ws_ptr, D_ptr, A_bytes, pre.workspace_bytes, D_bytes
        )
    if spec.direct_kind:
        # The direct kernels take the conv_direct kernarg block: the problem
        # extents and strides, with no tile counts.
        dp = _direct_problem(req)
        args = (
            ConvArgs.from_problem(dp)
            if req.direction == "fwd"
            else ConvArgs.from_problem(dp, direction=req.direction)
        )
        return args.to_launch_values(A_ptr, B_ptr, D_ptr, A_bytes, B_bytes, D_bytes)
    p = _problem(req)
    tile_m, tile_n = spec.tile_m, spec.tile_n
    if req.direction == "wgrad":
        # The resolved degree, not spec.split_k: the latter may be -1 (auto)
        # or exceed the clamp, while the kernel and _wgrad_grid both use the
        # resolver's value -- ks_count has to match the z extent it decodes.
        split_k, two_stage, _requested = _resolve_wgrad_split_k(spec, p)
        # Both workspace fields or neither: a two-stage launch missing only
        # ws_bytes would otherwise encode a zero-byte workspace, and Stage 1's
        # bounded stores would be silently dropped.
        has_ws_ptr = ws_ptr is not None
        has_ws_bytes = ws_bytes is not None
        if has_ws_ptr != has_ws_bytes or two_stage != has_ws_ptr:
            raise ValueError(
                "wgrad resolved to the "
                + ("two-stage" if two_stage else "single-stage")
                + " kernel; ws_ptr/ws_bytes must be passed exactly when it is "
                "two-stage"
            )
        # Deliberately the TRUE problem, not spec.group_merge's merged one.
        # Two separate reasons, and both have to hold:
        #   - The problem kernargs (p_C/p_K/p_cpg/p_kpg/p_wg_M/p_wg_N/...)
        #     describe the unmerged problem by contract: the kernel scales the
        #     ones the merged load side needs by Gm itself, since Gm is a
        #     build-time constant there. Handing it pre-merged dims would apply
        #     the factor twice.
        #   - p_num_pid_m/n are launch geometry, so they would have to be
        #     merged -- but wgrad_group_merge_available admits a merged spec
        #     only when grid_M <= tile_m and grid_N <= tile_n, i.e. the merged
        #     GEMM is exactly one tile. Both counts are therefore 1, and the
        #     unmerged dims (which are smaller still) also ceil to 1. The two
        #     agree by that gate, not by luck.
        # test_merged_tile_counts_are_single_tile pins the second one, so if
        # the gate ever admits a multi-tile merge this stops being true loudly.
        return ConvArgs.from_problem(
            p, direction="wgrad", tile_m=tile_m, tile_n=tile_n, tile_k=spec.tile_k
        ).to_launch_values(
            A_ptr,
            B_ptr,
            D_ptr,
            A_bytes,
            B_bytes,
            D_bytes,
            split_k=split_k,
            ws_ptr=ws_ptr,
            ws_bytes=ws_bytes,
        )
    if req.direction == "dgrad":
        if sub_gemm_buf is None or num_sub_gemms is None:
            raise ValueError(
                "dgrad needs sub_gemm_buf/num_sub_gemms: the tilde record "
                "buffer is always part of the dgrad kernarg ABI"
            )
        return ConvArgs.from_problem(
            p, direction="dgrad", tile_m=tile_m, tile_n=tile_n
        ).to_launch_values(
            A_ptr,
            B_ptr,
            D_ptr,
            A_bytes,
            B_bytes,
            D_bytes,
            sub_gemm_buf=sub_gemm_buf,
            num_sub_gemms=num_sub_gemms,
        )
    if spec.pad_c:
        # The implicit GEMM runs on the channel-padded problem, reading the
        # padded activation and filter from the two workspace segments.
        _x_rows, _w_rows, xp_bytes, wp_bytes = _channel_pad_segments(req, spec)
        p = replace(p, C=spec.pad_c)
        A_ptr, B_ptr = ws_ptr, ws_ptr + xp_bytes
        A_bytes, B_bytes = xp_bytes, wp_bytes
    return ConvArgs.from_problem(p, tile_m=tile_m, tile_n=tile_n).to_launch_values(
        A_ptr, B_ptr, D_ptr, A_bytes, B_bytes, D_bytes
    )


# ---------------------------------------------------------------------------
# Pre-pass (a second kernel launched before the dispatched one)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConvGroupedPrepass:
    """The kernel a :attr:`ConvGroupedSpec.prepass` spec needs launched first,
    on the same stream, before the dispatched kernel.

    The caller allocates ``workspace_bytes`` (no initialisation needed: the
    pre-pass writes every byte the main kernel reads), launches
    :meth:`build` with ``grid``/``block``/``signature`` and the values from
    :func:`prepass_launch_values_for`, then launches the dispatched kernel with
    :func:`launch_values_for` given the same ``ws_ptr``/``ws_bytes``.
    """

    kind: str  # "channel_pad" | "weight_transpose"
    kernel_spec: object
    grid: Tuple[int, int, int]
    block: Tuple[int, int, int]
    signature: tuple
    workspace_bytes: int

    def build(self, arch: str):
        if self.kind == "channel_pad":
            return build_conv_channel_pad(self.kernel_spec, arch=arch)
        return build_direct_transpose_weights_dgrad(self.kernel_spec, arch=arch)


def _channel_pad_segments(
    req: ConvGroupedRequest, spec: ConvGroupedSpec
) -> Tuple[int, int, int, int]:
    """``(x_rows, w_rows, xp_bytes, wp_bytes)`` of the channel-pad workspace:
    the padded activation segment first, the padded filter right after it."""
    x_rows = int(req.N) * int(req.Hi) * int(req.Wi)
    w_rows = int(req.K) * int(req.Y) * int(req.X)
    return x_rows, w_rows, x_rows * spec.pad_c * 2, w_rows * spec.pad_c * 2


def conv_grouped_prepass(
    req: ConvGroupedRequest, spec: ConvGroupedSpec
) -> Optional[ConvGroupedPrepass]:
    """The pre-pass ``spec`` needs before its main kernel, or None."""
    if spec.prepass == "channel_pad":
        ks = ChannelPadSpec(c_in=int(req.C), c_pad=spec.pad_c, dtype=req.dtype.lower())
        x_rows, w_rows, xp_bytes, wp_bytes = _channel_pad_segments(req, spec)
        return ConvGroupedPrepass(
            kind="channel_pad",
            kernel_spec=ks,
            grid=channel_pad_grid(ks, x_rows, w_rows),
            block=(ks.block_size, 1, 1),
            signature=tuple(channel_pad_signature(ks)),
            workspace_bytes=xp_bytes + wp_bytes,
        )
    if spec.prepass == "weight_transpose":
        from rocke.helpers.spec import SignatureBuilder

        p = _direct_problem(req)
        ks = DirectTransposeWeightsDgradSpec(problem=p)
        grid, block = direct_launch_geometry(ks)
        dtype = req.dtype.lower()
        sig = (
            SignatureBuilder()
            .ptr("A", dtype)
            .ptr("D", dtype)
            .scalar("A_bytes", "i32")
            .scalar("D_bytes", "i32")
            .build()
        )
        return ConvGroupedPrepass(
            kind="weight_transpose",
            kernel_spec=ks,
            grid=tuple(grid),
            block=tuple(block),
            signature=tuple(sig),
            workspace_bytes=direct_dgrad_workspace_bytes(p),
        )
    return None


def prepass_launch_values_for(
    req: ConvGroupedRequest,
    spec: ConvGroupedSpec,
    *,
    A_ptr: int,
    B_ptr: int,
    A_bytes: int,
    B_bytes: int,
    ws_ptr: int,
    ws_bytes: int,
) -> dict:
    """Kernarg values of the pre-pass, with the operand roles of
    :func:`launch_values_for` (fwd: A = X, B = W; dgrad: A = dY, B = W)."""
    pre = conv_grouped_prepass(req, spec)
    if pre is None:
        raise ValueError(f"{spec.kernel_name()} has no pre-pass")
    if ws_bytes < pre.workspace_bytes:
        raise ValueError(f"workspace {ws_bytes} B < required {pre.workspace_bytes} B")
    if pre.kind == "channel_pad":
        x_rows, w_rows, xp_bytes, _wp = _channel_pad_segments(req, spec)
        return {
            "X": A_ptr,
            "W": B_ptr,
            "Xp": ws_ptr,
            "Wp": ws_ptr + xp_bytes,
            "x_rows": x_rows,
            "w_rows": w_rows,
        }
    return {
        "A": B_ptr,
        "D": ws_ptr,
        "A_bytes": B_bytes,
        "D_bytes": pre.workspace_bytes,
    }


def dispatch_conv_grouped(
    req: ConvGroupedRequest, *, ranker: Ranker | None = None
) -> DispatchResult:
    """Select the appropriate grouped conv candidate (fwd or wgrad) for ``req``."""
    registry = _registry_for(req)
    candidate = registry.select(req, ranker=ranker)
    spec = candidate.select_spec(req)
    kid = _kernel_id(req, candidate, spec)
    explanation = [
        f"selected {candidate.name} ({req.direction}) on {req.arch}",
        f"algorithm={candidate.algorithm}",
        f"spec_id={candidate.spec_id}",
        f"epilogue={spec.epilogue} (vec_size_c={_vec_size_c(req)})",
        f"spec_hash={kid.spec_hash}",
        f"request_hash={kid.request_hash}",
    ]
    if req.direction == "wgrad":
        split_k, _two_stage, requested = _resolve_wgrad_split_k(spec, _problem(req))
        if split_k != requested:
            explanation.append(
                f"split_k {requested} -> {split_k}: the two-stage scratch would "
                f"exceed the {_MAX_WGRAD_WS_BYTES}-byte i32 limit, so the "
                f"plain store is used instead"
            )
    if spec.prepass:
        pre = conv_grouped_prepass(req, spec)
        explanation.append(
            f"prepass={spec.prepass}: launch it first with a "
            f"{pre.workspace_bytes}-byte workspace (conv_grouped_prepass)"
        )
    # AOT: signature includes all runtime problem-dim args.
    _sig = _signature_for(req, spec)
    return DispatchResult(
        request=req,
        candidate=candidate,
        spec=spec,
        kernel_id=kid,
        grid=candidate.grid(spec, req),
        block=candidate.block(spec),
        signature=_sig,
        explanation=tuple(explanation),
    )
