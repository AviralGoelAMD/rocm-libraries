# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Arch-neutral half of the GDN decode dispatcher family.

Holds everything that does not depend on the chip: the request dataclass, the
family identity, the dimension vocabulary the registry may gate on, the shared
request/selector validation, the request -> spec mapping, and the warp-tiled
candidate factory. The emitter is shared by every arch, so a per-arch module
(``gfx942.py``, ``gfx950.py``) declares only what differs by arch -- its name,
the gate kinds it serves, its static default tile and any measured tables --
and builds its candidates here.
"""

from __future__ import annotations

import dataclasses as dc
import numbers
from dataclasses import asdict, dataclass
from itertools import product
from typing import Callable, Optional, Tuple

from kernels.common.gdn_decode import (
    # Re-exported, never redeclared. The kernel owns what it covers; dispatch's
    # job is to state that coverage, not to restate it -- a copy drifts in the
    # direction that fails silently, rejecting a shape the kernel has since
    # learned to run.
    GDN_DTYPES,
    GdnDecodeSpec,
    build_gdn_decode,
    gdn_decode_grid,
    gdn_decode_signature,
    is_valid_spec,
)
from rocke.dispatch.core import (
    Capability,
    KernelCandidate,
    OperatorRequest,
    selector_matches as _shared_selector_matches,
)

FAMILY = "gdn_decode"

# Bumped when the kernel's argument contract changes, so a cached compile from
# an older layout can never be reused against a newer launcher.
GDN_ABI_VERSION = "rocke-gdn-decode/v1"

_DTYPE_ALIASES = {
    "bf16": "bf16",
    "bfloat16": "bf16",
    "f16": "f16",
    "fp16": "f16",
    "float16": "f16",
}


def normalize_dtype(dtype: str) -> str:
    d = str(dtype).strip().lower()
    return _DTYPE_ALIASES.get(d, d)


@dataclass(frozen=True)
class GdnDecodeRequest(OperatorRequest):
    """One gated-delta-rule single-token decode step.

    ``batch`` is the number of *active* sequences this step. It always
    contributes to the launch grid.

    For ``gate_kind="gdn"``, ``auto`` uses the static ``DEFAULT_TILE`` whenever
    it is legal; otherwise dispatch chooses a validator-admitted fallback.
    Neither ``batch`` nor ``num_v_heads`` chooses GDN's auto tile.

    For ``gate_kind="kda"``, ``auto`` is keyed on ``batch * num_v_heads`` -- the
    "work" -- so tensor parallelism selects the tile for heads local to a rank.

    ``gate_kind`` selects the forget-gate granularity: ``"gdn"`` (one scalar
    decay per head) or ``"kda"`` (a per-channel decay). It reaches the spec and
    therefore the kernel name, so the two never share a compile-cache entry.

    ``seq_len`` is not a field. This family is decode-only -- one token per
    sequence -- and a request carrying any other sequence length would be a
    different operator, so it is rejected rather than silently accepted.
    """

    batch: int
    arch: str
    num_k_heads: int = 16
    num_v_heads: int = 32
    head_k_dim: int = 128
    head_v_dim: int = 128
    op: str = "gdn_decode"
    dtype: str = "bf16"
    state_dtype: str = "bf16"
    use_qk_l2norm: bool = True
    algorithm: str = "auto"
    gate_kind: str = "gdn"
    spec_id: str = "auto"

    def normalized(self) -> dict:
        d = asdict(self)
        d["dtype"] = normalize_dtype(self.dtype)
        d["state_dtype"] = normalize_dtype(self.state_dtype)
        return d

    def dims(self) -> dict:
        return {
            "batch": int(self.batch),
            "num_k_heads": int(self.num_k_heads),
            "num_v_heads": int(self.num_v_heads),
            "head_k_dim": int(self.head_k_dim),
            "head_v_dim": int(self.head_v_dim),
        }


GDN_DIM_VOCABULARY = (
    "batch",
    "num_k_heads",
    "num_v_heads",
    "head_k_dim",
    "head_v_dim",
)


_INT_FIELDS = ("batch", "num_k_heads", "num_v_heads", "head_k_dim", "head_v_dim")
_STR_FIELDS = ("arch", "dtype", "state_dtype", "algorithm", "gate_kind", "spec_id")


def request_type_errors(req: GdnDecodeRequest) -> list:
    """Fields whose Python type is wrong for the request contract.

    The dataclass annotations are not enforced at runtime, and a request can
    arrive deserialized or hand-built. Casting a wrong type would rewrite it
    into a different valid request (``bool("False")`` is ``True``,
    ``int(128.9)`` is ``128``), so these are rejected instead. Any integral
    type is accepted for the integer fields; ``bool`` is not, although it is an
    ``int`` subclass.
    """
    errors = []
    for name in _INT_FIELDS:
        value = getattr(req, name)
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            errors.append(f"{name} must be an integer, got {value!r}")
    if not isinstance(req.use_qk_l2norm, bool):
        errors.append(f"use_qk_l2norm must be a bool, got {req.use_qk_l2norm!r}")
    for name in _STR_FIELDS:
        value = getattr(req, name)
        if not isinstance(value, str):
            errors.append(f"{name} must be a str, got {value!r}")
    return errors


def request_errors(req: OperatorRequest) -> list:
    """Shape-level rejections that are independent of any candidate or arch."""
    if not isinstance(req, GdnDecodeRequest):
        return [f"expected GdnDecodeRequest, got {type(req).__name__}"]
    errors = request_type_errors(req)
    if errors:
        return errors
    if req.batch <= 0:
        errors.append(f"batch must be positive, got {req.batch}")
    if req.num_k_heads <= 0 or req.num_v_heads <= 0:
        errors.append("head counts must be positive")
    elif req.num_v_heads % req.num_k_heads:
        errors.append(
            f"num_v_heads {req.num_v_heads} must be a multiple of "
            f"num_k_heads {req.num_k_heads}"
        )
    if req.head_k_dim <= 0 or req.head_v_dim <= 0:
        errors.append("head dims must be positive")
    if normalize_dtype(req.dtype) not in GDN_DTYPES:
        errors.append(f"unsupported dtype {req.dtype!r}")
    if normalize_dtype(req.state_dtype) not in GDN_DTYPES:
        errors.append(f"unsupported state_dtype {req.state_dtype!r}")
    if req.gate_kind not in ("gdn", "kda"):
        errors.append(
            f"unsupported gate_kind {req.gate_kind!r} (expected 'gdn' or 'kda')"
        )
    if req.gate_kind == "kda" and (req.head_k_dim != 128 or req.head_v_dim != 128):
        errors.append(
            "NOT_YET_IMPLEMENTED: KDA decode currently requires "
            "head_k_dim == head_v_dim == 128"
        )
    return errors


# Shared pin-selector, re-exported under this family's name (see dispatch core).
selector_matches = _shared_selector_matches


# ---------------------------------------------------------------------------
# Request -> spec mapping and the warp-tiled candidate factory.
#
# Everything below is arch-neutral: the emitter is shared, so a per-arch
# registry module states only what differs by arch -- its name, the gate kinds
# it serves, its static default tile and any measured tables -- and builds its
# candidates here.
# ---------------------------------------------------------------------------

# The configured warp-tiled tile space, identical on every arch. The kernel
# validator decides which tiles are legal for a given request.
NUM_WARPS = (1, 2, 4, 8, 16)
WARP_THREADS_K = (1, 2, 4, 8, 16, 32)
BLOCKS_PER_V_DIM = (1, 2, 4, 8, 16, 32)
TILE_SPACE = tuple(product(NUM_WARPS, WARP_THREADS_K, BLOCKS_PER_V_DIM))

Tile = Tuple[int, int, int]


def configured_tiles(default_tile: Tile) -> Tuple[Tile, ...]:
    """The full tile space in registry order: ``default_tile`` first."""
    if default_tile not in TILE_SPACE:
        raise ValueError(f"default tile {default_tile!r} is outside the tile space")
    return (default_tile,) + tuple(tile for tile in TILE_SPACE if tile != default_tile)


def tile_spec_id(tile: Tile) -> str:
    return f"nw{tile[0]}_wtk{tile[1]}_bpv{tile[2]}"


def make_spec(req: GdnDecodeRequest, tile: Tile) -> GdnDecodeSpec:
    """Map a request plus a chosen tile onto a concrete kernel spec.

    Refuses a request whose fields have the wrong type instead of coercing
    them: ``bool("False")`` is ``True`` and ``int(128.9)`` is ``128``, so a
    cast here would turn a malformed request into a different, valid kernel.
    """
    errors = request_type_errors(req)
    if errors:
        raise TypeError("; ".join(errors))
    num_warps, warp_threads_k, blocks_per_v_dim = tile
    return dc.replace(
        GdnDecodeSpec(),
        num_k_heads=int(req.num_k_heads),
        num_v_heads=int(req.num_v_heads),
        head_k_dim=int(req.head_k_dim),
        head_v_dim=int(req.head_v_dim),
        dtype=normalize_dtype(req.dtype),
        state_dtype=normalize_dtype(req.state_dtype),
        use_qk_l2norm=req.use_qk_l2norm,
        gate_kind=req.gate_kind,
        num_warps=num_warps,
        warp_threads_k=warp_threads_k,
        blocks_per_v_dim=blocks_per_v_dim,
    )


# ``auto`` policy for one candidate: given an ``auto`` request, the reason this
# candidate is NOT the auto choice, or ``None`` when it may serve it.
AutoReject = Callable[[GdnDecodeRequest], Optional[str]]


def static_default_auto(default_tile: Tile, tile: Tile) -> AutoReject:
    """GDN ``auto``: the static default tile whenever the validator admits it.

    When the default is illegal for the request (for example head dim 64),
    every legal tile stays eligible and registry priority picks the first.
    """

    def reject(req: GdnDecodeRequest) -> Optional[str]:
        if tile == default_tile:
            return None
        if is_valid_spec(make_spec(req, default_tile), arch=req.arch)[0]:
            return f"static GDN auto tile is {default_tile!r}, not {tile!r}"
        return None

    return reject


def _grid(spec: GdnDecodeSpec, req: OperatorRequest) -> Tuple[int, int, int]:
    assert isinstance(req, GdnDecodeRequest)
    return gdn_decode_grid(int(req.batch), spec)


def _build(spec: GdnDecodeSpec, arch: str):
    return build_gdn_decode(spec, arch=arch)


def make_candidate(
    *,
    arch: str,
    served_gate_kinds: Tuple[str, ...],
    tile: Tile,
    priority: int,
    auto_reject: AutoReject,
    gate_kind: str = "gdn",
    spec_id: Optional[str] = None,
) -> KernelCandidate:
    """One warp-tiled decode candidate for ``arch``.

    ``served_gate_kinds`` is every gate kind the arch registers. A request for
    any other gate kind is refused as not yet implemented on this arch, not as
    impossible: the emitter and validator are arch-neutral, only the on-device
    validation is missing.
    """
    spec_id = spec_id or tile_spec_id(tile)
    name = f"gdn_decode_{arch}_{spec_id}"

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, GdnDecodeRequest)
        if req.arch != arch:
            return False, f"candidate arch {arch} != request arch {req.arch!r}"
        if req.gate_kind not in served_gate_kinds:
            return False, (
                f"NOT_YET_IMPLEMENTED: {req.gate_kind!r} decode is not validated on "
                f"{arch}; this arch serves {served_gate_kinds!r}"
            )
        # A candidate belongs to exactly one gate kind's table. Serving the
        # other kind would hand the request a tile tuned for a different
        # kernel, which is the failure the split table exists to prevent.
        if req.gate_kind != gate_kind:
            return False, (
                f"candidate {spec_id!r} is tuned for the {gate_kind!r} gate, "
                f"request asks for {req.gate_kind!r}"
            )
        ok, why = selector_matches(req, candidate)
        if not ok:
            return False, why
        if req.spec_id.strip().lower() == "auto":
            why = auto_reject(req)
            if why is not None:
                return False, why
        # Final authority is the kernel's own validator.
        return is_valid_spec(make_spec(req, tile), arch=req.arch)

    def select(req: OperatorRequest) -> GdnDecodeSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, GdnDecodeRequest)
        return make_spec(req, tile)

    candidate = KernelCandidate(
        name=name,
        family=FAMILY,
        algorithm="warp_tiled",
        spec_id=spec_id,
        abi_version=GDN_ABI_VERSION,
        priority=priority,
        capability=Capability(arches=(arch,), dtypes=GDN_DTYPES),
        _supports=support,
        select_spec=select,
        signature=lambda spec: gdn_decode_signature(spec),
        grid=_grid,
        block=lambda spec: (int(spec.block_size), 1, 1),
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
        build=_build,
        # No `bind`: this family is selectable but not launchable through the
        # generic runner, so today EVERY launch goes through the driver's
        # `prepare()` and therefore through `_validate_decode_inputs`. That is
        # the only thing standing between a mis-shaped tensor and an
        # out-of-bounds access -- the kernel emits no buffer descriptor, so
        # there is no `num_records` to clamp one. Whoever adds `bind` must
        # route it through that validator, or the checks stop covering the
        # path callers actually use.
    )
    return candidate


def gdn_tile_candidates(
    *,
    arch: str,
    served_gate_kinds: Tuple[str, ...],
    default_tile: Tile,
    first_priority: int = 10,
) -> Tuple[KernelCandidate, ...]:
    """Every configured GDN tile for ``arch``, ``default_tile`` first."""
    return tuple(
        make_candidate(
            arch=arch,
            served_gate_kinds=served_gate_kinds,
            tile=tile,
            priority=first_priority + i,
            auto_reject=static_default_auto(default_tile, tile),
        )
        for i, tile in enumerate(configured_tiles(default_tile))
    )
