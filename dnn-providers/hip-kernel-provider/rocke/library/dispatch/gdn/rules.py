# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx950 GDN/KDA decode tuning rules: one :class:`GdnDecodeSpace` per tile.

The axes are data in :mod:`.axes`. This module states which axes a request
reads (scope) and what is legal, as the hooks of the shared
:class:`rocke.dispatch.tuning.KnobSpace`, which makes every spec.

``base`` is the tile's :class:`GdnDecodeSpec` for one request: the request's
problem fields, the tile, and every knob at its dataclass default. Knobs are a
delta against that base; the defaults are problem-independent and
fingerprinted into ``config_key``, so one knob set names one configuration on
every problem.
"""

from __future__ import annotations

import dataclasses as dc
from functools import lru_cache
from typing import Iterable, Tuple

from kernels.gfx950.gdn_decode import (
    GdnDecodeSpec,
    dpp_reduce_applies,
    interleave_cols_applies,
    is_valid_spec,
    row_loads_applies,
    row_loads_overrides,
    stream_rows_applies,
    xcd_remap_applies,
)
from rocke.dispatch.tuning import KnobAxis, WavesPerEuSpace, axis_knob_names
from rocke.dispatch.tuning.walk import iter_knob_sets

from .axes import (
    AXES,
    CONV_ONCE_AXIS,
    DPP_REDUCE_AXIS,
    FULL_WAVES_PER_EU,
    INTERLEAVE_COLS_AXIS,
    NORM_GATE_ONCE_AXIS,
    OUT_LDS_AXIS,
    PRODUCTION_WAVES_PER_EU,
    ROW_LOADS_AXIS,
    STATE_LOAD_HINT_AXIS,
    STATE_STORE_HINT_AXIS,
    STREAM_ROWS_AXIS,
    XCD_REMAP_AXIS,
)
from .common import GdnDecodeTunedSpec

# Fields a request sets (the problem) and fields a variant sets (the tile).
# Neither is a knob, and neither belongs in the defaults fingerprint.
PROBLEM_FIELDS = frozenset(
    {
        "num_k_heads",
        "num_v_heads",
        "head_k_dim",
        "head_v_dim",
        "dtype",
        "state_dtype",
        "use_qk_l2norm",
        "gate_kind",
        "fuse_conv",
        "fuse_out_norm",
    }
)
TILE_FIELDS = frozenset({"num_warps", "warp_threads_k", "blocks_per_v_dim"})

_KNOB_NAMES = axis_knob_names(AXES) | {"waves_per_eu"}
_FIELD_DEFAULTS = {f.name: f.default for f in dc.fields(GdnDecodeSpec)}
assert _KNOB_NAMES <= set(_FIELD_DEFAULTS), _KNOB_NAMES - set(_FIELD_DEFAULTS)
assert (PROBLEM_FIELDS | TILE_FIELDS) <= set(_FIELD_DEFAULTS)
assert (
    PRODUCTION_WAVES_PER_EU[0]
    == FULL_WAVES_PER_EU[0]
    == _FIELD_DEFAULTS["waves_per_eu"]
)
assert set(PRODUCTION_WAVES_PER_EU) <= set(FULL_WAVES_PER_EU)

# Every dataclass default that is not a problem or tile field: the
# problem-independent values the knobs are a delta against.
_SPEC_DEFAULTS = {
    name: default
    for name, default in _FIELD_DEFAULTS.items()
    if name not in PROBLEM_FIELDS | TILE_FIELDS
}

# Every value each knob axis can take (its base value included). A pinned
# value outside this set is refused even where the knob is out of scope and
# would otherwise be dropped: a malformed value (``dpp_reduce="true"``) is
# never a configuration, on any problem. ``1`` / ``0`` equal ``True`` /
# ``False`` here, and the shared kit stores them as the bools they equal.
_AXIS_VALUES = {
    axis.name: frozenset(
        {_FIELD_DEFAULTS[axis.name]}
        | {value for choice in axis.choices for _, value in choice}
    )
    for axis in AXES
}


def _axes_of(spec: GdnDecodeSpec) -> Tuple[KnobAxis, ...]:
    return _scoped_axes(
        bool(spec.fuse_conv),
        bool(spec.fuse_out_norm),
        dpp_reduce_applies(spec),
        xcd_remap_applies(spec),
        stream_rows_applies(spec),
        interleave_cols_applies(spec),
        row_loads_applies(spec),
    )


@lru_cache(maxsize=None)
def _scoped_axes(
    fuse_conv: bool,
    fuse_out_norm: bool,
    dpp_reduce: bool,
    xcd_remap: bool,
    stream_rows: bool,
    interleave_cols: bool,
    row_loads: bool,
) -> Tuple[KnobAxis, ...]:
    axes = [STATE_LOAD_HINT_AXIS, STATE_STORE_HINT_AXIS]
    if fuse_conv:
        axes.append(CONV_ONCE_AXIS)
        if fuse_out_norm:
            axes.append(NORM_GATE_ONCE_AXIS)
    for applies, axis in (
        (dpp_reduce, DPP_REDUCE_AXIS),
        (xcd_remap, XCD_REMAP_AXIS),
        (stream_rows, STREAM_ROWS_AXIS),
        (interleave_cols, INTERLEAVE_COLS_AXIS),
        (fuse_out_norm, OUT_LDS_AXIS),
        (row_loads, ROW_LOADS_AXIS),
    ):
        if applies:
            axes.append(axis)
    return tuple(axes)


def _restates_auto(kernel: GdnDecodeSpec) -> bool:
    """A forced ``row_loads`` that is the order ``auto`` already picks for
    ``kernel`` (its knobs and ``waves_per_eu`` included): auto's kernel."""
    default = _FIELD_DEFAULTS["row_loads"]
    return kernel.row_loads != default and not row_loads_overrides(kernel)


@dc.dataclass(frozen=True)
class GdnDecodeSpace(WavesPerEuSpace):
    """One registered tile's space. Scope (what a request can read) drops a
    knob rather than refusing it:

    * ``conv_once`` needs ``fuse_conv``; ``norm_gate_once`` needs ``fuse_conv``
      and ``fuse_out_norm``; ``out_lds`` needs ``fuse_out_norm``;
    * ``dpp_reduce``, ``xcd_remap``, ``stream_rows``, ``interleave_cols`` and
      ``row_loads`` follow the kernel's ``*_applies`` predicates (the tiles
      where they change the emitted code);
    * a forced ``row_loads`` that is the order ``auto`` picks anyway
      (``row_loads_overrides`` is False: its knobs and ``waves_per_eu`` give
      auto that order) is auto's kernel, so it is dropped too (:meth:`inert`).
      The shared walk applies that drop at each ``waves_per_eu`` value, so
      at each value a knob set lists one forced order: the one auto does not
      choose there.

    So ``tuning_knobs`` alone (``tuning_id`` ``auto``) carry a configuration
    across shapes: where one of the knobs is out of scope it is dropped and the
    rest still apply. A stored ``tuning_id`` is narrower. It names the knob set
    that was canonical where it was recorded, so it replays only where every
    one of those knobs stays in scope; elsewhere it names no configuration and
    the pin is refused (:meth:`out_of_scope` lets the candidate name the
    knobs).

    A value outside its knob's axis (``state_load_hint="nt"``,
    ``dpp_reduce="true"``) is refused whether or not the knob is in scope, and
    so is a ``waves_per_eu`` that is not an ``int`` (``True``, ``None``,
    ``4.0``). Within scope the kernel validator (``is_valid_spec``) is the
    legality check: ``norm_gate_once`` without ``conv_once`` is refused, as is
    any value the kernel cannot build.

    ``waves_per_eu`` is the outer knob and shows in the id stem
    (:class:`~rocke.dispatch.tuning.WavesPerEuSpace`):
    ``{spec_id}_wpe{N}@{config_key}``. Each sweep level lists the values
    ``axes.py`` gives it; a pin may name any legal value.
    """

    # Whether a forced row_loads restates auto's order depends on
    # waves_per_eu, the outer knob, so the shared walk applies inert's drop
    # at each waves_per_eu value; a pin is canonicalized at its own.
    outer_dependent_knobs = frozenset({"row_loads"})

    def axes(self, base):
        return _axes_of(base)

    def known_knobs(self, base):
        return axis_knob_names(AXES)

    def refuse(self, base, knobs):
        bad = sorted(
            f"{name}={value!r}"
            for name, value in knobs.items()
            if name in _AXIS_VALUES and value not in _AXIS_VALUES[name]
        )
        if bad:
            return f"{bad} are not values of their knobs on {self.candidate_name}"
        return None

    def out_of_scope(self, base, knobs) -> Tuple[str, ...]:
        """The knobs in ``knobs`` (``waves_per_eu`` may be among them) that
        are well formed but out of scope for ``base``, or a ``row_loads`` that
        restates auto's order: the ones canonicalization drops on this
        problem. Whether ``row_loads`` restates auto is only judged on a spec
        the kernel accepts (``waves_per_eu`` in range included), so a bad
        value is reported by resolution under its own name."""
        in_scope = axis_knob_names(self.axes(base))
        known = {
            name: value
            for name, value in knobs.items()
            if name in _AXIS_VALUES and value in _AXIS_VALUES[name]
        }
        dropped = {name for name in known if name not in in_scope}
        outer = knobs.get(self.outer_knob, base.waves_per_eu)
        if "row_loads" in in_scope and "row_loads" in known and type(outer) is int:
            kept = {k: v for k, v in known.items() if k not in dropped}
            spec = dc.replace(base, **kept, waves_per_eu=outer)
            if is_valid_spec(spec, arch=self.arch)[0] and _restates_auto(spec):
                dropped.add("row_loads")
        return tuple(sorted(dropped))

    def canonicalize(self, base, knobs):
        # The shared kit reads an outer value of another type as the int it
        # converts to (True -> 1, 4.0 -> 4) and None as unset. A pin is never
        # rewritten into another configuration, so only an int is taken.
        value = knobs.get(self.outer_knob, 0)
        if type(value) is not int:
            return None, (
                f"{self.outer_knob}={value!r} is a {type(value).__name__}; it "
                f"takes an int on {self.candidate_name}"
            )
        return super().canonicalize(base, knobs)

    def build(self, base, knobs):
        return dc.replace(base, **knobs)

    def inert(self, base, kernel):
        if _restates_auto(kernel):
            return {"row_loads": f"{kernel.row_loads!r} is auto's order here"}
        return {}

    def validate(self, base, kernel):
        return is_valid_spec(kernel, arch=self.arch)

    def defaults(self, base, kernel):
        return dict(_SPEC_DEFAULTS)

    def outer_values(self, base, level):
        if level == "production":
            return PRODUCTION_WAVES_PER_EU
        return FULL_WAVES_PER_EU

    def outer_default(self, base):
        return int(base.waves_per_eu)

    def production(self, base, axes, is_valid) -> Iterable[dict]:
        """Every combination of the in-scope axes, the base first. The space
        is small enough to walk whole, so production and full sweeps cover the
        same knob sets; they differ only in the ``waves_per_eu`` values."""
        return iter_knob_sets(axes, self.root(base), is_valid)

    def wrap(self, base, kernel, knobs, key, tid):
        return GdnDecodeTunedSpec(
            arch=self.arch,
            candidate_name=self.candidate_name,
            tuning_id=tid,
            variant_id=self.variant_id,
            config_key=key,
            knobs=knobs,
            kernel_spec=kernel,
        )
