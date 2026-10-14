# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Typed per-kernel policy for backend code generation.

Semantic kernel operations remain in :class:`KernelDef`; this module owns the
small set of validated backend choices that affect the compiled artifact without
becoming an unrestricted compiler-option interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from .ir import KernelDef

SCHEDULER_STRATEGY_ATTR = "scheduler_strategy"
F32_DENORMAL_MODE_ATTR = "f32_denormal_mode"


class SchedulerStrategy(str, Enum):
    """AMDGPU machine-scheduler strategies supported as stable kernel policy."""

    MAX_ILP = "max-ilp"
    MAX_MEMORY_CLAUSE = "max-memory-clause"
    ITERATIVE_ILP = "iterative-ilp"
    ITERATIVE_MINREG = "iterative-minreg"
    ITERATIVE_MAXOCC = "iterative-maxocc"


_SCHEDULER_STRATEGY_VALUES = frozenset(strategy.value for strategy in SchedulerStrategy)


def normalize_scheduler_strategy(value: object) -> str | None:
    """Return the canonical scheduler string or reject an unsupported value."""

    if value is None:
        return None
    if isinstance(value, SchedulerStrategy):
        return value.value
    if isinstance(value, str) and value in _SCHEDULER_STRATEGY_VALUES:
        return value
    choices = ", ".join(sorted(_SCHEDULER_STRATEGY_VALUES))
    raise ValueError(
        f"unsupported scheduler_strategy {value!r}; expected one of: {choices}, or None"
    )


class F32DenormalMode(str, Enum):
    """f32 denormal handling the kernel may assume (``denormal-fp-math-f32``).

    ``PRESERVE_SIGN`` flushes f32 denormal inputs/outputs to signed zero. The
    AMDGPU backend then lowers ``llvm.exp2`` / ``llvm.log2`` / ``llvm.sqrt`` /
    ``fdiv`` without the denormal range-scaling sequences it needs under IEEE.
    ``IEEE`` is the LLVM default and is never emitted explicitly.
    """

    IEEE = "ieee"
    PRESERVE_SIGN = "preserve-sign"


_F32_DENORMAL_MODE_VALUES = frozenset(mode.value for mode in F32DenormalMode)


def normalize_f32_denormal_mode(value: object) -> str | None:
    """Return the canonical f32 denormal mode, ``None`` for the default."""

    if value is None:
        return None
    if isinstance(value, F32DenormalMode):
        value = value.value
    if isinstance(value, str) and value in _F32_DENORMAL_MODE_VALUES:
        return None if value == F32DenormalMode.IEEE.value else value
    choices = ", ".join(sorted(_F32_DENORMAL_MODE_VALUES))
    raise ValueError(
        f"unsupported f32_denormal_mode {value!r}; expected one of: {choices}, or None"
    )


@dataclass(frozen=True)
class CodegenPolicy:
    """Validated code-generation choices belonging to one compiled kernel."""

    scheduler_strategy: str | SchedulerStrategy | None = None
    f32_denormal_mode: str | F32DenormalMode | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "scheduler_strategy",
            normalize_scheduler_strategy(self.scheduler_strategy),
        )
        object.__setattr__(
            self,
            "f32_denormal_mode",
            normalize_f32_denormal_mode(self.f32_denormal_mode),
        )


def codegen_policy_from_attrs(attrs: Mapping[str, Any]) -> CodegenPolicy:
    """Construct and validate policy from a kernel attribute mapping."""

    return CodegenPolicy(
        scheduler_strategy=attrs.get(SCHEDULER_STRATEGY_ATTR),
        f32_denormal_mode=attrs.get(F32_DENORMAL_MODE_ATTR),
    )


def codegen_policy_for_kernel(kernel: KernelDef) -> CodegenPolicy:
    """Return the validated code-generation policy carried by ``kernel``."""

    return codegen_policy_from_attrs(kernel.attrs)


def apply_codegen_policy(kernel: KernelDef, policy: CodegenPolicy) -> None:
    """Apply validated policy to ``kernel`` before serialization and lowering."""

    if not isinstance(policy, CodegenPolicy):
        raise TypeError(f"policy must be CodegenPolicy, got {type(policy).__name__}")
    for attr, value in (
        (SCHEDULER_STRATEGY_ATTR, policy.scheduler_strategy),
        (F32_DENORMAL_MODE_ATTR, policy.f32_denormal_mode),
    ):
        if value is None:
            kernel.attrs.pop(attr, None)
        else:
            kernel.attrs[attr] = value


__all__ = [
    "CodegenPolicy",
    "F32_DENORMAL_MODE_ATTR",
    "F32DenormalMode",
    "SCHEDULER_STRATEGY_ATTR",
    "SchedulerStrategy",
    "apply_codegen_policy",
    "codegen_policy_for_kernel",
    "codegen_policy_from_attrs",
    "normalize_f32_denormal_mode",
    "normalize_scheduler_strategy",
]
