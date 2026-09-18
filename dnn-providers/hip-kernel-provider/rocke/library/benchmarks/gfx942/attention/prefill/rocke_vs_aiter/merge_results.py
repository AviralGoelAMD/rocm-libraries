# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Merge gfx942 attention benchmark TSVs into machine- and human-readable reports."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

_INPUTS = {
    "dense": "rocke_dense.tsv",
    "unified": "rocke_unified.tsv",
    "AITER": "aiter.tsv",
    "CK": "ck.tsv",
}
_ARM_NAMES = ("dense", "unified", "AITER", "CK")

_IDENTITY_FIELDS = ("B", "S", "Hq", "Hkv", "GQA")
_RESULT_FIELDS = (
    "id",
    "B",
    "S",
    "Hq",
    "Hkv",
    "GQA",
    "dense_ms",
    "dense_tflops",
    "dense_max_abs",
    "dense_status",
    "dense_kernel",
    "dense_path",
    "dense_settings",
    "dense_reason",
    "unified_ms",
    "unified_tflops",
    "unified_max_abs",
    "unified_status",
    "unified_kernel",
    "unified_path",
    "unified_settings",
    "unified_reason",
    "best_rocke",
    "best_rocke_ms",
    "AITER_ms",
    "AITER_tflops",
    "AITER_gbps",
    "AITER_status",
    "AITER_validation_status",
    "AITER_kernel",
    "AITER_reason",
    "CK_ms",
    "CK_tflops",
    "CK_gbps",
    "CK_status",
    "CK_validation_status",
    "CK_kernel",
    "CK_reason",
    "dense_vs_unified",
    "AITER_vs_best_rocke",
    "CK_vs_best_rocke",
)


def _read_tsv(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            identifier = row.get("id")
            if not identifier:
                raise ValueError(f"{path.name}: row is missing id")
            if identifier in rows:
                raise ValueError(f"{path.name}: duplicate id {identifier}")
            rows[identifier] = row
    return rows


_TIMING_ELIGIBLE_VALIDATION_STATUSES = {"PASS", "UNAVAILABLE"}


def _ms(row: dict[str, str] | None) -> float | None:
    if row is None or row.get("status") != "PASS":
        return None
    try:
        value = float(row["ms"])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def _producer_field(row: dict[str, str] | None, field: str) -> float | str | None:
    """Preserve one producer metric while making ordinary numeric values numeric."""
    if row is None:
        return None
    value = row.get(field, "")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return value


def _ratio(base_ms: float | None, comparison_ms: float | None) -> float | None:
    if base_ms is None or comparison_ms is None:
        return None
    return base_ms / comparison_ms


def _arm(row: dict[str, str] | None, name: str) -> dict[str, Any]:
    status = "MISSING" if row is None else row.get("status", "MISSING")
    reason = "row is missing" if row is None else row.get("reason", "")
    validation_status = "MISSING" if row is None else row.get("validation_status", "")

    if (
        name in {"AITER", "CK"}
        and status == "PASS"
        and validation_status not in _TIMING_ELIGIBLE_VALIDATION_STATUSES
    ):
        status = "FAIL"
        if not reason:
            reason = f"validation status {validation_status or 'not recorded'}"

    result: dict[str, Any] = {
        f"{name}_ms": _ms(row) if status == "PASS" else None,
        f"{name}_status": status,
        f"{name}_kernel": "" if row is None else row.get("kernel", ""),
        f"{name}_reason": reason,
    }
    if name in {"dense", "unified"}:
        result.update(
            {
                f"{name}_tflops": _producer_field(row, "tflops"),
                f"{name}_max_abs": _producer_field(row, "max_abs"),
                f"{name}_path": "" if row is None else row.get("path", ""),
                f"{name}_settings": "" if row is None else row.get("settings", ""),
            }
        )
    if name in {"AITER", "CK"}:
        result.update(
            {
                f"{name}_tflops": _producer_field(row, "tflops"),
                f"{name}_gbps": _producer_field(row, "gbps"),
                f"{name}_validation_status": validation_status,
            }
        )
    return result


def _row_identity(*arms: dict[str, str] | None, identifier: str) -> dict[str, str]:
    source = next(arm for arm in arms if arm is not None)
    identity = {"id": identifier}
    for field in _IDENTITY_FIELDS:
        identity[field] = source.get(field, "")
        for arm in arms:
            if arm is not None and arm.get(field, "") != identity[field]:
                raise ValueError(f"id {identifier}: {field} differs between arms")
    return identity


def _merge_row(
    identifier: str,
    dense: dict[str, str] | None,
    unified: dict[str, str] | None,
    aiter: dict[str, str] | None,
    ck: dict[str, str] | None,
) -> dict[str, Any]:
    identity = _row_identity(dense, unified, aiter, ck, identifier=identifier)
    dense_ms = _ms(dense)
    unified_ms = _ms(unified)
    rocke_candidates = [
        ("dense", dense_ms),
        ("unified", unified_ms),
    ]
    best_rocke, best_rocke_ms = min(
        ((name, ms) for name, ms in rocke_candidates if ms is not None),
        key=lambda item: item[1],
        default=("", None),
    )
    aiter_ms = _ms(aiter)
    ck_ms = _ms(ck)

    return {
        **identity,
        **_arm(dense, "dense"),
        **_arm(unified, "unified"),
        "best_rocke": best_rocke,
        "best_rocke_ms": best_rocke_ms,
        **_arm(aiter, "AITER"),
        **_arm(ck, "CK"),
        "dense_vs_unified": _ratio(dense_ms, unified_ms),
        "AITER_vs_best_rocke": _ratio(best_rocke_ms, aiter_ms),
        "CK_vs_best_rocke": _ratio(best_rocke_ms, ck_ms),
    }


def _format_ms(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def _format_ratio(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}×"


def _format_metric(value: float | str | None, digits: int) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return value


def _markdown_cell(value: Any) -> str:
    """Render a TSV-derived value as one safe Markdown table cell."""
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\n", "<br>")
    )


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# MI300X/gfx942 BF16 causal attention benchmark",
        "",
        "Ratios are baseline latency divided by comparison latency. They are shown only when both arms passed with finite, positive GPU timings.",
        "",
        "| # | B | S | Hq | Hkv | GQA | rocKE dense ms | rocKE unified ms | best rocKE | AITER ms | CK ms | dense vs unified | AITER vs best rocKE | CK vs best rocKE |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        best = row["best_rocke"] or "—"
        if row["best_rocke_ms"] is not None:
            best = f"{best} ({_format_ms(row['best_rocke_ms'])} ms)"
        lines.append(
            f"| {_markdown_cell(row['id'])} | {_markdown_cell(row['B'])} | "
            f"{_markdown_cell(row['S'])} | {_markdown_cell(row['Hq'])} | "
            f"{_markdown_cell(row['Hkv'])} | {_markdown_cell(row['GQA'])} | "
            f"{_markdown_cell(_format_ms(row['dense_ms']))} | "
            f"{_markdown_cell(_format_ms(row['unified_ms']))} | "
            f"{_markdown_cell(best)} | {_markdown_cell(_format_ms(row['AITER_ms']))} | "
            f"{_markdown_cell(_format_ms(row['CK_ms']))} | "
            f"{_markdown_cell(_format_ratio(row['dense_vs_unified']))} | "
            f"{_markdown_cell(_format_ratio(row['AITER_vs_best_rocke']))} | "
            f"{_markdown_cell(_format_ratio(row['CK_vs_best_rocke']))} |"
        )

    lines.extend(
        [
            "",
            "## Arm details",
            "",
            "| # | arm | status | validation | kernel | reason |",
            "|---:|:---|:---|:---|:---|:---|",
        ]
    )
    for row in rows:
        for arm in _ARM_NAMES:
            validation_status = row.get(f"{arm}_validation_status") or "—"
            kernel = row[f"{arm}_kernel"] or "—"
            reason = row[f"{arm}_reason"] or "—"
            lines.append(
                f"| {_markdown_cell(row['id'])} | {_markdown_cell(arm)} | "
                f"{_markdown_cell(row[f'{arm}_status'])} | "
                f"{_markdown_cell(validation_status)} | {_markdown_cell(kernel)} | "
                f"{_markdown_cell(reason)} |"
            )

    lines.extend(
        [
            "",
            "## Throughput and rocKE metadata",
            "",
            "| # | arm | TFLOPS | GB/s | max abs | path | settings |",
            "|---:|:---|---:|---:|---:|:---|:---|",
        ]
    )
    for row in rows:
        for arm in _ARM_NAMES:
            lines.append(
                f"| {_markdown_cell(row['id'])} | {_markdown_cell(arm)} | "
                f"{_markdown_cell(_format_metric(row.get(f'{arm}_tflops'), 4))} | "
                f"{_markdown_cell(_format_metric(row.get(f'{arm}_gbps'), 4))} | "
                f"{_markdown_cell(_format_metric(row.get(f'{arm}_max_abs'), 6))} | "
                f"{_markdown_cell(row.get(f'{arm}_path') or '—')} | "
                f"{_markdown_cell(row.get(f'{arm}_settings') or '—')} |"
            )

    failures = []
    for row in rows:
        for arm in _ARM_NAMES:
            status = row[f"{arm}_status"]
            if status != "PASS":
                failures.append(
                    f"- Config {_markdown_cell(row['id'])} {_markdown_cell(arm)}: "
                    f"`{_markdown_cell(status)}` — "
                    f"{_markdown_cell(row[f'{arm}_reason'] or 'no reason recorded')}"
                )
    if failures:
        lines.extend(["", "## Nonpassing arms", "", *failures])

    path.write_text("\n".join(lines) + "\n")


def merge(out: Path) -> list[dict[str, Any]]:
    """Merge benchmark rows from ``out`` and write ``results.csv`` and Markdown."""
    inputs = {name: _read_tsv(out / filename) for name, filename in _INPUTS.items()}
    identifiers = sorted(
        set().union(*(rows.keys() for rows in inputs.values())), key=lambda value: int(value)
    )
    rows = [
        _merge_row(
            identifier,
            inputs["dense"].get(identifier),
            inputs["unified"].get(identifier),
            inputs["AITER"].get(identifier),
            inputs["CK"].get(identifier),
        )
        for identifier in identifiers
    ]
    with (out / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    _write_markdown(out / "benchmark_results.md", rows)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    merge(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
