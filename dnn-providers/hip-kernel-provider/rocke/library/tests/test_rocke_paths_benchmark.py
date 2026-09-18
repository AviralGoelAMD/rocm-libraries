# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import csv

import pytest
import torch

import benchmarks.gfx942.attention.prefill.rocke_vs_aiter.rocke_paths as rocke_paths


def test_configs_match_the_ten_row_benchmark_cohort() -> None:
    assert [
        (config.id, config.B, config.S, config.Hq, config.Hkv)
        for config in rocke_paths.CONFIGS
    ] == [
        (1, 1, 4096, 32, 8),
        (2, 1, 4096, 32, 16),
        (3, 1, 8192, 32, 8),
        (4, 1, 8192, 32, 16),
        (5, 1, 16384, 32, 8),
        (6, 16, 4096, 32, 8),
        (7, 16, 8192, 32, 8),
        (8, 16, 4096, 32, 16),
        (9, 64, 4096, 32, 8),
        (10, 64, 8192, 32, 8),
    ]


def test_identity_paged_kv_preserves_each_cpu_token() -> None:
    k = torch.arange(2 * 128 * 2 * 4, dtype=torch.float32).reshape(2, 128, 2, 4)
    v = k + k.numel()

    k_cache, v_cache, block_table = rocke_paths.make_identity_paged_kv(k, v)

    assert k_cache.shape == (4, 64, 2, 4)
    assert v_cache.shape == (4, 64, 2, 4)
    assert torch.equal(k_cache.reshape_as(k), k)
    assert torch.equal(v_cache.reshape_as(v), v)
    assert block_table.dtype is torch.int32
    assert block_table.device == k.device
    assert torch.equal(block_table, torch.tensor([[0, 1], [2, 3]], dtype=torch.int32))


def test_identity_paged_kv_materializes_noncontiguous_inputs() -> None:
    k = torch.arange(2 * 128 * 2 * 8, dtype=torch.float32).reshape(2, 128, 2, 8)[
        ..., ::2
    ]
    v = torch.arange(2 * 128 * 2 * 8, dtype=torch.float32).add_(2 * 128 * 2 * 8).reshape(
        2, 128, 2, 8
    )[..., ::2]

    k_cache, v_cache, _ = rocke_paths.make_identity_paged_kv(k, v)

    assert not k.is_contiguous()
    assert not v.is_contiguous()
    assert k_cache.is_contiguous()
    assert v_cache.is_contiguous()
    assert torch.equal(k_cache.reshape_as(k), k)
    assert torch.equal(v_cache.reshape_as(v), v)


def test_identity_paged_kv_rejects_kv_on_different_devices() -> None:
    k = torch.empty(2, 128, 2, 4)
    v = torch.empty(2, 128, 2, 4, device="meta")

    with pytest.raises(ValueError, match="same device"):
        rocke_paths.make_identity_paged_kv(k, v)


def test_identity_paged_kv_rejects_mismatched_shapes() -> None:
    k = torch.empty(2, 128, 2, 4)
    v = torch.empty(2, 128, 2, 5)

    with pytest.raises(ValueError, match="identical"):
        rocke_paths.make_identity_paged_kv(k, v)


def test_identity_paged_kv_rejects_partial_page() -> None:
    k = torch.empty(2, 65, 2, 4)

    with pytest.raises(ValueError, match="divisible"):
        rocke_paths.make_identity_paged_kv(k, k)


def test_dense_spec_preflight_rejects_config_10_for_32_bit_qo_extent() -> None:
    from kernels.gfx942.attention_dense import supports_attention_dense

    spec = rocke_paths.dense_spec(rocke_paths.CONFIGS[-1])

    supported, reason = supports_attention_dense(spec, arch="gfx942")

    assert not supported
    assert "Q/O" in reason
    assert "32-bit" in reason or "extent" in reason


def test_dense_preflight_rows_keep_config_10_unsupported_without_cuda_allocation() -> None:
    dense, unified = rocke_paths.preflight_rocke_rows(rocke_paths.CONFIGS[-1])

    assert dense["status"] == "UNSUPPORTED"
    assert dense["ms"] is None
    assert dense["reason"]
    assert "Q/O" in dense["reason"]
    assert unified is not dense
    assert unified["path"] == "auto"
    assert unified["status"] == "ERROR"
    assert unified["reason"] == ""

def test_main_writes_stable_ordered_dense_and_unified_tsvs(tmp_path, monkeypatch) -> None:
    def stub_pair(config, *, warmup, iters, seed):
        dense = {
            **rocke_paths._row(config, path="attention_dense"),
            "ms": 1.25,
            "tflops": 2.5,
            "max_abs": 0.0,
            "status": "PASS",
        }
        unified = {
            **rocke_paths._row(config, path="auto:2d"),
            "ms": 1.5,
            "tflops": 2.0,
            "max_abs": 0.0,
            "status": "PASS",
        }
        return dense, unified

    monkeypatch.setattr(rocke_paths, "run_rocke_pair", stub_pair)
    dense_path = tmp_path / "dense.tsv"
    unified_path = tmp_path / "unified.tsv"

    assert rocke_paths.main(
        [
            "--out-dense",
            str(dense_path),
            "--out-unified",
            str(unified_path),
            "--warmup",
            "0",
            "--iters",
            "1",
            "--seed",
            "7",
        ]
    ) == 0

    expected_fields = [
        "id", "B", "S", "Hq", "Hkv", "GQA", "ms", "tflops", "max_abs",
        "status", "kernel", "path", "settings", "reason",
    ]
    for path, expected_path in (
        (dense_path, "attention_dense"),
        (unified_path, "auto:2d"),
    ):
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            rows = list(reader)
        assert reader.fieldnames == expected_fields
        assert [int(row["id"]) for row in rows] == list(range(1, 11))
        assert all(row["path"] == expected_path for row in rows)



def test_merge_results_keeps_passing_unified_config_when_dense_is_unsupported(
    tmp_path,
) -> None:
    from benchmarks.gfx942.attention.prefill.rocke_vs_aiter import merge_results

    common = {
        "id": "10",
        "B": "64",
        "S": "8192",
        "Hq": "32",
        "Hkv": "8",
        "GQA": "4:1",
    }

    def write_tsv(name: str, fields: list[str], row: dict[str, str]) -> None:
        with (tmp_path / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerow(row)

    rocke_fields = [
        "id",
        "B",
        "S",
        "Hq",
        "Hkv",
        "GQA",
        "ms",
        "tflops",
        "max_abs",
        "status",
        "kernel",
        "path",
        "settings",
        "reason",
    ]
    write_tsv(
        "rocke_dense.tsv",
        rocke_fields,
        {
            **common,
            "ms": "",
            "tflops": "",
            "max_abs": "",
            "status": "UNSUPPORTED",
            "kernel": "dense",
            "path": "attention_dense",
            "settings": "",
            "reason": "Q/O extent exceeds 32-bit",
        },
    )
    write_tsv(
        "rocke_unified.tsv",
        rocke_fields,
        {
            **common,
            "ms": "3.0",
            "tflops": "1.0",
            "max_abs": "0.0",
            "status": "PASS",
            "kernel": "unified",
            "path": "auto:2d",
            "settings": "",
            "reason": "",
        },
    )

    external_fields = [
        "id",
        "B",
        "S",
        "Hq",
        "Hkv",
        "GQA",
        "ms",
        "tflops",
        "gbps",
        "status",
        "validation_status",
        "kernel",
        "reason",
    ]
    write_tsv(
        "aiter.tsv",
        external_fields,
        {
            **common,
            "ms": "2.0",
            "tflops": "1.5",
            "gbps": "2.5",
            "status": "PASS",
            "validation_status": "PASS",
            "kernel": "aiter-asm",
            "reason": "",
        },
    )
    write_tsv(
        "ck.tsv",
        external_fields,
        {
            **common,
            "ms": "2.5",
            "tflops": "1.25",
            "gbps": "2.0",
            "status": "PASS",
            "validation_status": "PASS",
            "kernel": "ck-tile",
            "reason": "",
        },
    )

    rows = merge_results.merge(tmp_path)

    assert rows == [
        {
            **common,
            "dense_ms": None,
            "dense_status": "UNSUPPORTED",
            "dense_kernel": "dense",
            "dense_reason": "Q/O extent exceeds 32-bit",
            "unified_ms": 3.0,
            "unified_status": "PASS",
            "unified_kernel": "unified",
            "unified_reason": "",
            "best_rocke": "unified",
            "best_rocke_ms": 3.0,
            "AITER_ms": 2.0,
            "AITER_status": "PASS",
            "AITER_validation_status": "PASS",
            "AITER_kernel": "aiter-asm",
            "AITER_reason": "",
            "CK_ms": 2.5,
            "CK_status": "PASS",
            "CK_validation_status": "PASS",
            "CK_kernel": "ck-tile",
            "CK_reason": "",
            "dense_vs_unified": None,
            "AITER_vs_best_rocke": 1.5,
            "CK_vs_best_rocke": 1.2,
        }
    ]
    with (tmp_path / "results.csv").open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[0]["best_rocke"] == "unified"
    assert csv_rows[0]["AITER_vs_best_rocke"] == "1.5"
    markdown = (tmp_path / "benchmark_results.md").read_text()
    assert "dense: UNSUPPORTED — Q/O extent exceeds 32-bit" in markdown
    assert "| 10 | dense | UNSUPPORTED | — | dense | Q/O extent exceeds 32-bit |" in markdown
    assert "| 10 | unified | PASS | — | unified | — |" in markdown
    assert "| 10 | AITER | PASS | PASS | aiter-asm | — |" in markdown
    assert "| 10 | CK | PASS | PASS | ck-tile | — |" in markdown


def _gfx942_gpu_ready() -> bool:
    if not torch.cuda.is_available():
        return False
    device_index = torch.cuda.current_device()
    return "gfx942" in torch.cuda.get_device_properties(device_index).gcnArchName.lower()


@pytest.mark.skipif(
    not _gfx942_gpu_ready(), reason="needs a gfx942 GPU with ROCm torch"
)
@pytest.mark.gpu
def test_shared_fixture_dense_and_auto_unified_match_fp32_reference() -> None:
    dense, unified = rocke_paths.run_rocke_pair(
        rocke_paths.Config(0, 1, 512, 32, 8), warmup=1, iters=1, seed=0
    )

    assert dense["status"] == "PASS", dense["reason"]
    assert unified["status"] == "PASS", unified["reason"]
    assert dense["max_abs"] < rocke_paths.MAX_ABS_TOL
    assert unified["max_abs"] < rocke_paths.MAX_ABS_TOL
