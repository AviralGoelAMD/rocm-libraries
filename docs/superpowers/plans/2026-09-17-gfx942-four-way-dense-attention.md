# gfx942 Four-Way Dense Attention Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a correctness-gated four-way gfx942 dense-attention benchmark that compares rocKE `attention_dense`, rocKE auto-unified attention, AITER FMHA-v3 ASM, and CK Tile FMHA over the existing ten rows.

**Architecture:** Move the rocKE portion out of the shell heredoc into an importable Python runner. That runner owns the fixed cohort, creates one BF16 dense fixture per row, maps the same K/V tensors into an identity-paged cache for auto-unified attention, validates both rocKE outputs against one FP32 GQA SDPA reference, and times both with rocKE HIP events. The shell script orchestrates that runner plus external AITER/CK native validation and merges four TSV files into one report.

**Tech Stack:** Python 3.12, PyTorch ROCm, rocKE Python DSL/runtime, pytest, Bash, AITER FMHA-v3, CK Tile FMHA.

---

## File structure

| File | Responsibility |
|---|---|
| `library/benchmarks/gfx942/attention/prefill/rocke_vs_aiter/rocke_paths.py` | Fixed cohort, dense-to-identity-page conversion, shared-fixture rocKE execution, FP32 reference gate, HIP-event timing, and dense/unified TSV output. |
| `library/benchmarks/gfx942/attention/prefill/rocke_vs_aiter/merge_results.py` | Four-TSV result merger, best-correct-rocKE selection, ratio calculation, CSV output, and Markdown report. |
| `library/tests/test_rocke_paths_benchmark.py` | CPU tests for the immutable cohort, identity-page mapping, and dense preflight coverage result; GPU test for shared-fixture dense/unified numerical parity. |
| `library/benchmarks/gfx942/attention/prefill/rocke_vs_aiter/run_mi300x_attention_bench_updated.sh` | Invokes the new rocKE runner, adds native AITER/CK GPU-validation passes, requests GPU timing for every external arm, and produces a four-way merged report. |
| `library/benchmarks/gfx942/attention/prefill/rocke_vs_aiter/README.md` | Documents the four arms, what is shared, correctness gates, result files, and the dense unsupported coverage row. |

## Task 1: Add testable rocKE benchmark primitives

**Files:**
- Create: `dnn-providers/hip-kernel-provider/rocke/library/benchmarks/gfx942/attention/prefill/rocke_vs_aiter/rocke_paths.py`
- Create: `dnn-providers/hip-kernel-provider/rocke/library/tests/test_rocke_paths_benchmark.py`

- [ ] **Step 1: Write the failing CPU tests for cohort and identity paging**

```python
from benchmarks.gfx942.attention.prefill.rocke_vs_aiter.rocke_paths import (
    CONFIGS,
    make_identity_paged_kv,
)


def test_cohort_preserves_the_ten_external_benchmark_rows():
    assert [(r.ident, r.batch, r.seqlen, r.hq, r.hkv) for r in CONFIGS] == [
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


def test_identity_paging_preserves_each_dense_kv_token():
    import torch

    k = torch.arange(2 * 128 * 2 * 4, dtype=torch.float16).reshape(2, 128, 2, 4)
    v = k + 1000
    k_cache, v_cache, block_table = make_identity_paged_kv(k, v, page_size=64)

    assert k_cache.shape == (4, 64, 2, 4)
    assert torch.equal(block_table, torch.tensor([[0, 1], [2, 3]], dtype=torch.int32))
    assert torch.equal(k_cache.reshape_as(k), k)
    assert torch.equal(v_cache.reshape_as(v), v)
```

- [ ] **Step 2: Run the CPU tests and verify they fail because the module does not exist**

Run:

```bash
PYTHONPATH=dnn-providers/hip-kernel-provider/rocke/library:dnn-providers/hip-kernel-provider/rocke/platform/python \
python -m pytest dnn-providers/hip-kernel-provider/rocke/library/tests/test_rocke_paths_benchmark.py -q
```

Expected: collection fails with `ModuleNotFoundError` for `rocke_paths`.

- [ ] **Step 3: Implement only the immutable data and identity-page mapping**

Create `rocke_paths.py` with these public definitions:

```python
from dataclasses import dataclass

import torch

PAGE_SIZE = 64
MAX_ABS_TOL = 4e-2


@dataclass(frozen=True)
class Config:
    ident: int
    batch: int
    seqlen: int
    hq: int
    hkv: int


CONFIGS = (
    Config(1, 1, 4096, 32, 8),
    Config(2, 1, 4096, 32, 16),
    Config(3, 1, 8192, 32, 8),
    Config(4, 1, 8192, 32, 16),
    Config(5, 1, 16384, 32, 8),
    Config(6, 16, 4096, 32, 8),
    Config(7, 16, 8192, 32, 8),
    Config(8, 16, 4096, 32, 16),
    Config(9, 64, 4096, 32, 8),
    Config(10, 64, 8192, 32, 8),
)


def make_identity_paged_kv(k, v, *, page_size: int = PAGE_SIZE):
    batch, seqlen, hkv, head_size = k.shape
    if v.shape != k.shape:
        raise ValueError(f"K/V shapes differ: {tuple(k.shape)} != {tuple(v.shape)}")
    if seqlen % page_size:
        raise ValueError(f"seqlen={seqlen} is not divisible by page_size={page_size}")
    pages_per_sequence = seqlen // page_size
    cache_shape = (batch * pages_per_sequence, page_size, hkv, head_size)
    k_cache = k.reshape(cache_shape).contiguous()
    v_cache = v.reshape(cache_shape).contiguous()
    block_table = torch.arange(
        batch * pages_per_sequence, dtype=torch.int32, device=k.device
    ).reshape(batch, pages_per_sequence)
    return k_cache, v_cache, block_table
```

Keep this task free of GPU compilation and timing.

- [ ] **Step 4: Run the CPU tests and verify they pass**

Run the command from Step 2.

Expected: `2 passed`.

- [ ] **Step 5: Commit the tested primitive**

```bash
git add dnn-providers/hip-kernel-provider/rocke/library/benchmarks/gfx942/attention/prefill/rocke_vs_aiter/rocke_paths.py \
        dnn-providers/hip-kernel-provider/rocke/library/tests/test_rocke_paths_benchmark.py
git commit -m "test: cover rocKE four-way benchmark paging"
```

## Task 2: Implement shared-fixture rocKE dense and auto-unified execution

**Files:**
- Modify: `dnn-providers/hip-kernel-provider/rocke/library/benchmarks/gfx942/attention/prefill/rocke_vs_aiter/rocke_paths.py`
- Modify: `dnn-providers/hip-kernel-provider/rocke/library/tests/test_rocke_paths_benchmark.py`

- [ ] **Step 1: Write the failing dense preflight and shared-fixture GPU test**

Add these tests. Keep the GPU test guarded by the existing `gcnArchName`-based gfx942 predicate used by `test_attention_dense_gfx942_numeric.py`.

```python
def test_dense_preflight_reports_the_required_final_coverage_gap():
    from benchmarks.gfx942.attention.prefill.rocke_vs_aiter.rocke_paths import (
        CONFIGS,
        Config,
        dense_spec,
    )
    from kernels.gfx942.attention_dense import supports_attention_dense

    spec = dense_spec(CONFIGS[-1])
    ok, reason = supports_attention_dense(spec, arch="gfx942")
    assert not ok
    assert "32-bit" in reason or "extent" in reason


@requires_gfx942_gpu
@pytest.mark.gpu
def test_dense_and_auto_unified_share_fixture_and_match_fp32_sdpa():
    from benchmarks.gfx942.attention.prefill.rocke_vs_aiter.rocke_paths import run_rocke_pair

    dense, unified = run_rocke_pair(
        Config(0, 1, 512, 32, 8), warmup=1, iters=1, seed=0
    )
    assert dense["status"] == "PASS"
    assert unified["status"] == "PASS"
    assert dense["max_abs"] < 4e-2
    assert unified["max_abs"] < 4e-2
```

- [ ] **Step 2: Run the tests and verify the new symbols fail to import**

Run:

```bash
PYTHONPATH=dnn-providers/hip-kernel-provider/rocke/library:dnn-providers/hip-kernel-provider/rocke/platform/python \
python -m pytest dnn-providers/hip-kernel-provider/rocke/library/tests/test_rocke_paths_benchmark.py -q
```

Expected: CPU collection fails because `dense_spec` and `run_rocke_pair` are absent. On a gfx942 runner, the GPU test must fail for the same missing symbols rather than skip for another reason.

- [ ] **Step 3: Add dispatch-resolved dense construction, auto-unified construction, and one FP32 reference**

Implement these behaviors in `rocke_paths.py`:

1. `dense_spec(config)` builds `AttentionRequest` with `arch="gfx942"`, `mask_type=1`, `dtype="bf16"`, `algorithm="attention_dense"`, and `spec_id="gfx942_attention_dense"`; resolve it through `dense_spec_for_request`.
2. Allocate deterministic BF16 dense tensors once: Q `[B,S,Hq,128]`, K/V `[B,S,Hkv,128]`, and use `scale = 1 / sqrt(128)`.
3. Build the FP32 causal GQA reference by transposing Q/K/V to `[B,H,S,D]`, expanding K/V with `repeat_interleave(Hq // Hkv, dim=1)`, calling `torch.nn.functional.scaled_dot_product_attention(..., is_causal=True, scale=scale)`, and transposing back.
4. Run dense via `run_attention_dense_torch` only after `supports_attention_dense` admits its spec. When rejected, return `status="UNSUPPORTED"`, its structured reason, and no timing or error.
5. Build `UnifiedAttentionProblem(total_q=B*S, num_seqs=B, num_query_heads=Hq, num_kv_heads=Hkv, head_size=128, block_size=64, max_seqlen_q=S, max_seqlen_k=S, dtype="bf16", num_cus=torch.cuda.get_device_properties(0).multi_processor_count)`.
6. Convert K/V once with `make_identity_paged_kv`; flatten Q and unified output to `[B*S,Hq,128]`; set `cu_seqlens_q=torch.arange(B+1, dtype=torch.int32, device="cuda") * S` and `seqused_k=torch.full((B,), S, dtype=torch.int32, device="cuda")`.
7. Invoke `run_unified_attention_torch(..., backend="auto", softcap=0.0, sinks=None, stream=current_stream)` once before validation and once per timing launch.
8. Validate both outputs against the one reference. Set `status="PASS"` only when `max_abs < MAX_ABS_TOL`; otherwise use `status="FAIL"` with a reason containing the measured error and threshold.
9. Time only launches through `rocke.runtime.time_launches`; do not include allocation, identity paging, compilation, or reference execution.
10. Record dense `gfx942_kernel_name(spec)`, dense dispatch fields, unified `problem.select_path()`, and the resolved auto-unified spec name. For tiled 2D use `_tiled_spec_from_problem(problem).kernel_name()`; for 3D record both `_tiled_3d_spec_from_problem(problem)` stage names; for scalar fallback record `rocke_unified_attention_2d_scalar`.

Return two independent row dictionaries with common fields `id`, `B`, `S`, `Hq`, `Hkv`, `GQA`, `ms`, `tflops`, `max_abs`, `status`, `kernel`, `path`, and `reason`.

- [ ] **Step 4: Run targeted tests and verify green**

Run the CPU test command from Step 2. On a gfx942 node, also run:

```bash
HIP_VISIBLE_DEVICES=0 \
PYTHONPATH=dnn-providers/hip-kernel-provider/rocke/library:dnn-providers/hip-kernel-provider/rocke/platform/python \
python -m pytest dnn-providers/hip-kernel-provider/rocke/library/tests/test_rocke_paths_benchmark.py -m gpu -q
```

Expected: CPU tests pass; GPU test passes with both errors below `4e-2`.

- [ ] **Step 5: Add the command-line TSV runner and commit**

Add `main()` with `--out-dense`, `--out-unified`, `--warmup`, `--iters`, and `--seed`. It loops over `CONFIGS`, writes the two TSV files, prints each full row, and returns nonzero if any supported rocKE result is `FAIL` or `ERROR`. It must return zero for the expected dense `UNSUPPORTED` final row.

```bash
git add dnn-providers/hip-kernel-provider/rocke/library/benchmarks/gfx942/attention/prefill/rocke_vs_aiter/rocke_paths.py \
        dnn-providers/hip-kernel-provider/rocke/library/tests/test_rocke_paths_benchmark.py
git commit -m "feat: compare dense and unified rocKE attention"
```

## Task 3: Wire four arms and their correctness statuses into the shell runner

**Files:**
- Modify: `dnn-providers/hip-kernel-provider/rocke/library/benchmarks/gfx942/attention/prefill/rocke_vs_aiter/run_mi300x_attention_bench_updated.sh:4-658`

- [ ] **Step 1: Write the failing four-way report test**

Add a CPU test that creates four one-row TSV fixtures where dense is `UNSUPPORTED`, unified/AITER/CK are `PASS`, then asserts the merged CSV labels unified as `best_rocke` and computes external ratios from unified rather than dense.

```python
import csv


def test_merge_uses_unified_when_dense_is_unsupported(tmp_path):
    from benchmarks.gfx942.attention.prefill.rocke_vs_aiter.merge_results import merge

    fields = ["id", "B", "S", "Hq", "Hkv", "GQA", "ms", "status", "kernel", "reason"]

    def write_arm(name, *, status, ms):
        with (tmp_path / f"{name}.tsv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerow(
                {
                    "id": 10,
                    "B": 64,
                    "S": 8192,
                    "Hq": 32,
                    "Hkv": 8,
                    "GQA": "4:1",
                    "ms": ms,
                    "status": status,
                    "kernel": name,
                    "reason": "32-bit extent" if status == "UNSUPPORTED" else "",
                }
            )

    write_arm("rocke_dense", status="UNSUPPORTED", ms="")
    write_arm("rocke_unified", status="PASS", ms="3.0")
    write_arm("aiter", status="PASS", ms="2.0")
    write_arm("ck", status="PASS", ms="2.5")

    rows = merge(tmp_path)
    assert rows[0]["best_rocke"] == "unified"
    assert rows[0]["best_rocke_ms"] == 3.0
    assert rows[0]["AITER_vs_best_rocke"] == 1.5
```

- [ ] **Step 2: Run the test and verify it fails because `merge_results` is absent**

Run:

```bash
PYTHONPATH=dnn-providers/hip-kernel-provider/rocke/library:dnn-providers/hip-kernel-provider/rocke/platform/python \
python -m pytest dnn-providers/hip-kernel-provider/rocke/library/tests/test_rocke_paths_benchmark.py -q
```

Expected: import failure for `merge_results`.

- [ ] **Step 3: Implement the checked-in four-way merger**

Create `merge_results.py` beside `rocke_paths.py`. Its `merge(out: Path)` reads `rocke_dense.tsv`, `rocke_unified.tsv`, `aiter.tsv`, and `ck.tsv`, then writes `results.csv` and `benchmark_results.md`.

For each ID:

```python
valid_rocke = [row for row in (dense, unified) if row["status"] == "PASS"]
best = min(valid_rocke, key=lambda row: float(row["ms"])) if valid_rocke else None
```

Emit `best_rocke`, `best_rocke_ms`, `dense_vs_unified`, `AITER_vs_best_rocke`, and `CK_vs_best_rocke`. A ratio is blank unless both operands are passing, finite latency values. The Markdown report must show dense and unified latency/status separately and list every non-passing arm with its reason.

- [ ] **Step 4: Make the shell runner call checked-in rocKE and merge scripts**

Replace the generated `rocke_runner.py` heredoc with:

```bash
OUT_DENSE="$OUT/rocke_dense.tsv" \
OUT_UNIFIED="$OUT/rocke_unified.tsv" \
bash -lc "source '$ROCKE_ENV'; cd '$ROCM_LIBS/dnn-providers/hip-kernel-provider'; \
python '$SCRIPT_DIR/rocke_paths.py' \
  --out-dense '$OUT/rocke_dense.tsv' \
  --out-unified '$OUT/rocke_unified.tsv' \
  --warmup '$WARMUP' --iters '$REPEAT'" \
  2>&1 | tee "$OUT/logs/rocke/all.log"
```

Change the directories to `logs/rocke`, `logs/aiter`, and `logs/ck`; preserve existing environment metadata and all ten shapes.

- [ ] **Step 5: Add external native validation before timing**

For both generated external Python runners:

1. Build a validation command with `-v=2`, `-warmup=0`, and `-repeat=1`.
2. Run it before the existing support/timing command and write `config_XX.validation.log`.
3. Set `status="FAIL"` with the validation return code when it fails; do not parse or accept its timing.
4. Preserve AITER’s separate `-is_v3_check=1` support check and require its expected ASM name on the timed command.
5. Add `-timer=gpu` to CK’s command alongside its existing `-num_splits=1` so both external commands explicitly use GPU timing.
6. Extend each external TSV schema with `validation_status` and preserve `kernel` and `reason`.

- [ ] **Step 6: Replace the report heredoc and run static validation**

Replace the final inline Python report block with:

```bash
OUT="$OUT" WARMUP="$WARMUP" REPEAT="$REPEAT" \
python "$SCRIPT_DIR/merge_results.py"
```

Run:

```bash
bash -n dnn-providers/hip-kernel-provider/rocke/library/benchmarks/gfx942/attention/prefill/rocke_vs_aiter/run_mi300x_attention_bench_updated.sh
PYTHONPATH=dnn-providers/hip-kernel-provider/rocke/library:dnn-providers/hip-kernel-provider/rocke/platform/python \
python -m pytest dnn-providers/hip-kernel-provider/rocke/library/tests/test_rocke_paths_benchmark.py -q
```

Expected: shell syntax succeeds and the merger test passes.

- [ ] **Step 7: Commit integration wiring**

```bash
git add dnn-providers/hip-kernel-provider/rocke/library/benchmarks/gfx942/attention/prefill/rocke_vs_aiter/rocke_paths.py \
        dnn-providers/hip-kernel-provider/rocke/library/benchmarks/gfx942/attention/prefill/rocke_vs_aiter/merge_results.py \
        dnn-providers/hip-kernel-provider/rocke/library/benchmarks/gfx942/attention/prefill/rocke_vs_aiter/run_mi300x_attention_bench_updated.sh \
        dnn-providers/hip-kernel-provider/rocke/library/tests/test_rocke_paths_benchmark.py
git commit -m "feat: report four-way gfx942 attention results"
```

## Task 4: Document and run the full acceptance benchmark

**Files:**
- Modify: `dnn-providers/hip-kernel-provider/rocke/library/benchmarks/gfx942/attention/prefill/rocke_vs_aiter/README.md`

- [ ] **Step 1: Update the README’s implementation and methodology sections**

Replace “three-way” descriptions with four arms. State exactly:

- the two rocKE paths share the same in-process BF16 Q/K/V fixture;
- unified reads identity-paged K/V outside timing, while the logical workload remains dense;
- both rocKE outputs must satisfy FP32 SDPA `max_abs < 4e-2`;
- AITER and CK run native GPU validation before their timed commands but cannot consume the in-process fixture;
- `B=64, S=8192, Hkv=8` is an expected `attention_dense` coverage rejection, not a timed rocKE-dense datapoint;
- `results.csv` names the fastest passing rocKE path before external ratios are calculated.

- [ ] **Step 2: Run focused tests before the GPU benchmark**

```bash
PYTHONPATH=dnn-providers/hip-kernel-provider/rocke/library:dnn-providers/hip-kernel-provider/rocke/platform/python \
python -m pytest dnn-providers/hip-kernel-provider/rocke/library/tests/test_rocke_paths_benchmark.py -q
bash -n dnn-providers/hip-kernel-provider/rocke/library/benchmarks/gfx942/attention/prefill/rocke_vs_aiter/run_mi300x_attention_bench_updated.sh
```

Expected: CPU tests and shell syntax pass.

- [ ] **Step 3: Run the full real-gfx942 smoke and acceptance workload**

On the prepared MI300X environment:

```bash
cd dnn-providers/hip-kernel-provider/rocke/library/benchmarks/gfx942/attention/prefill/rocke_vs_aiter
./setup_mi300x_attention_bench.sh
WARMUP=10 REPEAT=50 ./run_mi300x_attention_bench_updated.sh
```

Acceptance evidence:

- every supported dense/unified rocKE row has `status=PASS` and `max_abs < 4e-2`;
- row 10 dense is `UNSUPPORTED` with the 32-bit extent reason;
- row 10 unified/AITER/CK results are retained when their own arms pass;
- every AITER row names `fmha_fwd_hd128_bf16_causal_rtz`;
- every CK row names the fixed `CK_EXPECTED_KERNEL`;
- `results.csv` and `benchmark_results.md` choose the smallest passing rocKE time for every row.

- [ ] **Step 4: Commit README and verified benchmark integration**

```bash
git add dnn-providers/hip-kernel-provider/rocke/library/benchmarks/gfx942/attention/prefill/rocke_vs_aiter/README.md
git commit -m "docs: explain four-way gfx942 attention benchmark"
```

- [ ] **Step 5: Prepare the branch for review**

Run the focused tests from Step 2 again after the README commit. Keep benchmark result artifacts outside the source tree unless the repository already has an approved tracked location. Open a pull request from `users/avirgoel/rocke/fair-fourway`; do not push or merge any shared branch.
