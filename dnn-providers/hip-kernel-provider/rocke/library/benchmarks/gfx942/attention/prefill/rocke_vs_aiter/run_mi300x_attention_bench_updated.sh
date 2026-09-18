#!/usr/bin/env bash
set -Eeuo pipefail

# Exact four-way attention benchmark:
#   rocKE dense and auto-unified attention vs AITER FMHA-v3 ASM vs CK Tile FMHA.
#
# Common workload semantics:
#   BF16, D=128, Hq=32
#   causal, Sq=Sk, BSHD
#   warmup=10, measured runs=50 by default
#   reported ms = average latency of ONE kernel launch
#
# AITER:
#   FMHA-v3 gfx942 ASM
#   BF16 conversion RTZ (-v3_bf16_cvt=2)
#   fwd_v3=1; script rejects a run if the expected ASM kernel is not loaded.
#
# CK:
#   standalone CK Tile tile_example_fmha_fwd
#   num_splits=1, BF16, causal, BSHD, row-major V
#
#   checked-in shared-fixture dense and auto-unified runner

command -v git >/dev/null 2>&1 || { echo "ERROR: git not found" >&2; exit 1; }

# Resolve the already-cloned rocm-libraries checkout from this script's own
# location. No repo cloning is done by the benchmark script.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROCM_LIBS="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
[ -n "$ROCM_LIBS" ] || { echo "ERROR: this script must be run from inside a cloned rocm-libraries git repo" >&2; exit 1; }
[ "$(basename "$ROCM_LIBS")" = "rocm-libraries" ] || { echo "ERROR: git root is not rocm-libraries: $ROCM_LIBS" >&2; exit 1; }

ROOT="${GPU_BENCH_ROOT:-$(cd "$ROCM_LIBS/.." && pwd)}"
AITER="$ROOT/aiter"
CK="$AITER/3rdparty/composable_kernel"
ENV_DIR="$ROOT/env"
ROCKE_ENV="$ENV_DIR/rocke_env.sh"
AITER_ENV="$ENV_DIR/aiter_env.sh"
CK_ENV="$ENV_DIR/ck_env.sh"

cd "$SCRIPT_DIR"

WARMUP="${WARMUP:-10}"
REPEAT="${REPEAT:-50}"
D=128

# The setup script intentionally compiles only this CK Tile FMHA instance.
CK_EXPECTED_KERNEL="fmha_fwd_d128_bf16_batch_b128x128x32x128x32x128_r4x1x1_r4x1x1_w32x32x16_w32x32x16_qr_async_vr_psddv_nlogits_nbias_mask_nlse_ndropout_nskip_nqscale_ntrload_nsink"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT_DIR:-$ROOT/results/attention_$STAMP}"
mkdir -p "$OUT/logs/rocke" "$OUT/logs/aiter" "$OUT/logs/ck"

log() { printf '\n============================================================\n%s\n============================================================\n' "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }
trap 'echo "FAILED at line $LINENO: $BASH_COMMAND" >&2' ERR

[ -f "$ROCKE_ENV" ] || die "missing $ROCKE_ENV; run setup_mi300x_attention_bench.sh first"
[ -f "$AITER_ENV" ] || die "missing $AITER_ENV; run setup_mi300x_attention_bench.sh first"
[ -f "$CK_ENV" ] || die "missing $CK_ENV; run setup_mi300x_attention_bench.sh first"
[ -d "$ROCM_LIBS/.git" ] || die "missing $ROCM_LIBS"
[ -d "$AITER/.git" ] || die "missing $AITER"
[ -e "$CK/.git" ] || die "missing CK submodule: $CK"

cat > "$OUT/configs.tsv" <<'CFG'
id	B	S	Hq	Hkv
1	1	4096	32	8
2	1	4096	32	16
3	1	8192	32	8
4	1	8192	32	16
5	1	16384	32	8
6	16	4096	32	8
7	16	8192	32	8
8	16	4096	32	16
9	64	4096	32	8
10	64	8192	32	8
CFG

log "0. RECORD ENVIRONMENT"
{
    echo "run_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host=$(hostname)"
    echo "warmup=$WARMUP"
    echo "measured_runs=$REPEAT"
    echo "reported_time=average_per_launch"
    echo "dtype=bf16"
    echo "D=128"
    echo "Hq=32"
    echo "causal=true"
    echo "Sq=Sk"
    echo "layout=BSHD"
    echo "aiter_path=FMHA-v3 ASM"
    echo "aiter_bf16_conversion=RTZ"
    echo "ck_path=CK Tile FMHA"
    echo "ck_num_splits=1"
    echo "ck_expected_kernel=$CK_EXPECTED_KERNEL"
    echo "rocm_libraries_commit=$(git -C "$ROCM_LIBS" rev-parse HEAD)"
    echo "rocm_libraries_dirty=$(test -n "$(git -C "$ROCM_LIBS" status --porcelain)" && echo yes || echo no)"
    echo "aiter_commit=$(git -C "$AITER" rev-parse HEAD)"
    echo "aiter_dirty=$(test -n "$(git -C "$AITER" status --porcelain)" && echo yes || echo no)"
    echo "ck_commit=$(git -C "$CK" rev-parse HEAD 2>/dev/null || echo unavailable)"
    echo "system_rocm_version=$(cat /opt/rocm/.info/version 2>/dev/null || echo unavailable)"
    echo "ck_compiler=$(/opt/rocm/llvm/bin/clang++ --version 2>/dev/null | head -n 1 || echo unavailable)"
    AITER_ENV_PATH="$AITER_ENV" bash -c 'source "$AITER_ENV_PATH"; exec python -' <<'PY'
import importlib.metadata as im, torch
print('torch=' + torch.__version__)
print('torch_hip=' + str(torch.version.hip))
print('triton=' + im.version('triton'))
print('gpu=' + torch.cuda.get_device_name(0))
print('arch=' + torch.cuda.get_device_properties(0).gcnArchName)
print('cus=' + str(torch.cuda.get_device_properties(0).multi_processor_count))
PY
} | tee "$OUT/environment.txt"

log "1. ROCKE DENSE AND AUTO-UNIFIED"
ROCKE_ENV_PATH="$ROCKE_ENV" \
ROCKE_WORKDIR="$ROCM_LIBS/dnn-providers/hip-kernel-provider" \
ROCKE_RUNNER="$SCRIPT_DIR/rocke_paths.py" \
ROCKE_OUT_DENSE="$OUT/rocke_dense.tsv" \
ROCKE_OUT_UNIFIED="$OUT/rocke_unified.tsv" \
ROCKE_WARMUP="$WARMUP" \
ROCKE_REPEAT="$REPEAT" \
bash -c 'source "$ROCKE_ENV_PATH"; cd "$ROCKE_WORKDIR"; exec python "$ROCKE_RUNNER" --out-dense "$ROCKE_OUT_DENSE" --out-unified "$ROCKE_OUT_UNIFIED" --warmup "$ROCKE_WARMUP" --iters "$ROCKE_REPEAT"' \
    2>&1 | tee "$OUT/logs/rocke/all.log"

log "2. AITER FMHA-v3 ASM"
AITER_EXE="$AITER/op_tests/cpp/mha/fwd.exe"
[ -x "$AITER_EXE" ] || die "missing $AITER_EXE; rerun setup"

cat > "$OUT/aiter_runner.py" <<'PY'
import csv
import os
import re
import subprocess

EXE = os.environ['AITER_EXE']
OUT = os.environ['OUT_TSV']
LOG_DIR = os.environ['LOG_DIR']
WARMUP = int(os.environ['BENCH_WARMUP'])
REPEAT = int(os.environ['BENCH_REPEAT'])

CONFIGS = [
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

num = r'[0-9]+(?:\.[0-9]+)?'
perf_re = re.compile(rf'({num})\s*ms,\s*({num})\s*TFlops,\s*({num})\s*GB/s')
load_re = re.compile(r'LoadKernel:\s*(\S+)')
expected_kernel_token = 'fmha_fwd_hd128_bf16_causal_rtz'


def base_cmd(B, S, Hq, Hkv):
    return [
        EXE,
        '-prec=bf16',
        f'-b={B}',
        f'-h={Hq}',
        f'-h_k={Hkv}',
        '-d=128',
        '-d_v=128',
        f'-s={S}',
        f'-s_k={S}',
        '-iperm=0',             # BSHD
        '-operm=0',             # BSHD
        '-mask=1',              # causal; Sq==Sk so top-left == bottom-right
        '-lse=0',
        '-fwd_v3=1',            # force v3 ASM path
        '-v3_bf16_cvt=2',       # RTZ on gfx942
        '-mode=0',
        '-kname=1',
    ]

with open(OUT, 'w', newline='') as f:
    w = csv.writer(f, delimiter='\t')
    w.writerow([
        'id', 'B', 'S', 'Hq', 'Hkv', 'GQA', 'ms', 'tflops', 'gbps', 'status',
        'validation_status', 'kernel', 'reason',
    ])

    for cid, B, S, Hq, Hkv in CONFIGS:
        print('\n' + '=' * 110)
        print(f'AITER ASM CONFIG {cid}: B={B} S={S} Hq={Hq} Hkv={Hkv} GQA={Hq//Hkv}:1')
        print('=' * 110)

        # -is_v3_check prints a synthetic 1.000-ms line. Keep this output separate
        # and NEVER parse it as benchmark performance.
        support_cmd = base_cmd(B,S,Hq,Hkv) + [
            '-v=0', '-warmup=0', '-repeat=1', '-is_v3_check=1',
        ]
        support = subprocess.run(
            support_cmd, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, env=os.environ.copy()
        )
        bench_cmd = base_cmd(B,S,Hq,Hkv) + [
            '-timer=gpu', '-v=0', f'-warmup={WARMUP}', f'-repeat={REPEAT}',
        ]
        proc = None
        text = ''
        if support.returncode == 0:
            proc = subprocess.run(
                bench_cmd, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, env=os.environ.copy()
            )
            text = proc.stdout
            print(text, end='')

        with open(os.path.join(LOG_DIR, f'config_{cid:02d}.support.log'), 'w') as lf:
            lf.write('COMMAND: ' + ' '.join(support_cmd) + '\n\n' + support.stdout)
        if proc is not None:
            with open(os.path.join(LOG_DIR, f'config_{cid:02d}.log'), 'w') as lf:
                lf.write('COMMAND: ' + ' '.join(bench_cmd) + '\n\n' + text)

        perf = perf_re.findall(text)
        km = load_re.findall(text)
        kernel = km[-1] if km else ''
        validation_status = 'UNAVAILABLE'

        if support.returncode != 0:
            status, reason = 'UNSUPPORTED', f'ASM support check exit={support.returncode}'
            ms = tf = gb = ''
        elif proc is None or proc.returncode != 0:
            status, reason = 'ERROR', f'benchmark exit={proc.returncode if proc else "not run"}'
            ms = tf = gb = ''
        elif not perf:
            status, reason = 'ERROR', 'could not parse benchmark timing line'
            ms = tf = gb = ''
        elif expected_kernel_token not in kernel:
            status, reason = 'ERROR', f'expected ASM causal RTZ kernel, loaded: {kernel or "<none>"}'
            ms = tf = gb = ''
        else:
            ms, tf, gb = perf[-1]
            status = 'PASS'
            reason = 'numeric validation unavailable: native BSHD GPU validator is known invalid for this workload'

        w.writerow([
            cid, B, S, Hq, Hkv, f'{Hq//Hkv}:1', ms, tf, gb, status,
            validation_status, kernel, reason,
        ])
        f.flush()

print('wrote', OUT)
PY

AITER_ENV_PATH="$AITER_ENV" \
AITER_WORKDIR="$AITER/op_tests/cpp/mha" \
AITER_RUNNER="$OUT/aiter_runner.py" \
AITER_EXE="$AITER_EXE" \
OUT_TSV="$OUT/aiter.tsv" \
LOG_DIR="$OUT/logs/aiter" \
BENCH_WARMUP="$WARMUP" \
BENCH_REPEAT="$REPEAT" \
bash -c 'source "$AITER_ENV_PATH"; cd "$AITER_WORKDIR"; exec python "$AITER_RUNNER"' \
    2>&1 | tee "$OUT/logs/aiter/all.log"

log "3. COMPOSABLE KERNEL CK TILE FMHA"
CK_EXE="$CK/build/bin/tile_example_fmha_fwd"
[ -x "$CK_EXE" ] || die "missing $CK_EXE; rerun setup"

cat > "$OUT/ck_runner.py" <<'PY'
import csv
import os
import re
import subprocess

EXE = os.environ['CK_EXE']
OUT = os.environ['OUT_TSV']
LOG_DIR = os.environ['LOG_DIR']
WARMUP = int(os.environ['BENCH_WARMUP'])
REPEAT = int(os.environ['BENCH_REPEAT'])
EXPECTED_KERNEL = os.environ['CK_EXPECTED_KERNEL']

CONFIGS = [
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

num = r'[0-9]+(?:\.[0-9]+)?'
perf_re = re.compile(rf'({num})\s*ms,\s*({num})\s*TFlops,\s*({num})\s*GB/s')
kernel_re = re.compile(r'\b(fmha_fwd_[^,\s]+)')


def base_cmd(B, S, Hq, Hkv):
    return [
        EXE,
        '-mode=0',
        f'-b={B}',
        f'-h={Hq}',
        f'-h_k={Hkv}',
        f'-s={S}',
        f'-s_k={S}',
        '-d=128',
        '-d_v=128',
        '-scale_s=0',            # 1/sqrt(D)
        '-iperm=0',              # BSHD
        '-operm=0',              # BSHD
        '-bias=n',
        '-prec=bf16',
        '-mask=1',               # top-left causal; equivalent here because Sq==Sk
        '-vlayout=r',            # row-major V
        '-lse=0',
        '-kname=1',
        '-num_splits=1',         # do not let a heuristic alter the algorithm
    ]

with open(OUT, 'w', newline='') as f:
    w = csv.writer(f, delimiter='\t')
    w.writerow([
        'id', 'B', 'S', 'Hq', 'Hkv', 'GQA', 'ms', 'tflops', 'gbps', 'status',
        'validation_status', 'kernel', 'reason',
    ])

    for cid, B, S, Hq, Hkv in CONFIGS:
        bench_cmd = base_cmd(B,S,Hq,Hkv) + [
            '-timer=gpu', '-v=0', f'-warmup={WARMUP}', f'-repeat={REPEAT}',
        ]
        print('\n' + '=' * 110)
        print(f'CK CONFIG {cid}: B={B} S={S} Hq={Hq} Hkv={Hkv} GQA={Hq//Hkv}:1')
        print('COMMAND:', ' '.join(bench_cmd))
        print('=' * 110)

        proc = subprocess.run(
            bench_cmd, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, env=os.environ.copy()
        )
        text = proc.stdout
        print(text, end='')

        with open(os.path.join(LOG_DIR, f'config_{cid:02d}.log'), 'w') as lf:
            lf.write('COMMAND: ' + ' '.join(bench_cmd) + '\n\n' + text)

        perf = perf_re.findall(text)
        kernels = kernel_re.findall(text)
        kernel = kernels[-1] if kernels else ''
        validation_status = 'UNAVAILABLE'

        if proc.returncode != 0:
            status, reason = 'ERROR', f'benchmark exit={proc.returncode}'
            ms = tf = gb = ''
        elif not perf:
            status, reason = 'ERROR', 'could not parse benchmark timing line'
            ms = tf = gb = ''
        elif not kernel:
            status, reason = 'ERROR', 'timing parsed but selected CK kernel name was not found'
            ms = tf = gb = ''
        elif kernel != EXPECTED_KERNEL:
            status, reason = 'ERROR', f'expected CK kernel {EXPECTED_KERNEL}, selected: {kernel}'
            ms = tf = gb = ''
        else:
            ms, tf, gb = perf[-1]
            status = 'PASS'
            reason = 'numeric validation unavailable: native BSHD GPU validator is known invalid for this workload'

        w.writerow([
            cid, B, S, Hq, Hkv, f'{Hq//Hkv}:1', ms, tf, gb, status,
            validation_status, kernel, reason,
        ])
        f.flush()

print('wrote', OUT)
PY

CK_ENV_PATH="$CK_ENV" \
CK_RUNNER="$OUT/ck_runner.py" \
CK_EXE="$CK_EXE" \
OUT_TSV="$OUT/ck.tsv" \
LOG_DIR="$OUT/logs/ck" \
BENCH_WARMUP="$WARMUP" \
BENCH_REPEAT="$REPEAT" \
CK_EXPECTED_KERNEL="$CK_EXPECTED_KERNEL" \
bash -c 'source "$CK_ENV_PATH"; exec python3 "$CK_RUNNER"' \
    2>&1 | tee "$OUT/logs/ck/all.log"

log "4. GENERATE FOUR-WAY TABLE"
python3 "$SCRIPT_DIR/merge_results.py" "$OUT"

log "DONE"
echo "Results: $OUT"
echo "Table:   $OUT/benchmark_results.md"
echo "CSV:     $OUT/results.csv"
