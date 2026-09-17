#!/usr/bin/env bash
set -Eeuo pipefail

# Exact three-way attention benchmark from this chat:
#   ROCKE dense attention vs AITER FMHA-v3 ASM vs CK Tile FMHA.
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
# ROCKE:
#   production dispatch via dense_request -> resolve_dense_spec -> run

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
    bash -lc "source '$AITER_ENV'; python - <<'PY'
import importlib.metadata as im, torch
print('torch=' + torch.__version__)
print('torch_hip=' + str(torch.version.hip))
print('triton=' + im.version('triton'))
print('gpu=' + torch.cuda.get_device_name(0))
print('arch=' + torch.cuda.get_device_properties(0).gcnArchName)
print('cus=' + str(torch.cuda.get_device_properties(0).multi_processor_count))
PY"
} | tee "$OUT/environment.txt"

log "1. ROCKE"
cat > "$OUT/rocke_runner.py" <<'PY'
import argparse
import csv
import os
import traceback
import torch

from builders.gfx942.attention.prefill.attention_dense_prefill import (
    dense_request,
    resolve_dense_spec,
    describe_dense_spec,
    run,
)
from kernels.gfx942.attention_dense import supports_attention_dense

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

warmup = int(os.environ['BENCH_WARMUP'])
iters = int(os.environ['BENCH_REPEAT'])
out_path = os.environ['OUT_TSV']

args = argparse.Namespace(
    persistent=None,
    num_persistent=None,
    persist_decode=None,
    block_n=None,
    waves_per_eu=None,
    interleave=None,
    lds_k_group_pad=None,
    sliding_window=None,
)

with open(out_path, 'w', newline='') as f:
    w = csv.writer(f, delimiter='\t')
    w.writerow(['id','B','S','Hq','Hkv','GQA','ms','tflops','status','kernel','reason'])

    for cid, B, S, Hq, Hkv in CONFIGS:
        print('\n' + '=' * 110)
        print(f'ROCKE CONFIG {cid}: B={B} S={S} Hq={Hq} Hkv={Hkv} GQA={Hq//Hkv}:1')
        print('=' * 110)

        ms = tf = None
        kernel = ''
        status = 'PASS'
        reason = ''

        try:
            req = dense_request(
                args,
                batch=B,
                seqlen_q=S,
                seqlen_kv=S,
                num_query_heads=Hq,
                num_kv_heads=Hkv,
                head_size=128,
                causal=True,
                dtype='bf16',
            )
            spec = resolve_dense_spec(req, {})
            kernel = describe_dense_spec(spec)

            # Check support before allocating the large Q/K/V/O tensors. This is
            # important for the B=64,S=8192 case that can hit the 32-bit extent limit.
            ok, why = supports_attention_dense(spec, arch='gfx942')
            if not ok:
                status = 'UNSUPPORTED'
                reason = str(why)
                print('UNSUPPORTED:', reason)
            else:
                print('kernel:', kernel)
                # Timing only. Numerical correctness is a separate validation pass.
                ms, tf, _ = run(
                    spec,
                    warmup=warmup,
                    iters=iters,
                    check=False,
                    overrides={},
                )
        except Exception as e:
            text = f'{type(e).__name__}: {e}'
            if 'unsupported' in text.lower() or '32-bit' in text.lower() or 'extent' in text.lower():
                status = 'UNSUPPORTED'
            else:
                status = 'ERROR'
                traceback.print_exc()
            reason = text
            print(status + ':', reason)
        finally:
            torch.cuda.empty_cache()

        w.writerow([
            cid, B, S, Hq, Hkv, f'{Hq//Hkv}:1',
            '' if ms is None else f'{ms:.9f}',
            '' if tf is None else f'{tf:.9f}',
            status, kernel, reason,
        ])
        f.flush()

print('wrote', out_path)
PY

BENCH_WARMUP="$WARMUP" \
BENCH_REPEAT="$REPEAT" \
OUT_TSV="$OUT/rocke.tsv" \
bash -lc "source '$ROCKE_ENV'; cd '$ROCM_LIBS/dnn-providers/hip-kernel-provider'; python '$OUT/rocke_runner.py'" \
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
        '-timer=gpu',
        '-kname=1',
        '-v=0',
    ]

with open(OUT, 'w', newline='') as f:
    w = csv.writer(f, delimiter='\t')
    w.writerow(['id','B','S','Hq','Hkv','GQA','ms','tflops','gbps','status','kernel','reason'])

    for cid, B, S, Hq, Hkv in CONFIGS:
        print('\n' + '=' * 110)
        print(f'AITER ASM CONFIG {cid}: B={B} S={S} Hq={Hq} Hkv={Hkv} GQA={Hq//Hkv}:1')
        print('=' * 110)

        # -is_v3_check prints a synthetic 1.000-ms line. Keep this output separate
        # and NEVER parse it as benchmark performance.
        support_cmd = base_cmd(B,S,Hq,Hkv) + ['-warmup=0','-repeat=1','-is_v3_check=1']
        support = subprocess.run(
            support_cmd, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, env=os.environ.copy()
        )

        bench_cmd = base_cmd(B,S,Hq,Hkv) + [f'-warmup={WARMUP}', f'-repeat={REPEAT}']
        proc = subprocess.run(
            bench_cmd, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, env=os.environ.copy()
        )
        text = proc.stdout
        print(text, end='')

        with open(os.path.join(LOG_DIR, f'config_{cid:02d}.support.log'), 'w') as lf:
            lf.write('COMMAND: ' + ' '.join(support_cmd) + '\n\n' + support.stdout)
        with open(os.path.join(LOG_DIR, f'config_{cid:02d}.log'), 'w') as lf:
            lf.write('COMMAND: ' + ' '.join(bench_cmd) + '\n\n' + text)

        perf = perf_re.findall(text)
        km = load_re.findall(text)
        kernel = km[-1] if km else ''

        if support.returncode != 0:
            status, reason = 'UNSUPPORTED', f'ASM support check exit={support.returncode}'
            ms = tf = gb = ''
        elif proc.returncode != 0:
            status, reason = 'ERROR', f'benchmark exit={proc.returncode}'
            ms = tf = gb = ''
        elif not perf:
            status, reason = 'ERROR', 'could not parse benchmark timing line'
            ms = tf = gb = ''
        elif expected_kernel_token not in kernel:
            status, reason = 'ERROR', f'expected ASM causal RTZ kernel, loaded: {kernel or "<none>"}'
            ms = tf = gb = ''
        else:
            ms, tf, gb = perf[-1]
            status, reason = 'PASS', ''

        w.writerow([cid,B,S,Hq,Hkv,f'{Hq//Hkv}:1',ms,tf,gb,status,kernel,reason])
        f.flush()

print('wrote', OUT)
PY

AITER_EXE="$AITER_EXE" \
OUT_TSV="$OUT/aiter.tsv" \
LOG_DIR="$OUT/logs/aiter" \
BENCH_WARMUP="$WARMUP" \
BENCH_REPEAT="$REPEAT" \
bash -lc "source '$AITER_ENV'; cd '$AITER/op_tests/cpp/mha'; python '$OUT/aiter_runner.py'" \
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


def cmd_for(B, S, Hq, Hkv):
    return [
        EXE,
        '-v=0',
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
        f'-warmup={WARMUP}',
        f'-repeat={REPEAT}',
    ]

with open(OUT, 'w', newline='') as f:
    w = csv.writer(f, delimiter='\t')
    w.writerow(['id','B','S','Hq','Hkv','GQA','ms','tflops','gbps','status','kernel','reason'])

    for cid, B, S, Hq, Hkv in CONFIGS:
        cmd = cmd_for(B,S,Hq,Hkv)
        print('\n' + '=' * 110)
        print(f'CK CONFIG {cid}: B={B} S={S} Hq={Hq} Hkv={Hkv} GQA={Hq//Hkv}:1')
        print('COMMAND:', ' '.join(cmd))
        print('=' * 110)

        proc = subprocess.run(
            cmd, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, env=os.environ.copy()
        )
        text = proc.stdout
        print(text, end='')

        with open(os.path.join(LOG_DIR, f'config_{cid:02d}.log'), 'w') as lf:
            lf.write('COMMAND: ' + ' '.join(cmd) + '\n\n' + text)

        perf = perf_re.findall(text)
        kernels = kernel_re.findall(text)
        kernel = kernels[-1] if kernels else ''

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
            status, reason = 'PASS', ''

        w.writerow([cid,B,S,Hq,Hkv,f'{Hq//Hkv}:1',ms,tf,gb,status,kernel,reason])
        f.flush()

print('wrote', OUT)
PY

CK_EXE="$CK_EXE" \
OUT_TSV="$OUT/ck.tsv" \
LOG_DIR="$OUT/logs/ck" \
BENCH_WARMUP="$WARMUP" \
BENCH_REPEAT="$REPEAT" \
CK_EXPECTED_KERNEL="$CK_EXPECTED_KERNEL" \
bash -lc "source '$CK_ENV'; python3 '$OUT/ck_runner.py'" \
    2>&1 | tee "$OUT/logs/ck/all.log"

log "4. GENERATE THREE-WAY TABLE"
OUT="$OUT" WARMUP="$WARMUP" REPEAT="$REPEAT" python3 - <<'PY'
import csv
import math
import os
from pathlib import Path

out = Path(os.environ['OUT'])


def read_tsv(path):
    with open(path, newline='') as f:
        return {int(r['id']): r for r in csv.DictReader(f, delimiter='\t')}


def val(r, key):
    try:
        return float(r[key]) if r and r.get(key, '') else None
    except ValueError:
        return None


def fmt(x, n=4):
    return '—' if x is None else f'{x:.{n}f}'


def ratio(base_ms, faster_ms):
    return None if base_ms is None or faster_ms is None or faster_ms == 0 else base_ms / faster_ms


def sfmt(x):
    return '—' if x is None else f'{x:.2f}×'


def geomean(vals):
    vals = [x for x in vals if x is not None and x > 0]
    return math.exp(sum(math.log(x) for x in vals) / len(vals)) if vals else None


def status_display(row, ms):
    if row['status'] == 'PASS':
        return fmt(ms)
    if row['status'] == 'UNSUPPORTED':
        return 'unsupported'
    return row['status'].lower()


rocke = read_tsv(out / 'rocke.tsv')
ck = read_tsv(out / 'ck.tsv')
aiter = read_tsv(out / 'aiter.tsv')
rows = []

for cid in range(1, 11):
    r, c, a = rocke[cid], ck[cid], aiter[cid]
    B, S, Hq, Hkv = map(int, [r['B'], r['S'], r['Hq'], r['Hkv']])

    rms = val(r,'ms') if r['status'] == 'PASS' else None
    cms = val(c,'ms') if c['status'] == 'PASS' else None
    ams = val(a,'ms') if a['status'] == 'PASS' else None

    rows.append({
        'id': cid,
        'B': B,
        'S': S,
        'Hq': Hq,
        'Hkv': Hkv,
        'GQA': f'{Hq//Hkv}:1',
        'ROCKE_ms': rms,
        'CK_ms': cms,
        'AITER_ms': ams,
        'CK_vs_ROCKE': ratio(rms, cms),
        'AITER_vs_ROCKE': ratio(rms, ams),
        'AITER_vs_CK': ratio(cms, ams),
        'ROCKE_TFLOPS': val(r,'tflops') if r['status']=='PASS' else None,
        'CK_TFLOPS': val(c,'tflops') if c['status']=='PASS' else None,
        'AITER_TFLOPS': val(a,'tflops') if a['status']=='PASS' else None,
        'CK_GBps': val(c,'gbps') if c['status']=='PASS' else None,
        'AITER_GBps': val(a,'gbps') if a['status']=='PASS' else None,
        'ROCKE_status': r['status'],
        'CK_status': c['status'],
        'AITER_status': a['status'],
        'ROCKE_kernel': r.get('kernel',''),
        'CK_kernel': c.get('kernel',''),
        'AITER_kernel': a.get('kernel',''),
        'ROCKE_reason': r.get('reason',''),
        'CK_reason': c.get('reason',''),
        'AITER_reason': a.get('reason',''),
    })

with open(out / 'results.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

ck_r_vals = [r['CK_vs_ROCKE'] for r in rows]
a_r_vals = [r['AITER_vs_ROCKE'] for r in rows]
a_c_vals = [r['AITER_vs_CK'] for r in rows]
ck_r_gm = geomean(ck_r_vals)
a_r_gm = geomean(a_r_vals)
a_c_gm = geomean(a_c_vals)

lines = [
    '# MI300X/gfx942 BF16 causal attention benchmark',
    '',
    f'Common methodology: **BF16**, `D=128`, `Hq=32`, causal, `Sq=Sk`, BSHD, **{os.environ["WARMUP"]} warmups + {os.environ["REPEAT"]} measured launches**.',
    '',
    'Reported `ms` is the **average time for one kernel launch**, not the total for all measured launches.',
    '',
    'Implementations:',
    '- **ROCKE**: production dense-attention dispatch (`dense_request -> resolve_dense_spec -> run`).',
    '- **AITER**: `fwd_v3=1`, gfx942 BF16 **RTZ** (`v3_bf16_cvt=2`) ASM; the runner checks for `fmha_fwd_hd128_bf16_causal_rtz`.',
    '- **CK**: standalone `tile_example_fmha_fwd`, BF16, causal, BSHD, row-major V, `num_splits=1`; setup compiles one exact CK kernel instance and this runner verifies that instance is selected.',
    '',
    '| # | B | S | Hq | Hkv | GQA | ROCKE ms | CK ms | AITER ASM ms | CK vs ROCKE | AITER vs ROCKE | AITER vs CK |',
    '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
]

for r in rows:
    rr, cc, aa = rocke[r['id']], ck[r['id']], aiter[r['id']]
    lines.append(
        f"| {r['id']} | {r['B']} | {r['S']} | {r['Hq']} | {r['Hkv']} | {r['GQA']} | "
        f"{status_display(rr, r['ROCKE_ms'])} | {status_display(cc, r['CK_ms'])} | {status_display(aa, r['AITER_ms'])} | "
        f"{sfmt(r['CK_vs_ROCKE'])} | {sfmt(r['AITER_vs_ROCKE'])} | {sfmt(r['AITER_vs_CK'])} |"
    )

lines += [
    '',
    '## Throughput',
    '',
    '| # | ROCKE TFLOPS | CK TFLOPS | AITER TFLOPS | CK GB/s | AITER GB/s |',
    '|---:|---:|---:|---:|---:|---:|',
]
for r in rows:
    lines.append(
        f"| {r['id']} | {fmt(r['ROCKE_TFLOPS'],1)} | {fmt(r['CK_TFLOPS'],2)} | "
        f"{fmt(r['AITER_TFLOPS'],2)} | {fmt(r['CK_GBps'],2)} | {fmt(r['AITER_GBps'],2)} |"
    )

lines += ['', '## Summary', '']
if ck_r_gm is not None:
    n = sum(x is not None and x > 0 for x in ck_r_vals)
    lines.append(f'- Geometric-mean CK latency speedup over ROCKE across {n} common supported configs: **{ck_r_gm:.3f}×**.')
if a_r_gm is not None:
    n = sum(x is not None and x > 0 for x in a_r_vals)
    lines.append(f'- Geometric-mean AITER ASM latency speedup over ROCKE across {n} common supported configs: **{a_r_gm:.3f}×**.')
if a_c_gm is not None:
    n = sum(x is not None and x > 0 for x in a_c_vals)
    lines.append(f'- Geometric-mean AITER ASM latency speedup over CK across {n} common supported configs: **{a_c_gm:.3f}×**.')

bad = [r for r in rows if r['ROCKE_status'] != 'PASS' or r['CK_status'] != 'PASS' or r['AITER_status'] != 'PASS']
if bad:
    lines += ['', '## Unsupported / errors', '']
    for r in bad:
        if r['ROCKE_status'] != 'PASS':
            lines.append(f"- Config {r['id']} ROCKE: `{r['ROCKE_status']}` — {r['ROCKE_reason'] or 'no reason recorded'}")
        if r['CK_status'] != 'PASS':
            lines.append(f"- Config {r['id']} CK: `{r['CK_status']}` — {r['CK_reason'] or 'no reason recorded'}")
        if r['AITER_status'] != 'PASS':
            lines.append(f"- Config {r['id']} AITER: `{r['AITER_status']}` — {r['AITER_reason'] or 'no reason recorded'}")

lines += [
    '',
    '## Reproduction artifacts',
    '',
    '- `environment.txt` — GPU, ROCm/Torch/Triton, git commits, CK compiler, warmup/repeat',
    '- `rocke.tsv` — raw parsed ROCKE measurements',
    '- `ck.tsv` — raw parsed CK measurements and selected kernel names',
    '- `aiter.tsv` — raw parsed AITER ASM measurements and loaded kernel names',
    '- `results.csv` — merged machine-readable comparison',
    '- `logs/rocke/all.log` — complete ROCKE output',
    '- `logs/ck/config_*.log` — exact CK commands and raw output',
    '- `logs/aiter/config_*.log` — exact AITER benchmark commands and raw output',
    '- `logs/aiter/config_*.support.log` — separate AITER ASM support checks',
    '',
    'Note: the GB/s values printed by the CK/AITER benchmark programs are benchmark-derived traffic metrics; they are not rocprofiler-measured HBM traffic.',
]

(out / 'benchmark_results.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
PY

log "DONE"
echo "Results: $OUT"
echo "Table:   $OUT/benchmark_results.md"
echo "CSV:     $OUT/results.csv"
