#!/usr/bin/env bash
set -Eeuo pipefail

# MI300X/gfx942 setup for the exact attention benchmark from this chat:
#   1) ROCKE dense attention
#   2) AITER FMHA-v3 ASM (gfx942 BF16 RTZ)
#   3) Composable Kernel (CK Tile) FMHA
#
# Working target stack from the chat:
#   GPU          : AMD Instinct MI300X VF (gfx942)
#   ROCm wheels  : 10.0.0
#   torch        : 2.13.0+rocm10.0.0
#   HIP          : 7.15.26333
#   Triton       : 3.8.0+git4cff872c.rocm10.0.0
#   Python       : 3.12
#
# Fixes intentionally preserved from the debugging session:
#   * ROCKE uses wheel libamd_comgr/libamdhip64 to avoid mixed-LLVM failures.
#   * AITER runtime requirements are installed (including PyYAML).
#   * AITER is prevented from replacing ROCm-10 Triton with its ROCm-7.2 fallback.
#   * AITER fwd_v3 is built explicitly and linked with the wheel SDK hipcc.
#   * CK is built as the standalone CK Tile example from AITER's pinned CK submodule.

command -v git >/dev/null 2>&1 || { echo "ERROR: git not found" >&2; exit 1; }

# This script is intended to live inside the already-cloned rocm-libraries repo,
# under the benchmark directory. Resolve the repo from the script location so the
# engineer can clone the repo once and run this script in-place.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROCM_LIBS="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
[ -n "$ROCM_LIBS" ] || { echo "ERROR: this script must be run from inside a cloned rocm-libraries git repo" >&2; exit 1; }
[ "$(basename "$ROCM_LIBS")" = "rocm-libraries" ] || { echo "ERROR: git root is not rocm-libraries: $ROCM_LIBS" >&2; exit 1; }

ROOT="${GPU_BENCH_ROOT:-$(cd "$ROCM_LIBS/.." && pwd)}"
VROOT="${VENV_ROOT:-$HOME/venvs}"
PYTHON="${PYTHON:-python3.12}"
AMD_INDEX="${AMD_WHL_INDEX:-https://stable.repo.amd.com/rocm/whl-next/}"
ROCM_WHEEL_VERSION="${ROCM_WHEEL_VERSION:-10.0.0}"
TORCH_VERSION="${TORCH_VERSION:-2.13.0+rocm10.0.0}"
TRITON_VERSION="${TRITON_VERSION:-3.8.0+git4cff872c.rocm10.0.0}"
AITER_REF="${AITER_REF:-cdf6ee88a128c2c160b0512fe37cfb67be161b1d}"

AITER="$ROOT/aiter"
ROCKE_VENV="$VROOT/rocke"
AITER_VENV="$VROOT/aiter"
ENV_DIR="$ROOT/env"

AITER_URL="${AITER_URL:-https://github.com/ROCm/aiter.git}"

cd "$SCRIPT_DIR"

log() { printf '\n============================================================\n%s\n============================================================\n' "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }
trap 'echo "FAILED at line $LINENO: $BASH_COMMAND" >&2' ERR

mkdir -p "$ROOT" "$VROOT" "$ENV_DIR"

log "0. PRE-FLIGHT"
command -v "$PYTHON" >/dev/null 2>&1 || die "$PYTHON not found"
command -v git >/dev/null 2>&1 || die "git not found"

if command -v rocminfo >/dev/null 2>&1; then
    rocminfo 2>/dev/null | grep -m 6 -E 'Name:[[:space:]]+gfx|Marketing Name' || true
fi

if ! command -v cmake >/dev/null 2>&1 || ! command -v ninja >/dev/null 2>&1; then
    if command -v apt-get >/dev/null 2>&1 && [ "$(id -u)" -eq 0 ]; then
        apt-get update
        DEBIAN_FRONTEND=noninteractive apt-get install -y \
            git cmake ninja-build build-essential pkg-config \
            python3.12-venv python3.12-dev libnuma-dev
    else
        die "cmake/ninja missing; install build tools first"
    fi
fi

log "1. SOURCE CHECKOUTS"
# rocm-libraries is deliberately NOT cloned here. This script itself is stored
# in that checkout, so cloning the repo is the prerequisite reproduction step.
[ -d "$ROCM_LIBS/.git" ] || die "rocm-libraries checkout not found: $ROCM_LIBS"
echo "Using existing rocm-libraries checkout: $ROCM_LIBS"

if [ -d "$AITER/.git" ]; then
    echo "Keeping existing AITER checkout: $AITER"
    cur="$(git -C "$AITER" rev-parse HEAD)"
    if [ "$cur" != "$AITER_REF" ]; then
        echo "WARNING: AITER is at $cur"
        echo "         benchmarked chat commit was $AITER_REF"
        echo "         preserving the existing checkout; metadata will record it"
    fi
else
    git clone "$AITER_URL" "$AITER"
    git -C "$AITER" checkout "$AITER_REF"
fi

git -C "$AITER" submodule update --init --recursive

ROCKE_ROOT="$ROCM_LIBS/dnn-providers/hip-kernel-provider/rocke"
[ -d "$ROCKE_ROOT/platform" ] || die "missing $ROCKE_ROOT/platform"
[ -d "$ROCKE_ROOT/library" ] || die "missing $ROCKE_ROOT/library"

CK="$AITER/3rdparty/composable_kernel"
[ -e "$CK/.git" ] || die "AITER CK submodule is missing: $CK"

create_rocm10_venv() {
    local venv="$1"
    if [ ! -x "$venv/bin/python" ]; then
        "$PYTHON" -m venv "$venv"
    fi
    "$venv/bin/python" -m pip install -U pip setuptools wheel packaging ninja psutil
    "$venv/bin/python" -m pip install \
        --index-url "$AMD_INDEX" \
        "rocm[devel,device-gfx942,libraries]==$ROCM_WHEEL_VERSION" \
        "torch[device-gfx942]==$TORCH_VERSION"
    "$venv/bin/rocm-sdk" init >/dev/null
}

log "2. ROCKE ENVIRONMENT"
create_rocm10_venv "$ROCKE_VENV"
"$ROCKE_VENV/bin/python" -m pip install -e "$ROCKE_ROOT/platform"
"$ROCKE_VENV/bin/python" -m pip install -e "$ROCKE_ROOT/library" --no-deps

cat > "$ENV_DIR/rocke_env.sh" <<EOF_ROCKE
# Generated by setup_mi300x_attention_bench.sh
source "$ROCKE_VENV/bin/activate"
rocm-sdk init >/dev/null
export ROCM_HOME="\$(rocm-sdk path --root)"
export ROCM_PATH="\$ROCM_HOME"
export HIP_PATH="\$ROCM_HOME"
export PATH="\$(rocm-sdk path --bin):\$PATH"
export HIP_PLATFORM=amd

# Exact ROCKE fixes from the working MI300X session.
export ROCKE_BACKEND=python
export ROCKE_LLVM_FLAVOR=llvm23

_ROCM_CORE_LIB="\$(python - <<'PY'
import importlib.util, pathlib
s = importlib.util.find_spec('_rocm_sdk_core')
if s is None or not s.submodule_search_locations:
    raise SystemExit('_rocm_sdk_core not found')
r = pathlib.Path(list(s.submodule_search_locations)[0])
print((r / 'lib').resolve())
PY
)"
export ROCKE_COMGR_LIB="\$_ROCM_CORE_LIB/libamd_comgr.so.3"
export ROCKE_HIP_LIB="\$_ROCM_CORE_LIB/libamdhip64.so.7"
export LD_LIBRARY_PATH="\$_ROCM_CORE_LIB:\${LD_LIBRARY_PATH:-}"
unset _ROCM_CORE_LIB
EOF_ROCKE

bash -lc "source '$ENV_DIR/rocke_env.sh'; test -f \"\$ROCKE_COMGR_LIB\"; test -f \"\$ROCKE_HIP_LIB\"; python - <<'PY'
import torch
print('torch      :', torch.__version__)
print('torch HIP  :', torch.version.hip)
print('GPU        :', torch.cuda.get_device_name(0))
print('arch       :', torch.cuda.get_device_properties(0).gcnArchName)
print('CUs        :', torch.cuda.get_device_properties(0).multi_processor_count)
PY"

log "3. AITER ENVIRONMENT"
create_rocm10_venv "$AITER_VENV"

# Fixes the earlier missing-yaml/runtime-dependency failure.
"$AITER_VENV/bin/python" -m pip install -r "$AITER/requirements.txt"

# Do not allow AITER to replace ROCm-10 Triton with its ROCm-7.2 fallback.
"$AITER_VENV/bin/python" -m pip uninstall -y triton-kernels triton_kernels >/dev/null 2>&1 || true
"$AITER_VENV/bin/python" -m pip install \
    --index-url "$AMD_INDEX" \
    --force-reinstall --no-deps \
    "triton==$TRITON_VERSION"

export AITER_USE_SYSTEM_TRITON=1
export PREBUILD_KERNELS=0
"$AITER_VENV/bin/python" -m pip install -e "$AITER"

# Hard guard: editable install must not leave the old ROCm-7.2 Triton behind.
"$AITER_VENV/bin/python" -m pip uninstall -y triton-kernels triton_kernels >/dev/null 2>&1 || true
"$AITER_VENV/bin/python" -m pip install \
    --index-url "$AMD_INDEX" \
    --force-reinstall --no-deps \
    "triton==$TRITON_VERSION"
"$AITER_VENV/bin/python" -m pip check

cat > "$ENV_DIR/aiter_env.sh" <<EOF_AITER
# Generated by setup_mi300x_attention_bench.sh
source "$AITER_VENV/bin/activate"
rocm-sdk init >/dev/null
export ROCM_HOME="\$(rocm-sdk path --root)"
export ROCM_PATH="\$ROCM_HOME"
export HIP_PATH="\$ROCM_HOME"
export PATH="\$(rocm-sdk path --bin):\$PATH"
export HIP_PLATFORM=amd
export AITER_USE_SYSTEM_TRITON=1
export PREBUILD_KERNELS=0
export AITER_ASM_DIR="$AITER/hsa/"
_ROCM_CORE_LIB="\$(python - <<'PY'
import importlib.util, pathlib
s = importlib.util.find_spec('_rocm_sdk_core')
if s is None or not s.submodule_search_locations:
    raise SystemExit('_rocm_sdk_core not found')
r = pathlib.Path(list(s.submodule_search_locations)[0])
print((r / 'lib').resolve())
PY
)"
export LD_LIBRARY_PATH="$AITER/op_tests/cpp/mha:\$_ROCM_CORE_LIB:\${LD_LIBRARY_PATH:-}"
unset _ROCM_CORE_LIB
EOF_AITER

bash -lc "source '$ENV_DIR/aiter_env.sh'; python - <<'PY'
import importlib.metadata as im
import torch, aiter
print('AITER      :', aiter.__file__)
print('torch      :', torch.__version__)
print('torch HIP  :', torch.version.hip)
print('triton     :', im.version('triton'))
print('GPU        :', torch.cuda.get_device_name(0))
print('arch       :', torch.cuda.get_device_properties(0).gcnArchName)
print('CUs        :', torch.cuda.get_device_properties(0).multi_processor_count)
assert im.version('triton') == '$TRITON_VERSION'
PY"

log "4. BUILD AITER FMHA-v3 ASM"
MHA="$AITER/op_tests/cpp/mha"

# Build ASM-only device library, then link the official benchmark with the
# wheel SDK hipcc. This avoids accidentally mixing a different compiler stack.
bash -lc "source '$ENV_DIR/aiter_env.sh'; cd '$MHA'; python compile.py --api=fwd_v3; \
HIPCC=\"\$(rocm-sdk path --bin)/hipcc\"; \
\"\$HIPCC\" \
  -I'$AITER/3rdparty/composable_kernel/include' \
  -I'$AITER/3rdparty/composable_kernel/example/ck_tile/01_fmha/' \
  -I'$AITER/csrc/include' \
  -std=c++20 -O3 \
  -DUSE_ROCM=1 -DENABLE_CK=1 -DCK_TILE_FMHA_FWD_SPLITKV_API=0 \
  --offload-arch=native \
  -L '$MHA' -lmha_fwd \
  '$MHA/benchmark_mha_fwd.cpp' \
  -Wl,-rpath,'$MHA' \
  -o '$MHA/fwd.exe'"

[ -f "$MHA/libmha_fwd.so" ] || die "AITER libmha_fwd.so missing"
[ -x "$MHA/fwd.exe" ] || die "AITER fwd.exe missing"

# AITER's -is_v3_check=1 prints a synthetic 1.000 ms line; this is ONLY a
# support check. The benchmark script never parses this line as performance.
bash -lc "source '$ENV_DIR/aiter_env.sh'; cd '$MHA'; ./fwd.exe \
  -prec=bf16 -b=1 -h=32 -h_k=8 -d=128 -d_v=128 \
  -s=4096 -s_k=4096 -iperm=0 -operm=0 -mask=1 -lse=0 \
  -fwd_v3=1 -v3_bf16_cvt=2 -mode=0 -kname=1 -v=0 -is_v3_check=1"

log "5. BUILD COMPOSABLE KERNEL TILE FMHA"

# Build only the ONE CK Tile FMHA instance used by this benchmark instead of
# CK's full generated FMHA catalog.
CK_KERNEL="fmha_fwd_d128_bf16_batch_b128x128x32x128x32x128_r4x1x1_r4x1x1_w32x32x16_w32x32x16_qr_async_vr_psddv_nlogits_nbias_mask_nlse_ndropout_nskip_nqscale_ntrload_nsink"

[ -x /opt/rocm/llvm/bin/clang++ ] || die "/opt/rocm/llvm/bin/clang++ missing; CK dev preset requires it"

cat > "$ENV_DIR/ck_env.sh" <<'EOF_CK'
# Generated by setup_mi300x_attention_bench.sh
export ROCM_HOME=/opt/rocm
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm
export HIP_PLATFORM=amd
export PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:${LD_LIBRARY_PATH:-}
EOF_CK

# CK supports --filter in its FMHA generator, but the CMake wrapper does not
# expose it as a cache option. Patch the pinned submodule locally, then restore
# the file automatically on success, failure, or Ctrl-C.
CK_CMAKE="$CK/example/ck_tile/01_fmha/CMakeLists.txt"
CK_CMAKE_BACKUP="$CK_CMAKE.gpu_bench_backup"

[ -f "$CK_CMAKE" ] || die "CK FMHA CMakeLists missing: $CK_CMAKE"

if [ -f "$CK_CMAKE_BACKUP" ]; then
    echo "Restoring CK CMakeLists from previous interrupted setup"
    mv -f "$CK_CMAKE_BACKUP" "$CK_CMAKE"
fi

cp "$CK_CMAKE" "$CK_CMAKE_BACKUP"

restore_ck_cmake() {
    if [ -f "$CK_CMAKE_BACKUP" ]; then
        mv -f "$CK_CMAKE_BACKUP" "$CK_CMAKE"
    fi
}
trap restore_ck_cmake EXIT

"$PYTHON" - "$CK_CMAKE" "$CK_KERNEL" <<'PY_CK_FILTER'
import sys
from pathlib import Path

path = Path(sys.argv[1])
kernel = sys.argv[2]
text = path.read_text()

needle = "  --api ${FMHA_FWD_APIS}\n"
replacement = needle + f"  --filter {kernel}\n"

if needle not in text:
    raise SystemExit("Could not locate FMHA_FWD_CODE_GEN_COMMON_ARGS in CK CMakeLists")

path.write_text(text.replace(needle, replacement, 1))
PY_CK_FILTER

CK_JOBS="${CK_JOBS:-$(nproc)}"

# Fresh graph: never reuse an older thousands-of-files CK Ninja build.
rm -rf "$CK/build"
mkdir -p "$CK/build"

bash -lc "source '$ENV_DIR/ck_env.sh'; cd '$CK/build'; \
  ../script/cmake-ck-dev.sh --minimal .. gfx942 -G Ninja \
    -DBUILD_CK_EXAMPLES=ON \
    -DBUILD_TESTING=OFF \
    -DFMHA_FWD_ENABLE_APIS=fwd"

# Verify that codegen produced exactly one kernel instance before compiling.
FWD_BLOB_LIST="$(find "$CK/build" -name fwd_blob_list.txt -type f -print -quit)"
[ -n "$FWD_BLOB_LIST" ] || die "CK fwd_blob_list.txt not generated"

CK_GENERATED_KERNELS="$(grep -vc 'fmha_fwd_api.cpp' "$FWD_BLOB_LIST" || true)"
echo "CK generated FMHA kernel instances: $CK_GENERATED_KERNELS"
cat "$FWD_BLOB_LIST"
[ "$CK_GENERATED_KERNELS" -eq 1 ] || die "Expected exactly 1 CK FMHA kernel instance, got $CK_GENERATED_KERNELS"

time bash -lc "source '$ENV_DIR/ck_env.sh'; cd '$CK/build'; \
  ninja -j'$CK_JOBS' tile_example_fmha_fwd"

restore_ck_cmake
trap - EXIT

CK_EXE="$CK/build/bin/tile_example_fmha_fwd"
[ -x "$CK_EXE" ] || die "CK tile_example_fmha_fwd missing: $CK_EXE"

# Smoke-test the exact BF16/causal/BSHD path and require this kernel.
CK_SMOKE="$(bash -lc "source '$ENV_DIR/ck_env.sh'; '$CK_EXE' \
  -v=0 -mode=0 -b=1 -h=32 -h_k=8 -s=4096 -s_k=4096 \
  -d=128 -d_v=128 -scale_s=0 -iperm=0 -operm=0 -bias=n \
  -prec=bf16 -mask=1 -vlayout=r -lse=0 -kname=1 -num_splits=1 \
  -warmup=1 -repeat=1")"
printf '%s\n' "$CK_SMOKE"
printf '%s\n' "$CK_SMOKE" | grep -F "$CK_KERNEL" >/dev/null || \
    die "CK smoke test did not select expected kernel: $CK_KERNEL"

log "6. WRITE METADATA"
META="$ROOT/setup_metadata.txt"
{
    echo "setup_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host=$(hostname)"
    echo "rocm_libraries_path=$ROCM_LIBS"
    echo "rocm_libraries_commit=$(git -C "$ROCM_LIBS" rev-parse HEAD)"
    echo "rocm_libraries_dirty=$(test -n "$(git -C "$ROCM_LIBS" status --porcelain)" && echo yes || echo no)"
    echo "aiter_path=$AITER"
    echo "aiter_commit=$(git -C "$AITER" rev-parse HEAD)"
    echo "aiter_dirty=$(test -n "$(git -C "$AITER" status --porcelain)" && echo yes || echo no)"
    echo "ck_path=$CK"
    echo "ck_commit=$(git -C "$CK" rev-parse HEAD 2>/dev/null || echo unavailable)"
    echo "system_rocm_version=$(cat /opt/rocm/.info/version 2>/dev/null || echo unavailable)"
    echo "ck_compiler=$(/opt/rocm/llvm/bin/clang++ --version 2>/dev/null | head -n 1 || echo unavailable)"
    echo "ck_cmake_preset=dev-minimal"
    echo "ck_build_jobs=$CK_JOBS"
    echo "ck_kernel_filter=$CK_KERNEL"
    bash -lc "source '$ENV_DIR/aiter_env.sh'; python - <<'PY'
import importlib.metadata as im, torch
print('torch=' + torch.__version__)
print('torch_hip=' + str(torch.version.hip))
print('triton=' + im.version('triton'))
print('gpu=' + torch.cuda.get_device_name(0))
print('arch=' + torch.cuda.get_device_properties(0).gcnArchName)
print('cus=' + str(torch.cuda.get_device_properties(0).multi_processor_count))
PY"
} | tee "$META"

cat <<EOF_DONE

SETUP COMPLETE

Implementations prepared:
  ROCKE dense attention
  AITER FMHA-v3 ASM (BF16 RTZ)
  Composable Kernel CK Tile FMHA

Environment helpers:
  $ENV_DIR/rocke_env.sh
  $ENV_DIR/aiter_env.sh
  $ENV_DIR/ck_env.sh

AITER ASM executable:
  $MHA/fwd.exe

CK executable:
  $CK_EXE

Metadata:
  $META

Next:
  cd "$SCRIPT_DIR"
  ./run_mi300x_attention_bench.sh
EOF_DONE
