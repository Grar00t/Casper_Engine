#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="release"
FORCE_ARCH=""
CC_OVERRIDE=""
CXX_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug) CONFIG="debug" ;;
        --release) CONFIG="release" ;;
        --arch) [[ $# -ge 2 ]] || { echo "[build_gcc] missing --arch value" >&2; exit 2; }; FORCE_ARCH="$2"; shift ;;
        --compiler) [[ $# -ge 2 ]] || { echo "[build_gcc] missing --compiler value" >&2; exit 2; }; CC_OVERRIDE="$2"; shift ;;
        *) echo "[build_gcc] unknown flag: $1" >&2; exit 2 ;;
    esac
    shift
done

if [[ -n "$CC_OVERRIDE" ]]; then
    CC="$CC_OVERRIDE"
    case "$CC" in
        *clang*) CXX="${CC/clang/clang++}" ;;
        *gcc*) CXX="${CC/gcc/g++}" ;;
        *) CXX="g++" ;;
    esac
elif command -v gcc >/dev/null 2>&1; then
    CC=gcc; CXX=g++
elif command -v clang >/dev/null 2>&1; then
    CC=clang; CXX=clang++
else
    echo "[build_gcc] no C compiler found (gcc/clang)" >&2
    exit 1
fi

command -v "$CC" >/dev/null 2>&1 || { echo "[build_gcc] compiler unavailable: $CC" >&2; exit 1; }
command -v "$CXX" >/dev/null 2>&1 || { echo "[build_gcc] C++ compiler unavailable: $CXX" >&2; exit 1; }

if [[ -n "$FORCE_ARCH" ]]; then
    ARCH="$FORCE_ARCH"
else
    case "$(uname -m)" in
        x86_64) ARCH="x86_64" ;;
        aarch64|arm64) ARCH="arm64" ;;
        *) ARCH="generic" ;;
    esac
fi

case "$ARCH" in
    x86_64) ARCH_FLAGS="-mavx2 -mfma -march=native" ;;
    arm64) ARCH_FLAGS="-march=armv8.2-a" ;;
    generic) ARCH_FLAGS="" ;;
    *) echo "[build_gcc] unsupported arch: $ARCH" >&2; exit 2 ;;
esac

WARN_C="-Wall -Wextra -Werror -Wstrict-prototypes -Wmissing-prototypes -Wcast-align -Wwrite-strings -Wshadow -pedantic"
WARN_CXX="-Wall -Wextra -Werror -Wcast-align -Wshadow"

if [[ "$CONFIG" == "release" ]]; then
    OPT_C="-O3 -DNDEBUG"; OPT_CXX="-O3 -DNDEBUG"; LINK_FLAGS="-flto"
else
    OPT_C="-O0 -g3 -DDEBUG"; OPT_CXX="-O0 -g3 -DDEBUG"; LINK_FLAGS=""
    if "$CC" -fsanitize=address -x c /dev/null -o /dev/null >/dev/null 2>&1; then
        OPT_C+=" -fsanitize=address,undefined"; OPT_CXX+=" -fsanitize=address,undefined"; LINK_FLAGS+=" -fsanitize=address,undefined"
    fi
fi

build_c() { local out="$1"; shift; "$CC" $OPT_C $ARCH_FLAGS $WARN_C "$@" -o "$out" -lm $LINK_FLAGS; [[ -s "$out" ]]; }
build_cxx() { local out="$1"; shift; "$CXX" -std=c++17 $OPT_CXX $ARCH_FLAGS $WARN_CXX "$@" -o "$out" $LINK_FLAGS; [[ -s "$out" ]]; }

NIYAH_OUT="$ROOT/Core_CPP/niyah"
TRAIN_OUT="$ROOT/Core_CPP/trainer"
HYBRID_OUT="$ROOT/Core_CPP/niyah_hybrid"
BENCH_OUT="$ROOT/Core_CPP/bench_niyah"

build_c "$NIYAH_OUT" "$ROOT/Core_CPP/niyah_core.c" "$ROOT/Core_CPP/niyah_main.c"
[[ -f "$ROOT/Core_CPP/trainer.cpp" ]] && build_cxx "$TRAIN_OUT" "$ROOT/Core_CPP/trainer.cpp"

if [[ -f "$ROOT/Core_CPP/niyah_hybrid_main.c" ]]; then
    build_c "$HYBRID_OUT" \
        "$ROOT/Core_CPP/niyah_hybrid_main.c" \
        "$ROOT/Core_CPP/niyah_core.c" \
        "$ROOT/Core_CPP/hybrid_reasoner.c" \
        "$ROOT/Core_CPP/constraint_solver.c" \
        "$ROOT/Core_CPP/rule_parser.c" \
        "$ROOT/Core_CPP/proof_generator.c" \
        "$ROOT/Core_CPP/khz_q_svd.c" \
        "$ROOT/Core_CPP/casper_rag.c" \
        "$ROOT/tokenizer.c" || { echo "[build_gcc] hybrid build failed" >&2; exit 1; }
fi

if [[ -f "$ROOT/Core_CPP/bench_niyah.c" ]]; then
    build_c "$BENCH_OUT" "$ROOT/Core_CPP/bench_niyah.c" "$ROOT/Core_CPP/niyah_core.c" || { echo "[build_gcc] benchmark build failed" >&2; exit 1; }
fi

for art in "$NIYAH_OUT" "$TRAIN_OUT"; do
    [[ -s "$art" ]] || { echo "[build_gcc] required artifact missing: $art" >&2; exit 1; }
    sha256sum "$art" 2>/dev/null || shasum -a 256 "$art"
done

if [[ "${RUN_LINT:-0}" == "1" ]]; then
    command -v cppcheck >/dev/null 2>&1 || { echo "[build_gcc] cppcheck unavailable" >&2; exit 1; }
    cppcheck --enable=all --error-exitcode=1 --suppress=missingIncludeSystem --suppress=unusedFunction --std=c11 --language=c \
        "$ROOT/Core_CPP/niyah_core.c" "$ROOT/Core_CPP/niyah_main.c"
fi

if [[ "${RUN_SMOKE:-0}" == "1" ]]; then
    cd "$ROOT/Core_CPP"
    ./niyah
    smoke_exit=$?
    [[ $smoke_exit -eq 0 ]] || { echo "[build_gcc] smoke failed: exit $smoke_exit" >&2; exit "$smoke_exit"; }
fi

echo "[build_gcc] BUILD PASS (exit 0; artifacts verified)."
