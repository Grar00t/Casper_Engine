#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="release"
FORCE_ARCH=""
CC_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug) CONFIG="debug" ;;
        --release) CONFIG="release" ;;
        --arch) [[ $# -ge 2 ]] || { echo "[build_gcc] missing --arch value" >&2; exit 2; }; FORCE_ARCH="$2"; shift ;;
        --compiler) [[ $# -ge 2 ]] || { echo "[build_gcc] missing --compiler value" >&2; exit 2; }; CC_OVERRIDE="$2"; shift ;;
        --lint) RUN_LINT=1 ;;
        --smoke) RUN_SMOKE=1 ;;
        --bench) RUN_BENCH=1 ;;
        *) echo "[build_gcc] unknown flag: $1" >&2; exit 2 ;;
    esac
    shift
done

RUN_LINT="${RUN_LINT:-0}"
RUN_SMOKE="${RUN_SMOKE:-0}"
RUN_BENCH="${RUN_BENCH:-0}"

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

WARN="-Wall -Wextra -Werror -Wstrict-prototypes -Wmissing-prototypes -Wcast-align -Wwrite-strings -Wshadow -pedantic"

if [[ "$CONFIG" == "release" ]]; then
    OPT="-O3 -DNDEBUG"
    LDFLAGS="-lm -flto"
else
    OPT="-O0 -g3 -DDEBUG"
    LDFLAGS="-lm"
    if "$CC" -fsanitize=address -x c /dev/null -o /dev/null >/dev/null 2>&1; then
        OPT="$OPT -fsanitize=address,undefined -fno-omit-frame-pointer"
        LDFLAGS="$LDFLAGS -fsanitize=address,undefined"
    fi
fi

build_c() {
    local out="$1"; shift
    "$CC" $OPT $ARCH_FLAGS $WARN "$@" -o "$out" $LDFLAGS
    [[ -s "$out" ]] || { echo "[build_gcc] artifact missing/empty: $out" >&2; return 1; }
}

NIYAH_OUT="$ROOT/Core_CPP/niyah"
TRAIN_OUT="$ROOT/Core_CPP/trainer"
HYBRID_OUT="$ROOT/Core_CPP/niyah_hybrid"
BENCH_OUT="$ROOT/Core_CPP/bench_niyah"
CASPER_OUT="$ROOT/Core_CPP/casper"

build_c "$NIYAH_OUT" \
    "$ROOT/Core_CPP/niyah_core.c" \
    "$ROOT/Core_CPP/niyah_main.c"

build_c "$TRAIN_OUT" \
    "$ROOT/Core_CPP/niyah_train.c" \
    "$ROOT/Core_CPP/niyah_core.c" \
    "$ROOT/tokenizer.c"

build_c "$HYBRID_OUT" \
    "$ROOT/Core_CPP/niyah_hybrid_main.c" \
    "$ROOT/Core_CPP/niyah_core.c" \
    "$ROOT/Core_CPP/hybrid_reasoner.c" \
    "$ROOT/Core_CPP/constraint_solver.c" \
    "$ROOT/Core_CPP/rule_parser.c" \
    "$ROOT/Core_CPP/proof_generator.c" \
    "$ROOT/Core_CPP/khz_q_svd.c" \
    "$ROOT/Core_CPP/casper_rag.c" \
    "$ROOT/tokenizer.c"

build_c "$CASPER_OUT" \
    "$ROOT/Core_CPP/casper_cli.c" \
    "$ROOT/Core_CPP/casper_rag.c" \
    "$ROOT/Core_CPP/rule_parser.c" \
    "$ROOT/Core_CPP/proof_generator.c" \
    "$ROOT/Core_CPP/khz_q_svd.c"

if [[ -f "$ROOT/Core_CPP/bench_niyah.c" ]]; then
    build_c "$BENCH_OUT" \
        "$ROOT/Core_CPP/bench_niyah.c" \
        "$ROOT/Core_CPP/niyah_core.c"
fi

if [[ "$RUN_LINT" == "1" ]]; then
    command -v cppcheck >/dev/null 2>&1 || { echo "[build_gcc] cppcheck unavailable" >&2; exit 1; }
    cppcheck \
        --enable=warning,style,performance,portability \
        --error-exitcode=1 \
        --suppress=missingIncludeSystem \
        --suppress=unusedFunction \
        --std=c11 \
        -I "$ROOT/include" \
        "$ROOT/Core_CPP/niyah_core.c" \
        "$ROOT/Core_CPP/niyah_main.c" \
        "$ROOT/Core_CPP/niyah_train.c" \
        "$ROOT/Core_CPP/niyah_hybrid_main.c" \
        "$ROOT/Core_CPP/casper_cli.c" \
        "$ROOT/Core_CPP/casper_rag.c" \
        "$ROOT/Core_CPP/rule_parser.c" \
        "$ROOT/Core_CPP/proof_generator.c" \
        "$ROOT/Core_CPP/constraint_solver.c" \
        "$ROOT/Core_CPP/hybrid_reasoner.c" \
        "$ROOT/Core_CPP/khz_q_svd.c"
fi

if [[ "$RUN_SMOKE" == "1" ]]; then
    if "$NIYAH_OUT"; then
        :
    else
        smoke_exit=$?
        echo "[build_gcc] smoke failed: exit $smoke_exit" >&2
        exit "$smoke_exit"
    fi
    if "$HYBRID_OUT" --smoke; then
        :
    else
        smoke_exit=$?
        echo "[build_gcc] hybrid smoke failed: exit $smoke_exit" >&2
        exit "$smoke_exit"
    fi
fi

if [[ "$RUN_BENCH" == "1" ]]; then
    [[ -s "$BENCH_OUT" ]] || { echo "[build_gcc] benchmark artifact missing" >&2; exit 1; }
    if "$BENCH_OUT"; then :; else bench_exit=$?; echo "[build_gcc] benchmark failed: exit $bench_exit" >&2; exit "$bench_exit"; fi
fi

for art in "$NIYAH_OUT" "$TRAIN_OUT" "$HYBRID_OUT" "$CASPER_OUT"; do
    [[ -s "$art" ]] || { echo "[build_gcc] required artifact missing: $art" >&2; exit 1; }
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$art"
    else
        shasum -a 256 "$art"
    fi
done

echo "[build_gcc] BUILD PASS (exit 0; required artifacts verified)."