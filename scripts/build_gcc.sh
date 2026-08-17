#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="release"
FORCE_ARCH=""
CC_OVERRIDE=""

fail() { printf '[build_gcc] ERROR: %s\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug) CONFIG=debug ;;
        --release) CONFIG=release ;;
        --arch) [[ $# -ge 2 ]] || fail 'missing --arch value'; FORCE_ARCH="$2"; shift ;;
        --compiler) [[ $# -ge 2 ]] || fail 'missing --compiler value'; CC_OVERRIDE="$2"; shift ;;
        --help|-h)
            printf '%s\n' 'usage: build_gcc.sh [--debug|--release] [--arch x86_64|arm64|generic] [--compiler gcc|clang|path]';
            exit 0 ;;
        *) fail "unknown flag: $1" ;;
    esac
    shift
done

if [[ -n "$CC_OVERRIDE" ]]; then
    CC="$CC_OVERRIDE"
    case "$(basename "$CC")" in
        gcc) CXX=g++ ;;
        clang) CXX=clang++ ;;
        *) CXX=g++ ;;
    esac
elif command -v gcc >/dev/null 2>&1; then
    CC=gcc; CXX=g++
elif command -v clang >/dev/null 2>&1; then
    CC=clang; CXX=clang++
else
    fail 'no C compiler found'
fi

command -v "$CC" >/dev/null 2>&1 || fail "compiler unavailable: $CC"
command -v "$CXX" >/dev/null 2>&1 || fail "C++ compiler unavailable: $CXX"

if [[ -n "$FORCE_ARCH" ]]; then
    ARCH="$FORCE_ARCH"
else
    case "$(uname -m)" in
        x86_64) ARCH=x86_64 ;;
        aarch64|arm64) ARCH=arm64 ;;
        *) ARCH=generic ;;
    esac
fi

case "$ARCH" in
    x86_64) ARCH_FLAGS=(-mavx2 -mfma -march=native) ;;
    arm64) ARCH_FLAGS=(-march=armv8.2-a) ;;
    generic) ARCH_FLAGS=() ;;
    *) fail "unsupported architecture: $ARCH" ;;
esac

WARN_C=(-Wall -Wextra -Werror -Wstrict-prototypes -Wmissing-prototypes -Wcast-align -Wwrite-strings -Wshadow -pedantic)
WARN_CXX=(-Wall -Wextra -Werror -Wcast-align -Wshadow -pedantic)

if [[ "$CONFIG" == release ]]; then
    CFLAGS=(-O3 -DNDEBUG)
    CXXFLAGS=(-O3 -DNDEBUG)
    LDFLAGS=(-flto)
else
    CFLAGS=(-O0 -g3 -DDEBUG)
    CXXFLAGS=(-O0 -g3 -DDEBUG)
    LDFLAGS=()
    if "$CC" -fsanitize=address -x c /dev/null -o /dev/null >/dev/null 2>&1; then
        CFLAGS+=(-fsanitize=address,undefined)
        CXXFLAGS+=(-fsanitize=address,undefined)
        LDFLAGS+=(-fsanitize=address,undefined)
    fi
fi

build_c() {
    local out="$1"; shift
    mkdir -p "$(dirname "$out")"
    "$CC" "${CFLAGS[@]}" "${ARCH_FLAGS[@]}" "${WARN_C[@]}" -std=c11 \
        -I"$ROOT/include" -I"$ROOT/Core_CPP" "$@" -o "$out" -lm "${LDFLAGS[@]}"
    [[ -s "$out" ]] || fail "artifact missing/empty: $out"
}

build_cxx() {
    local out="$1"; shift
    mkdir -p "$(dirname "$out")"
    "$CXX" "${CXXFLAGS[@]}" "${ARCH_FLAGS[@]}" "${WARN_CXX[@]}" -std=c++17 \
        -I"$ROOT/include" -I"$ROOT/Core_CPP" "$@" -o "$out" "${LDFLAGS[@]}"
    [[ -s "$out" ]] || fail "artifact missing/empty: $out"
}

NIYAH_OUT="$ROOT/Core_CPP/niyah"
TRAIN_OUT="$ROOT/niyah_train"
HYBRID_OUT="$ROOT/Core_CPP/niyah_hybrid"
BENCH_OUT="$ROOT/Core_CPP/bench_niyah"

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

build_c "$BENCH_OUT" \
    "$ROOT/Core_CPP/bench_niyah.c" \
    "$ROOT/Core_CPP/niyah_core.c"

hash_file() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}';
    elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | awk '{print $1}';
    else fail 'no SHA-256 utility available'; fi
}

for artifact in "$NIYAH_OUT" "$TRAIN_OUT" "$HYBRID_OUT" "$BENCH_OUT"; do
    [[ -s "$artifact" ]] || fail "required artifact missing: $artifact"
    printf 'SHA256 %s %s\n' "$(hash_file "$artifact")" "$(basename "$artifact")"
done

if [[ "${RUN_LINT:-0}" == 1 ]]; then
    command -v cppcheck >/dev/null 2>&1 || fail 'RUN_LINT=1 but cppcheck is unavailable'
    cppcheck --enable=all --error-exitcode=1 --suppress=missingIncludeSystem --std=c11 --language=c \
        "$ROOT/Core_CPP/niyah_core.c" \
        "$ROOT/Core_CPP/niyah_main.c" \
        "$ROOT/Core_CPP/casper_rag.c"
fi

if [[ "${RUN_SMOKE:-0}" == 1 ]]; then
    [[ -x "$NIYAH_OUT" ]] || fail "smoke artifact is not executable: $NIYAH_OUT"
    set +e
    "$NIYAH_OUT"
    smoke_exit=$?
    set -e
    [[ $smoke_exit -eq 0 ]] || fail "smoke failed: exit=$smoke_exit"
    printf '%s\n' '[build_gcc] SMOKE PASS (exit 0)'
fi

printf '%s\n' '[build_gcc] BUILD PASS (exit 0; artifacts verified).'
