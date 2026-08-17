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

if ! command -v "$CC" >/dev/null 2>&1; then
    echo "[build_gcc] compiler unavailable: $CC" >&2
    exit 1
fi
if ! command -v "$CXX" >/dev/null 2>&1; then
    echo "[build_gcc] C++ compiler unavailable: $CXX" >&2
    exit 1
fi

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
    OPT_C="-O3 -DNDEBUG"
    OPT_CXX="-O3 -DNDEBUG"
    LINK_FLAGS="-flto"
else
    OPT_C="-O0 -g3 -DDEBUG"
    OPT_CXX="-O0 -g3 -DDEBUG"
    LINK_FLAGS=""
    if "$CC" -fsanitize=address -x c /dev/null -o /dev/null >/dev/null 2>&1; then
        OPT_C+=" -fsanitize=address,undefined"
        OPT_CXX+=" -fsanitize=address,undefined"
        LINK_FLAGS+=" -fsanitize=address,undefined"
    fi
fi

build_c() {
    local out="$1"; shift
    "$CC" $OPT_C $ARCH_FLAGS $WARN_C "$@" -o "$out" -lm $LINK_FLAGS
    [[ -s "$out" ]] || { echo "[build_gcc] artifact missing/empty: $out" >&2; return 1; }
}

build_cxx() {
    local out="$1"; shift
    "$CXX" -std=c++17 $OPT_CXX $ARCH_FLAGS $WARN_CXX "$@" -o "$out" $LINK_FLAGS
    [[ -s "$out" ]] || { echo "[build_gcc] artifact missing/empty: $out" >&2; return 1; }
}

NIYAH_OUT="$ROOT/Core_CPP/niyah"
TRAIN_OUT="$ROOT/Core_CPP/trainer"

build_c "$NIYAH_OUT" \
    "$ROOT/Core_CPP/niyah_core.c" \
    "$ROOT/Core_CPP/niyah_main.c"

if [[ -f "$ROOT/Core_CPP/trainer.cpp" ]]; then
    build_cxx "$TRAIN_OUT" "$ROOT/Core_CPP/trainer.cpp"
else
    echo "[build_gcc] trainer.cpp missing" >&2
    exit 1
fi

for art in "$NIYAH_OUT" "$TRAIN_OUT"; do
    [[ -s "$art" ]] || { echo "[build_gcc] required artifact missing: $art" >&2; exit 1; }
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$art"
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$art"
    fi
done

if [[ "${RUN_LINT:-0}" == "1" ]]; then
    if ! command -v cppcheck >/dev/null 2>&1; then
        echo "[build_gcc] cppcheck unavailable" >&2
        exit 1
    fi
    cppcheck --enable=all --error-exitcode=1 --suppress=missingIncludeSystem --suppress=unusedFunction --std=c11 --language=c \
        "$ROOT/Core_CPP/niyah_core.c" "$ROOT/Core_CPP/niyah_main.c"
fi

if [[ "${RUN_SMOKE:-0}" == "1" ]]; then
    cd "$ROOT/Core_CPP"
    ./niyah
    smoke_exit=$?
    [[ $smoke_exit -eq 0 ]] || { echo "[build_gcc] smoke failed: exit $smoke_exit" >&2; exit "$smoke_exit"; }
    echo "[build_gcc] SMOKE PASS (exit 0)"
fi

echo "[build_gcc] BUILD PASS (exit 0; artifacts verified)."