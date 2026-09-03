#!/usr/bin/env bash
#
# scripts/build.sh - single build entry point for the Casper/NIYAH C sources.
#
# Replaces the two divergent scripts that used to exist:
#   Core_CPP/build_gcc.sh  had -std=c11 and -I include, but only built niyah
#                          and looked for bench_niyah.c in a bench/ directory
#                          that does not exist in this repository.
#   scripts/build_gcc.sh   built every target, but with no -std=, no -I, and
#                          wrote artifacts straight into Core_CPP/, which is
#                          how compiled binaries kept landing in git.
#
# Usage:
#   bash scripts/build.sh                 # release
#   bash scripts/build.sh --debug         # -O0 -g3, ASan+UBSan when available
#   bash scripts/build.sh --arch generic  # no -march=native, portable binary
#   bash scripts/build.sh --lint          # cppcheck gate before compiling
#   bash scripts/build.sh --smoke         # run binaries after building
#   bash scripts/build.sh --bench         # run the benchmark after building
#
# Every artifact is written to build/ and hashed at the end.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CORE="$ROOT/Core_CPP"
INCLUDE="$ROOT/include"
BUILD="$ROOT/build"

CONFIG="release"
FORCE_ARCH=""
CC_OVERRIDE=""
RUN_LINT="${RUN_LINT:-0}"
RUN_SMOKE="${RUN_SMOKE:-0}"
RUN_BENCH="${RUN_BENCH:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug)    CONFIG="debug" ;;
        --release)  CONFIG="release" ;;
        --arch)     [[ $# -ge 2 ]] || { echo "[build] --arch needs a value" >&2; exit 2; }
                    FORCE_ARCH="$2"; shift ;;
        --compiler) [[ $# -ge 2 ]] || { echo "[build] --compiler needs a value" >&2; exit 2; }
                    CC_OVERRIDE="$2"; shift ;;
        --lint)     RUN_LINT=1 ;;
        --smoke)    RUN_SMOKE=1 ;;
        --bench)    RUN_BENCH=1 ;;
        -h|--help)  sed -n '3,24p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *)          echo "[build] unknown flag: $1" >&2; exit 2 ;;
    esac
    shift
done

# ---- compiler -------------------------------------------------------------
if [[ -n "$CC_OVERRIDE" ]]; then
    CC="$CC_OVERRIDE"
elif [[ -n "${CC:-}" ]]; then
    CC="$CC"
elif command -v gcc >/dev/null 2>&1; then
    CC=gcc
elif command -v clang >/dev/null 2>&1; then
    CC=clang
else
    echo "[build] no C compiler found (gcc/clang)" >&2
    exit 1
fi
command -v "$CC" >/dev/null 2>&1 || { echo "[build] compiler unavailable: $CC" >&2; exit 1; }

# ---- architecture ---------------------------------------------------------
if [[ -n "$FORCE_ARCH" ]]; then
    ARCH="$FORCE_ARCH"
else
    case "$(uname -m)" in
        x86_64)       ARCH="x86_64" ;;
        aarch64|arm64) ARCH="arm64" ;;
        *)            ARCH="generic" ;;
    esac
fi

case "$ARCH" in
    x86_64)
        # -march=native makes the binary unusable on older CPUs. Use
        # --arch generic for anything you intend to distribute.
        if echo "" | "$CC" -x c -mavx2 -mfma -E - >/dev/null 2>&1; then
            ARCH_FLAGS="-mavx2 -mfma -march=native"
            ARCH_NAME="x86_64 AVX2+FMA (native)"
        else
            ARCH_FLAGS="-march=native"
            ARCH_NAME="x86_64 scalar (native)"
        fi
        ;;
    arm64)   ARCH_FLAGS="-march=armv8.2-a"; ARCH_NAME="arm64 NEON (armv8.2-a)" ;;
    generic) ARCH_FLAGS="";                 ARCH_NAME="generic portable" ;;
    *)       echo "[build] unsupported arch: $ARCH" >&2; exit 2 ;;
esac

# ---- flags ---------------------------------------------------------------
STD="-std=c11"
WARN="-Wall -Wextra -Werror -Wstrict-prototypes -Wmissing-prototypes"
WARN="$WARN -Wcast-align -Wwrite-strings -Wshadow -pedantic"

SAN=""
if [[ "$CONFIG" == "release" ]]; then
    OPT="-O3 -DNDEBUG -flto"
    LDFLAGS="-lm -flto"
else
    OPT="-O0 -g3 -DDEBUG"
    LDFLAGS="-lm"
    # The old probe compiled AND linked an empty translation unit, so it
    # always failed on "undefined reference to main" and ASan was never
    # enabled. Compile only (-c) to test flag support.
    if echo 'int main(void){return 0;}' \
         | "$CC" -fsanitize=address,undefined -x c -c - -o /dev/null >/dev/null 2>&1; then
        SAN="-fsanitize=address,undefined -fno-omit-frame-pointer"
        OPT="$OPT $SAN"
        LDFLAGS="$LDFLAGS $SAN"
    else
        echo "[build] note: sanitizers unavailable, continuing without them"
    fi
fi

CFLAGS="$STD $OPT $ARCH_FLAGS $WARN -I$INCLUDE -I$CORE"

echo "============================================="
echo "  Casper / NIYAH build"
echo "  CC:      $("$CC" --version | head -1)"
echo "  Config:  $CONFIG"
echo "  Arch:    $ARCH_NAME"
echo "  Output:  ${BUILD#"$ROOT/"}/"
echo "============================================="

mkdir -p "$BUILD"

# ---- cppcheck gate -------------------------------------------------------
if [[ "$RUN_LINT" == "1" ]]; then
    command -v cppcheck >/dev/null 2>&1 || {
        echo "[build] cppcheck not found - apt install cppcheck" >&2; exit 1; }
    echo "-- cppcheck ---------------------------------"
    cppcheck \
        --error-exitcode=1 \
        --enable=warning,style,performance,portability \
        --suppress=missingIncludeSystem \
        --suppress=unusedFunction \
        --std=c11 \
        -I "$INCLUDE" -I "$CORE" \
        "$ROOT/tokenizer.c" \
        "$CORE/niyah_core.c"      "$CORE/niyah_main.c" \
        "$CORE/niyah_train.c"     "$CORE/niyah_hybrid_main.c" \
        "$CORE/casper_cli.c"      "$CORE/casper_rag.c" \
        "$CORE/rule_parser.c"     "$CORE/proof_generator.c" \
        "$CORE/constraint_solver.c" "$CORE/hybrid_reasoner.c" \
        "$CORE/khz_q_svd.c"
    echo "   cppcheck: clean"
fi

# ---- build ---------------------------------------------------------------
build_target() {
    local out="$BUILD/$1"; shift
    printf '%-16s' "$(basename "$out")"
    # shellcheck disable=SC2086
    "$CC" $CFLAGS "$@" -o "$out" $LDFLAGS
    [[ -s "$out" ]] || { echo "  MISSING"; return 1; }
    printf '  ok  %8d bytes\n' "$(wc -c <"$out")"
}

echo "-- compile ----------------------------------"

build_target niyah \
    "$CORE/niyah_core.c" "$CORE/niyah_main.c"

build_target trainer \
    "$CORE/niyah_train.c" "$CORE/niyah_core.c" "$ROOT/tokenizer.c"

build_target niyah_hybrid \
    "$CORE/niyah_hybrid_main.c" "$CORE/niyah_core.c" \
    "$CORE/hybrid_reasoner.c" "$CORE/constraint_solver.c" \
    "$CORE/rule_parser.c" "$CORE/proof_generator.c" \
    "$CORE/khz_q_svd.c" "$CORE/casper_rag.c" "$ROOT/tokenizer.c"

# casper_cli.c includes casper_rag.h, rule_parser.h and proof_generator.h
# only, and calls no tokenizer function, so tokenizer.c is deliberately not
# linked here.
build_target casper \
    "$CORE/casper_cli.c" "$CORE/casper_rag.c" "$CORE/rule_parser.c" \
    "$CORE/proof_generator.c" "$CORE/khz_q_svd.c"

if [[ -f "$CORE/bench_niyah.c" ]]; then
    build_target bench_niyah "$CORE/bench_niyah.c" "$CORE/niyah_core.c"
fi

# Tokenizer round-trip test: fails the build if any character decodes to '?'.
printf '%-16s' "tokenizer_test"
# shellcheck disable=SC2086
"$CC" $CFLAGS -DTOKENIZER_TEST "$ROOT/tokenizer.c" -o "$BUILD/tokenizer_test" $LDFLAGS
printf '  ok\n'
"$BUILD/tokenizer_test"

# ---- optional runs -------------------------------------------------------
if [[ "$RUN_SMOKE" == "1" ]]; then
    echo "-- smoke ------------------------------------"
    "$BUILD/niyah"
    "$BUILD/niyah_hybrid" --smoke
fi

if [[ "$RUN_BENCH" == "1" && -x "$BUILD/bench_niyah" ]]; then
    echo "-- bench ------------------------------------"
    "$BUILD/bench_niyah"
fi

# ---- manifest ------------------------------------------------------------
echo "-- artifacts --------------------------------"
for art in "$BUILD/niyah" "$BUILD/trainer" "$BUILD/niyah_hybrid" "$BUILD/casper"; do
    [[ -s "$art" ]] || { echo "[build] required artifact missing: $art" >&2; exit 1; }
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$art"
    else
        shasum -a 256 "$art"
    fi
done

echo "[build] PASS  config=$CONFIG arch=$ARCH"
