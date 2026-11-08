#!/bin/bash
#
# Checks that every vendored ONNX Runtime prebuilt was built with every
# operator in moonshine-required-operators.config.
#
# The minimal ORT builds under core/third-party/onnxruntime/lib/ only carry
# the kernels named in that config, and a model that needs one they lack
# fails at session creation on the device. The config is regenerated whenever
# a model changes, but the prebuilts are rebuilt by hand per platform, so the
# two drift apart silently: the 0.1.5 and 0.1.6 Android AARs shipped an ORT
# built from the August 1 config, three weeks before QLinearMatMul was added
# for Kokoro, and Kokoro text to speech could not load on Android (GitHub
# issue #242). Nothing in the test suites noticed because the Android Kokoro
# test is opt-in and runs against the CDN.
#
# Operator names survive stripping as the string literals kernel definitions
# are registered with, so `strings` on the library is enough to tell: an
# operator whose name never appears was not compiled in. Two toolchain
# details shape how the match is done:
#
# - Short names such as Add and Cos need `strings -n 2`; the default minimum
#   length of 4 drops them.
# - GNU ld and lld tail-merge string literals, so on Linux "Add" is stored
#   only as the tail of "QLinearAdd". The match therefore accepts an operator
#   name as the suffix of any string. That means an operator whose name is a
#   suffix of another present operator (Add, MatMul, Gelu, Conv, Concat,
#   Sigmoid, LeakyRelu, LSTM, QuantizeLinear) cannot be proven absent, but
#   those are all long-standing core kernels; the ones that get added for a
#   new model (QLinearMatMul, Elu, MultiHeadAttention, ...) are detected.
#
# The WASM archive is LLVM bitcode (LTO objects), in which string literals
# are not stored as plain bytes, so it is skipped; the WASM test suite loads
# every model that ships on the web and covers it instead. The Windows
# import library (onnxruntime.lib) carries no kernels and is not matched.
#
# Usage:
#   scripts/check-ort-prebuilt-ops.sh [library ...]
#
# With no arguments every vendored library is checked. Exits non-zero if any
# library is missing any operator, printing which.
set -uo pipefail

# `strings` output includes arbitrary bytes; without this, sort and grep abort
# on the first invalid UTF-8 sequence and the library reads as having nothing.
export LC_ALL=C

REPO_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OP_CONFIG="${REPO_ROOT_DIR}/core/third-party/onnxruntime/moonshine-required-operators.config"
LIB_ROOT="${REPO_ROOT_DIR}/core/third-party/onnxruntime/lib"

if [ ! -f "${OP_CONFIG}" ]; then
    echo "ERROR: operator config not found at ${OP_CONFIG}" >&2
    exit 2
fi

# Config lines look like `ai.onnx;10;ConvInteger,QLinearConv,QLinearMatMul`;
# the third field is the operator list.
ops="$(grep -v '^[[:space:]]*#' "${OP_CONFIG}" | grep -v '^[[:space:]]*$' \
    | cut -d';' -f3 | tr ',' '\n' | sed 's/[[:space:]]//g' | grep -v '^$' | sort -u)"
if [ -z "${ops}" ]; then
    echo "ERROR: no operators parsed from ${OP_CONFIG}" >&2
    exit 2
fi

if [ $# -gt 0 ]; then
    libs=("$@")
else
    libs=()
    while IFS= read -r lib; do
        libs+=("${lib}")
    done < <(find "${LIB_ROOT}" -type f \
        \( -name 'libonnxruntime*.so*' -o -name 'libonnxruntime*.a' \
           -o -name 'libonnxruntime*.dylib' -o -name 'onnxruntime.dll' \) \
        | sort)
fi

if [ "${#libs[@]}" -eq 0 ]; then
    echo "ERROR: no ONNX Runtime libraries found under ${LIB_ROOT}" >&2
    exit 2
fi

status=0
for lib in "${libs[@]}"; do
    rel="${lib#"${REPO_ROOT_DIR}/"}"
    if [ ! -f "${lib}" ]; then
        echo "FAIL ${rel}: not found" >&2
        status=1
        continue
    fi
    # A Git LFS pointer file is a few hundred bytes of text, not a library.
    if head -c 40 "${lib}" | grep -q 'version https://git-lfs'; then
        echo "FAIL ${rel}: is a Git LFS pointer; run 'git lfs pull'" >&2
        status=1
        continue
    fi
    case "${rel}" in
        */lib/wasm/*)
            echo "skip ${rel}: LLVM bitcode, covered by scripts/test-wasm.sh"
            continue
            ;;
    esac
    names="$(strings -n 2 "${lib}" 2>/dev/null | sort -u)"
    missing=()
    while IFS= read -r op; do
        if ! grep -q -- "${op}\$" <<<"${names}"; then
            missing+=("${op}")
        fi
    done <<<"${ops}"
    if [ "${#missing[@]}" -eq 0 ]; then
        echo "ok   ${rel}"
    else
        echo "FAIL ${rel}: missing ${#missing[@]} operator(s) from the config:" >&2
        printf '       %s\n' "${missing[@]}" >&2
        status=1
    fi
done

if [ ${status} -ne 0 ]; then
    cat >&2 <<EOF

A library above was built from an older operator config than the one in the
tree, so a model that uses one of the listed operators will fail to load on
that platform. Rebuild and re-vendor it:
  scripts/build-ort-android.sh force
  scripts/build-ort-ios.sh force
EOF
fi
exit ${status}
