# shellcheck shell=bash
# Shared setup for the from-source ONNX Runtime builds.
#
# Sourced by build-ort-wasm.sh, build-ort-android.sh and build-ort-ios.sh. It
# exists so the three cannot drift: they must build the same ORT version, from
# one checkout, against one operator config. A mismatch between them is not a
# build error, it is a runtime failure on one platform only.
#
# Not executable on its own.

# The ORT version every platform is pinned to.
#
# Keep in lockstep with core/third-party/onnxruntime/find-ort-library-path.cmake
# and with the prebuilt libraries vendored for platforms we do not build from
# source (see core/third-party/onnxruntime/lib/*/README.md). The vendored
# headers are shared across all platforms, so a library built from a different
# version can fail in ways that do not show up until a session is created.
ORT_VERSION="${ORT_VERSION:-1.23.2}"

# Where the checkout and per-platform build trees live. Shared across
# platforms: a recursive ORT checkout is around a gigabyte, and op reduction
# writes into the build directory rather than the source, so one source tree
# serves every target.
if [ -z "${MOONSHINE_ORT_ROOT:-}" ]; then
    if [ -d "${HOME}/moonshine-ort-wasm/onnxruntime/.git" ]; then
        # Where the wasm build put things before the mobile builds existed.
        MOONSHINE_ORT_ROOT="${HOME}/moonshine-ort-wasm"
    else
        MOONSHINE_ORT_ROOT="${HOME}/moonshine-ort"
    fi
fi

ORT_SRC="${MOONSHINE_ORT_ROOT}/onnxruntime"
OP_CONFIG="${REPO_ROOT_DIR}/core/third-party/onnxruntime/moonshine-required-operators.config"

# Fails unless the operator config is present.
#
# Every from-source build is restricted to the operators in this file. Building
# without it would silently produce a full-size library.
ort_require_op_config() {
    if [ ! -f "${OP_CONFIG}" ]; then
        echo "ERROR: operator config not found at ${OP_CONFIG}" >&2
        echo "Generate it with scripts/generate-ort-op-config.py" >&2
        exit 1
    fi
}

# Clones the pinned ORT if absent, else refreshes submodules.
ort_prepare_checkout() {
    if [ ! -d "${ORT_SRC}/.git" ]; then
        mkdir -p "${MOONSHINE_ORT_ROOT}"
        echo "cloning onnxruntime v${ORT_VERSION} (recursive)..."
        git clone --recursive --depth 1 --branch "v${ORT_VERSION}" \
            https://github.com/microsoft/onnxruntime.git "${ORT_SRC}"
    else
        echo "reusing existing ORT checkout at ${ORT_SRC}"
        git -C "${ORT_SRC}" submodule update --init --recursive
    fi
}

# The build flags that make a build minimal, shared by all platforms.
#
#   --minimal_build extended custom_ops
#       Drops the ONNX parser and the graph optimisers a minimal build cannot
#       use, so only ORT-format models load. `extended` keeps the runtime
#       optimisers and is *required* by execution providers that compile
#       kernels at load time (NNAPI, CoreML); `custom_ops` keeps custom-op
#       registration, which ZipVoice needs.
#   --include_ops_by_config
#       Restricts the build to the operators our models use. ORT writes the
#       reduced kernel registrations into the build directory (see
#       reduce_ops() in tools/ci_build/build.py), deleting any it generated
#       before, so the source tree is never edited and stale exclusions cannot
#       carry over between builds.
#   --disable_ml_ops
#       Drops the classical-ML (ai.onnx.ml) kernels, which no model uses.
#   --compile_no_warning_as_error
#       Works around ORT 1.23: in a minimal build
#       `min_ort_version_with_shape_inference` in core/session/custom_ops.cc is
#       unused and the build otherwise fails on
#       -Werror,-Wunused-const-variable. Drop this once the upstream fix lands
#       in a version we pin to.
ort_minimal_flags() {
    printf '%s\n' \
        --minimal_build extended custom_ops \
        --include_ops_by_config "${OP_CONFIG}" \
        --disable_ml_ops \
        --compile_no_warning_as_error
}

# Prints a size in MB for a file.
ort_report_size() {
    local label="$1" path="$2"
    if [ -f "${path}" ]; then
        local bytes
        bytes="$(stat -f%z "${path}" 2>/dev/null || stat -c%s "${path}")"
        awk -v l="${label}" -v b="${bytes}" \
            'BEGIN { printf "  %-22s %8.1f MB\n", l, b / 1048576 }'
    fi
}
