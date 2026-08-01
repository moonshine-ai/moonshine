#!/usr/bin/env bash
# Report what Moonshine actually costs a mobile app to install.
#
# Archive sizes on disk are misleading on both platforms, in opposite
# directions. An Android .so ships whole, so its file size is close to honest,
# but the AAR holds every ABI while a device downloads only one. An iOS static
# .a is far *larger* than what it contributes, because the app linker drops the
# object files nothing references. So this links a real binary for iOS and
# reports per-ABI figures for Android.
#
# Use it before and after changing how ONNX Runtime is built, so a claimed
# saving is a measurement rather than an inference from archive sizes.
#
# Usage:
#   scripts/measure-mobile-size.sh            # both platforms
#   scripts/measure-mobile-size.sh android
#   scripts/measure-mobile-size.sh ios
set -euo pipefail

REPO_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORT_LIB_DIR="${REPO_ROOT_DIR}/core/third-party/onnxruntime/lib"
XCFRAMEWORK="${REPO_ROOT_DIR}/swift/Moonshine.xcframework"

WHICH="${1:-all}"

mb() { awk -v b="$1" 'BEGIN { printf "%.1f MB", b / 1048576 }'; }

file_size() {
    stat -f%z "$1" 2>/dev/null || stat -c%s "$1"
}

# Reports per-ABI native payload from a built AAR.
#
# Measured from the AAR rather than from build/intermediates, because Gradle
# strips the native libraries on the way in: the intermediates run to ~98 MB
# for a library that ships at ~18 MB. A device installs one ABI, so the per-ABI
# total is what a user pays; the AAR total is not.
measure_android() {
    echo "=== Android ==="
    local aar="${MOONSHINE_AAR:-}"
    if [ -z "${aar}" ]; then
        # Most recently modified wins, across the three places a build can leave
        # an AAR: Gradle's own output directory for a plain `build-android.sh`,
        # build/publish/staging for a release build, and the local Maven cache
        # for `build-android.sh local`. Sorting by version instead would pick a
        # stale release over the local build you just made.
        aar="$(find "${REPO_ROOT_DIR}/build/outputs/aar" \
            "${REPO_ROOT_DIR}/build/publish" \
            "${HOME}/.m2/repository/ai/moonshine" \
            -name '*.aar' 2>/dev/null \
            | while IFS= read -r f; do
                  printf '%s %s\n' "$(stat -f%m "$f" 2>/dev/null || stat -c%Y "$f")" "$f"
              done | sort -n | tail -1 | cut -d' ' -f2-)"
    fi
    if [ -z "${aar}" ] || [ ! -f "${aar}" ]; then
        echo "  no AAR found; run scripts/build-android.sh local"
        echo "  (or set MOONSHINE_AAR=/path/to.aar)"
        echo "  vendored ORT libraries meanwhile:"
        local dir
        for dir in "${ORT_LIB_DIR}"/android/*/; do
            [ -f "${dir}/libonnxruntime.so" ] || continue
            printf "    %-14s %s\n" "$(basename "${dir}")" \
                "$(mb "$(file_size "${dir}/libonnxruntime.so")")"
        done
        return
    fi

    echo "  ${aar#"${REPO_ROOT_DIR}"/}"
    printf "  %-14s %s (all ABIs; nobody downloads this)\n" "AAR total" \
        "$(mb "$(file_size "${aar}")")"
    unzip -l "${aar}" | awk '
        /jni\/.*\.so$/ {
            split($4, parts, "/")
            abi = parts[2]
            total[abi] += $1
            if ($4 ~ /libonnxruntime\.so$/) ort[abi] = $1
            seen[abi] = 1
        }
        END {
            for (abi in seen)
                printf "  %-14s ort %8.1f MB   total %8.1f MB\n",
                    abi, ort[abi] / 1048576, total[abi] / 1048576
        }' | sort
}

# Links a minimal iOS binary against the framework and reports its __TEXT.
#
# -dead_strip is what makes this meaningful: it reproduces the linker's choice
# about which of ORT's object files an app actually pulls in.
measure_ios() {
    echo "=== iOS (arm64 device slice) ==="
    # MOONSHINE_IOS_LIB points the measurement at some other libmoonshine.a,
    # which is how a before-and-after is taken: build the device slice against
    # the old ONNX Runtime somewhere else and measure that archive.
    local lib="${MOONSHINE_IOS_LIB:-${XCFRAMEWORK}/ios-arm64/libmoonshine.a}"
    if [ ! -f "${lib}" ]; then
        echo "  no ${lib}; run scripts/build-swift.sh first"
        echo "  (or set MOONSHINE_IOS_LIB=/path/to/libmoonshine.a)"
        return
    fi
    printf "  static archive        %s (not what ships)\n" "$(mb "$(file_size "${lib}")")"

    local work
    work="$(mktemp -d)"
    trap 'rm -rf "${work}"' RETURN

    # Reference the entry points a real app uses, so the linker keeps the
    # transcription, TTS and embedding paths and everything they pull in.
    cat >"${work}/main.c" <<'EOF'
#include <stdint.h>
extern int32_t moonshine_load_transcriber_from_files(const char *, int32_t,
                                                     const void *, uint64_t,
                                                     void **);
extern int32_t moonshine_create_tts_synthesizer_from_files(const char *,
                                                           const void *,
                                                           uint64_t, void **);
extern int32_t moonshine_create_embedding_model(const char *, const void *,
                                                uint64_t, void **);
int main(void) {
    void *handle = 0;
    moonshine_load_transcriber_from_files("m", 0, 0, 0, &handle);
    moonshine_create_tts_synthesizer_from_files("m", 0, 0, &handle);
    moonshine_create_embedding_model("m", 0, 0, &handle);
    return 0;
}
EOF

    local sdk
    sdk="$(xcrun --sdk iphoneos --show-sdk-path)"
    if ! xcrun clang -target arm64-apple-ios15.1 -isysroot "${sdk}" \
        -fno-objc-arc -O2 -dead_strip \
        "${work}/main.c" "${lib}" \
        -lc++ -framework Foundation -framework CoreFoundation \
        -framework Accelerate \
        -o "${work}/app" 2>"${work}/link.err"; then
        echo "  link failed; see below"
        sed 's/^/    /' "${work}/link.err" | head -20
        return
    fi

    printf "  linked binary         %s\n" "$(mb "$(file_size "${work}/app")")"
    # __TEXT is the executable code and read-only data, which is the part that
    # grows and shrinks with how much of ORT gets linked in.
    xcrun size -m "${work}/app" | awk '
        /^Segment __TEXT:/ { printf "  __TEXT                %.1f MB\n", $3 / 1048576 }
        /^Segment __DATA/  { data += $3 }
        END { if (data) printf "  __DATA                %.1f MB\n", data / 1048576 }'
}

case "${WHICH}" in
    android) measure_android ;;
    ios) measure_ios ;;
    all) measure_android; echo; measure_ios ;;
    *) echo "usage: $0 [android|ios|all]" >&2; exit 1 ;;
esac
