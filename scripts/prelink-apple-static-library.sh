#!/usr/bin/env bash
#
# Collapse the static archives that make up moonshine.framework into a single
# archive whose only global symbols are the public C API (moonshine_*).
#
# Why: `libtool -static` just concatenates archive members, so the framework
# used to re-export every global from the bundled ONNX Runtime
# (_OrtGetApiBase, _OrtSessionOptionsAppendExecutionProvider_CPU and about
# ten thousand others). An app that also linked a different ONNX Runtime, for
# example through sherpa-onnx, could have its references bound to Moonshine's
# minimal build and fail with "ONNX format model is not supported in this
# build" (GitHub issue #235). The same applied to utf8proc, kaldi-native-fbank
# and the other vendored code.
#
# How: for each architecture slice,
#
#   1. The ONNX Runtime archive is relinked with `ld -r` into one object. Its
#      two entry points that Moonshine calls by name are given moonshine_ort_*
#      aliases, and only those aliases stay global; every other symbol becomes
#      static.
#   2. Every other archive (Moonshine's own code, the vendored tokenizer and
#      TTS code, and core/moonshine-ort-shim.c, which forwards OrtGetApiBase
#      to the alias) is relinked with `ld -r` into a second object, exporting
#      only the moonshine_* symbols it defines.
#
# Two objects rather than one because the vendored ONNX Runtime archive is
# itself an `ld -r` product without MH_SUBSECTIONS_VIA_SYMBOLS; merging
# Moonshine's code into it would make the whole framework ineligible for
# -dead_strip in the app, costing several megabytes for apps that only use
# part of the library. Kept apart, Moonshine's object keeps the flag.
#
# The script checks its own result: it fails if any slice still exports a
# symbol outside moonshine_* or still has an unresolved _Ort* reference.
#
# Usage:
#   prelink-apple-static-library.sh --platform macos|ios|ios-simulator \
#       --min-version 13.0 --onnxruntime libonnxruntime.a --output out.a \
#       archive.a [archive.a ...]
#
# Called from core/CMakeLists.txt for IOS and MOONSHINE_BUILD_SWIFT builds.
set -euo pipefail

PLATFORM=""
MIN_VERSION=""
ORT_ARCHIVE=""
OUTPUT=""
ARCHIVES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --platform) PLATFORM="$2"; shift 2 ;;
    --min-version) MIN_VERSION="$2"; shift 2 ;;
    --onnxruntime) ORT_ARCHIVE="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    --*) echo "prelink-apple-static-library: unknown option $1" >&2; exit 2 ;;
    *) ARCHIVES+=("$1"); shift ;;
  esac
done

case "${PLATFORM}" in
  macos) SDK=macosx; : "${MIN_VERSION:=13.0}" ;;
  ios) SDK=iphoneos; : "${MIN_VERSION:=15.1}" ;;
  ios-simulator) SDK=iphonesimulator; : "${MIN_VERSION:=15.1}" ;;
  *) echo "prelink-apple-static-library: --platform must be macos, ios or ios-simulator" >&2; exit 2 ;;
esac
if [[ -z "${ORT_ARCHIVE}" || -z "${OUTPUT}" || ${#ARCHIVES[@]} -eq 0 ]]; then
  echo "prelink-apple-static-library: need --onnxruntime, --output and at least one archive" >&2
  exit 2
fi
for f in "${ORT_ARCHIVE}" "${ARCHIVES[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "prelink-apple-static-library: missing input $f" >&2
    exit 1
  fi
done

SDK_VERSION="$(xcrun --sdk "${SDK}" --show-sdk-version)"
LD="$(xcrun --sdk "${SDK}" --find ld)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/moonshine-prelink.XXXXXX")"
trap 'rm -rf "${WORK}"' EXIT

# The ONNX Runtime entry points Moonshine calls by name, and the private
# names they get inside the framework (see core/moonshine-ort-shim.c).
ORT_ALIASES=(
  "_OrtGetApiBase _moonshine_ort_get_api_base"
  "_OrtSessionOptionsAppendExecutionProvider_CPU _moonshine_ort_append_cpu_ep"
)
ALIAS_ARGS=()
: > "${WORK}/ort-exports.txt"
for pair in "${ORT_ALIASES[@]}"; do
  set -- ${pair}
  ALIAS_ARGS+=(-alias "$1" "$2")
  echo "$2" >> "${WORK}/ort-exports.txt"
done

# Public API: every extern "C" moonshine_* symbol defined by the first-party
# archives. Derived from the inputs rather than from a hand-kept list so a
# new C API function cannot be hidden by mistake.
symbol_names() {  # nm output -> bare names, dropping archive/member headers
  awk '!/:$/ && NF { print $NF }'
}
nm -gU "${ARCHIVES[@]}" 2>/dev/null | symbol_names | grep '^_moonshine_' | sort -u > "${WORK}/api-exports.txt"
if [[ ! -s "${WORK}/api-exports.txt" ]]; then
  echo "prelink-apple-static-library: no moonshine_* symbols found in inputs" >&2
  exit 1
fi

# Build one thin archive per architecture, then recombine.
ARCHS="$(lipo -archs "${ARCHIVES[0]}")"
THIN_OUTPUTS=()
for arch in ${ARCHS}; do
  common=(-r -arch "${arch}" -platform_version "${PLATFORM}" "${MIN_VERSION}" "${SDK_VERSION}")

  # 1. ONNX Runtime: alias the two entry points, hide everything. The vendored
  # archive may carry a newer minimum OS than ours (ld warns otherwise), so
  # relink it at its own.
  ort_min="$(otool -l -arch "${arch}" "${ORT_ARCHIVE}" 2>/dev/null | awk '/^ *minos /{print $2; exit}')"
  "${LD}" -r -arch "${arch}" -platform_version "${PLATFORM}" "${ort_min:-${MIN_VERSION}}" "${SDK_VERSION}" \
    -all_load "${ORT_ARCHIVE}" \
    "${ALIAS_ARGS[@]}" \
    -exported_symbols_list "${WORK}/ort-exports.txt" \
    -o "${WORK}/onnxruntime-${arch}.o"

  # 2. Everything else: export the C API only.
  "${LD}" "${common[@]}" -all_load "${ARCHIVES[@]}" \
    -exported_symbols_list "${WORK}/api-exports.txt" \
    -o "${WORK}/moonshine-${arch}.o"

  libtool -static -no_warning_for_no_symbols \
    -o "${WORK}/prelinked-${arch}.a" \
    "${WORK}/moonshine-${arch}.o" "${WORK}/onnxruntime-${arch}.o"
  THIN_OUTPUTS+=("${WORK}/prelinked-${arch}.a")
done

if [[ ${#THIN_OUTPUTS[@]} -eq 1 ]]; then
  cp "${THIN_OUTPUTS[0]}" "${OUTPUT}"
else
  lipo -create "${THIN_OUTPUTS[@]}" -output "${OUTPUT}"
fi

# Verify: only moonshine_* globals, and no ONNX Runtime reference left to be
# satisfied by whatever else the app links.
leaked="$(nm -gU "${OUTPUT}" 2>/dev/null | symbol_names | grep -v '^_moonshine_' | sort -u || true)"
if [[ -n "${leaked}" ]]; then
  echo "prelink-apple-static-library: ${OUTPUT} still exports non-moonshine symbols:" >&2
  echo "${leaked}" | sed -n '1,20p' >&2
  exit 1
fi
unresolved="$(nm -u "${OUTPUT}" 2>/dev/null | symbol_names | grep '^_Ort' | sort -u || true)"
if [[ -n "${unresolved}" ]]; then
  echo "prelink-apple-static-library: ${OUTPUT} still references ONNX Runtime by name:" >&2
  echo "${unresolved}" >&2
  echo "Add the symbol to ORT_ALIASES here and a forwarding stub to core/moonshine-ort-shim.c." >&2
  exit 1
fi
echo "prelink-apple-static-library: ${OUTPUT} (${ARCHS// /, }) exports $(wc -l < "${WORK}/api-exports.txt" | tr -d ' ') moonshine_* symbols"
