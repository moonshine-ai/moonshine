#!/usr/bin/env bash
#
# Fail if an Apple static library or xcframework exports anything other than
# the moonshine_* C API. scripts/prelink-apple-static-library.sh is what makes
# that true (issue #235: the framework used to re-export the bundled ONNX
# Runtime and capture other libraries' references to it); this is the
# independent check on the artifact that gets zipped and shipped, run from
# scripts/build-swift.sh after -create-xcframework.
#
# Usage:
#   scripts/check-framework-exports.sh path/to/Moonshine.xcframework
#   scripts/check-framework-exports.sh path/to/libmoonshine.a [...]
set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "usage: $0 <xcframework or static library> [...]" >&2
  exit 2
fi

libs=()
for arg in "$@"; do
  if [[ -d "${arg}" ]]; then
    while IFS= read -r lib; do libs+=("${lib}"); done \
      < <(find "${arg}" -name '*.a' -type f | sort)
  else
    libs+=("${arg}")
  fi
done
if [[ ${#libs[@]} -eq 0 ]]; then
  echo "check-framework-exports: no static libraries found under $*" >&2
  exit 1
fi

status=0
for lib in "${libs[@]}"; do
  archs="$(lipo -archs "${lib}")"
  leaked="$(nm -gU "${lib}" 2>/dev/null | awk '!/:$/ && NF { print $NF }' | grep -v '^_moonshine_' | sort -u || true)"
  unresolved="$(nm -u "${lib}" 2>/dev/null | awk '!/:$/ && NF { print $NF }' | grep '^_Ort' | sort -u || true)"
  if [[ -z "${leaked}" && -z "${unresolved}" ]]; then
    echo "ok   ${lib} (${archs// /, })"
    continue
  fi
  status=1
  echo "FAIL ${lib} (${archs// /, })"
  if [[ -n "${leaked}" ]]; then
    echo "  exports $(echo "${leaked}" | wc -l | tr -d ' ') symbols outside moonshine_*, e.g.:"
    # sed rather than head: head closing the pipe early trips pipefail.
    echo "${leaked}" | sed -n '1,10s/^/    /p'
  fi
  if [[ -n "${unresolved}" ]]; then
    echo "  unresolved ONNX Runtime references (would bind to another copy in the app):"
    echo "${unresolved}" | sed 's/^/    /'
  fi
done
exit "${status}"
