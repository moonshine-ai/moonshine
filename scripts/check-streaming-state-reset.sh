#!/usr/bin/env bash
#
# Every data member of MoonshineStreamingState belongs to one audio stream,
# and MoonshineStreamingState::reset() is the only thing that separates one
# stream from the next. A field that is added to the struct but not cleared in
# reset() leaks one stream's audio into the following one. That is the shape
# of GitHub issues #229 and #231, and the per-layer encoder cache added for
# #236 is exactly the kind of field this is about, so the rule is enforced
# here rather than remembered.
#
# The check is textual: each member declared in the struct (in
# core/moonshine-streaming-model.h) must appear as the target of an assignment,
# `.clear()`, `.assign(...)`, or `.resize(...)` inside the body of
# MoonshineStreamingState::reset in core/moonshine-streaming-model.cpp.
# Registered as the `check-streaming-state-reset` ctest.
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT_DIR="$(dirname "${SCRIPTS_DIR}")"
HEADER="${REPO_ROOT_DIR}/core/moonshine-streaming-model.h"
SOURCE="${REPO_ROOT_DIR}/core/moonshine-streaming-model.cpp"

# Struct body: from the declaration to the first line that is just "};".
struct_body="$(awk '
  /^struct MoonshineStreamingState \{/ { inside = 1; next }
  inside && /^\};/ { exit }
  inside { print }
' "${HEADER}")"
if [[ -z "${struct_body}" ]]; then
  echo "check-streaming-state-reset: could not find struct MoonshineStreamingState in ${HEADER}" >&2
  exit 2
fi

# Member names: the last identifier before ';' on a declaration line, skipping
# comments, blank lines, and the reset() declaration itself.
members=()
while IFS= read -r line; do
  stripped="$(printf '%s' "${line}" | sed -E 's://.*$::')"
  [[ "${stripped}" =~ \; ]] || continue
  [[ "${stripped}" =~ \( ]] && continue
  name="$(printf '%s' "${stripped}" | sed -E 's/^.*[[:space:]*&]([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*(\[[^]]*\])?[[:space:]]*(=[^;]*)?;.*$/\1/')"
  [[ -n "${name}" && "${name}" != "${stripped}" ]] && members+=("${name}")
done <<< "${struct_body}"

if [[ ${#members[@]} -eq 0 ]]; then
  echo "check-streaming-state-reset: no members parsed from MoonshineStreamingState" >&2
  exit 2
fi

reset_body="$(awk '
  /^void MoonshineStreamingState::reset\(/ { inside = 1; next }
  inside && /^\}/ { exit }
  inside { print }
' "${SOURCE}")"
if [[ -z "${reset_body}" ]]; then
  echo "check-streaming-state-reset: could not find MoonshineStreamingState::reset in ${SOURCE}" >&2
  exit 2
fi

missing=()
for m in "${members[@]}"; do
  if ! printf '%s\n' "${reset_body}" | grep -Eq "(^|[^A-Za-z0-9_.])${m}[[:space:]]*(=[^=]|\.clear\(|\.assign\(|\.resize\()"; then
    missing+=("${m}")
  fi
done

if [[ ${#missing[@]} -gt 0 ]]; then
  echo "check-streaming-state-reset: FAIL" >&2
  echo "  MoonshineStreamingState::reset() does not clear:" >&2
  for m in "${missing[@]}"; do
    echo "    ${m}" >&2
  done
  echo "  Every field of the state is per-stream; clear it in reset() (see" >&2
  echo "  the comment above reset() in core/moonshine-streaming-model.h)." >&2
  exit 1
fi

echo "check-streaming-state-reset: ok (${#members[@]} members cleared in reset())"
