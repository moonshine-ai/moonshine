#!/usr/bin/env bash
#
# Format (default) or check (--check) the first-party C++ in core/ against the
# repo's Google-based .clang-format. Vendored/third-party trees are never
# touched. Runs anywhere clang-format is installed (e.g. `brew install
# clang-format` on macOS); no Linux or full build required.
#
# Usage:
#   scripts/format-core.sh            # format in place
#   scripts/format-core.sh --check    # non-zero exit if anything is unformatted
#
# Override the binary with CLANG_FORMAT=/path/to/clang-format.
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT_DIR="$(dirname "${SCRIPTS_DIR}")"
CORE_DIR="${REPO_ROOT_DIR}/core"
CLANG_FORMAT="${CLANG_FORMAT:-clang-format}"

MODE="fix"
if [[ "${1:-}" == "--check" ]]; then
  MODE="check"
elif [[ -n "${1:-}" ]]; then
  echo "usage: $0 [--check]" >&2
  exit 2
fi

if ! command -v "${CLANG_FORMAT}" >/dev/null 2>&1; then
  echo "error: '${CLANG_FORMAT}' not found." >&2
  echo "  Install it (macOS: 'brew install clang-format') or set CLANG_FORMAT." >&2
  exit 1
fi

status=0
while IFS= read -r file; do
  case "${MODE}" in
    check)
      if ! "${CLANG_FORMAT}" --dry-run --Werror "${file}" >/dev/null 2>&1; then
        echo "needs formatting: ${file#"${REPO_ROOT_DIR}"/}"
        status=1
      fi
      ;;
    fix)
      echo "clang-format: ${file#"${REPO_ROOT_DIR}"/}"
      "${CLANG_FORMAT}" -i "${file}"
      ;;
  esac
done < <(
  find "${CORE_DIR}" \
    \( -path "${CORE_DIR}/third-party" \
       -o -path "${CORE_DIR}/cpp-annote" \
       -o -name build \) -prune -o \
    -type f \( \
      -name '*.c' -o \
      -name '*.cc' -o \
      -name '*.cpp' -o \
      -name '*.h' -o \
      -name '*.hpp' \
    \) -print \
  | LC_ALL=C sort
)

if [[ "${MODE}" == "check" && "${status}" -ne 0 ]]; then
  echo "" >&2
  echo "Run 'scripts/format-core.sh' to reformat the files listed above." >&2
fi

exit "${status}"
