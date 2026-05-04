#!/usr/bin/env bash
set -euxo pipefail

VERSION="0.0.60"

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT_DIR="$(dirname "${SCRIPTS_DIR}")"

"${REPO_ROOT_DIR}/scripts/test-examples.sh" --local-examples

cd "${REPO_ROOT_DIR}"

# Check if the GitHub release exists; create it if missing
if ! gh release view "v${VERSION}" >/dev/null 2>&1; then
	gh release create "v${VERSION}" --title "v${VERSION}" --notes "Release v${VERSION}"
fi

EXAMPLES_DIR="${REPO_ROOT_DIR}/examples"

for PLATFORM_PATH in "${EXAMPLES_DIR}"/*; do
	[[ -d "${PLATFORM_PATH}" ]] || continue
	PLATFORM="$(basename "${PLATFORM_PATH}")"
	for PROJECT_PATH in "${PLATFORM_PATH}"/*; do
		[[ -d "${PROJECT_PATH}" ]] || continue
		NAME="$(basename "${PROJECT_PATH}")"
		TAR_NAME="${PLATFORM}-${NAME}.tar.gz"
		TAR_PATH="${TMPDIR:-/tmp}/${TAR_NAME}"
		rm -f "${TAR_PATH}"
		tar -czf "${TAR_PATH}" -C "${PLATFORM_PATH}" "${NAME}"
		gh release upload "v${VERSION}" "${TAR_PATH}" --clobber
		rm -f "${TAR_PATH}"
	done
done
