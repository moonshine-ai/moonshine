#!/usr/bin/env bash
set -euxo pipefail

VERSION="0.0.70"

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
	if [[ "${PLATFORM}" == "windows" ]]; then
		continue
	fi
	# The portable C++ example is a flat folder of source files (no per-project
	# subdirectories), so it is shipped as one archive below rather than via the
	# per-project loop.
	if [[ "${PLATFORM}" == "c++" ]]; then
		continue
	fi
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

# Ship the portable C++ example as a single archive that extracts to a `c++/`
# folder containing the sources, README, and download-library.sh helper.
CPP_DIR="${EXAMPLES_DIR}/c++"
if [[ -d "${CPP_DIR}" ]]; then
	TAR_NAME="cpp-examples.tar.gz"
	TAR_PATH="${TMPDIR:-/tmp}/${TAR_NAME}"
	rm -f "${TAR_PATH}"
	tar -czf "${TAR_PATH}" -C "${EXAMPLES_DIR}" "c++"
	gh release upload "v${VERSION}" "${TAR_PATH}" --clobber
	rm -f "${TAR_PATH}"
fi
