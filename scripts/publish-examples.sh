#!/usr/bin/env bash
set -euxo pipefail

VERSION="0.1.1"

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT_DIR="$(dirname "${SCRIPTS_DIR}")"

# Arguments:
#   upload - attach the example archives to the GitHub release. Without it the
#            examples are still built, tested and packaged, but nothing is
#            uploaded. Matches scripts/publish-examples.bat, which gates on the
#            same word.
DO_UPLOAD=""
case "${1:-}" in
"") ;;
upload) DO_UPLOAD=1 ;;
*)
	echo "Unknown argument: '${1}' (expected one of: <none>, upload)" >&2
	exit 1
	;;
esac

# The examples pin an exact library version: the Android ones resolve
# moonshine-voice from Maven Central, the iOS ones resolve moonshine-swift from
# GitHub, and the C++ one downloads this release's library archive from GitHub
# Releases. During a release those are already published by the time this runs, so
# --local-examples (new example sources, published library) is the right test.
# Without `upload` nothing has been published, so the same run would fail
# resolving a version that does not exist yet; --local-library additionally points
# the examples at this checkout's AAR (~/.m2), swift package and library archive.
if [[ -n "${DO_UPLOAD}" ]]; then
	EXAMPLES_ARGS=(--local-examples)
else
	EXAMPLES_ARGS=(--local-library)
fi

"${REPO_ROOT_DIR}/scripts/test-examples.sh" "${EXAMPLES_ARGS[@]}"

cd "${REPO_ROOT_DIR}"

# Check if the GitHub release exists; create it if missing
if [[ -n "${DO_UPLOAD}" ]] && ! gh release view "v${VERSION}" >/dev/null 2>&1; then
	gh release create "v${VERSION}" --title "v${VERSION}" --notes "Release v${VERSION}"
fi

EXAMPLES_DIR="${REPO_ROOT_DIR}/examples"

# Web demos are packaged as self-contained archives (demo + assets + serve.mjs),
# not as bare project folders. See scripts/web-example-archive.sh.
WEB_ARCHIVE_SCRIPT="${SCRIPTS_DIR}/web-example-archive.sh"

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
	if [[ "${PLATFORM}" == "web" ]]; then
		while IFS= read -r NAME; do
			[[ -z "${NAME}" ]] && continue
			TAR_NAME="web-${NAME}.tar.gz"
			TAR_PATH="${TMPDIR:-/tmp}/${TAR_NAME}"
			rm -f "${TAR_PATH}"
			"${WEB_ARCHIVE_SCRIPT}" pack "${NAME}" "${TAR_PATH}"
			if [[ -n "${DO_UPLOAD}" ]]; then
				gh release upload "v${VERSION}" "${TAR_PATH}" --clobber
			fi
			rm -f "${TAR_PATH}"
		done < <("${WEB_ARCHIVE_SCRIPT}" list)
		continue
	fi
	for PROJECT_PATH in "${PLATFORM_PATH}"/*; do
		[[ -d "${PROJECT_PATH}" ]] || continue
		NAME="$(basename "${PROJECT_PATH}")"
		# Internal latency harness, not a user-facing example (uses in-tree
		# ../../../swift). Exercised by scripts/test-mobile-latency.sh instead.
		if [[ "${PLATFORM}" == "ios" && "${NAME}" == "StreamingLatency" ]]; then
			continue
		fi
		TAR_NAME="${PLATFORM}-${NAME}.tar.gz"
		TAR_PATH="${TMPDIR:-/tmp}/${TAR_NAME}"
		rm -f "${TAR_PATH}"
		tar -czf "${TAR_PATH}" -C "${PLATFORM_PATH}" "${NAME}"
		if [[ -n "${DO_UPLOAD}" ]]; then
			gh release upload "v${VERSION}" "${TAR_PATH}" --clobber
		fi
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
	if [[ -n "${DO_UPLOAD}" ]]; then
		gh release upload "v${VERSION}" "${TAR_PATH}" --clobber
	fi
	rm -f "${TAR_PATH}"
fi

if [[ -z "${DO_UPLOAD}" ]]; then
	echo "No 'upload' argument; example archives were built and discarded."
fi
