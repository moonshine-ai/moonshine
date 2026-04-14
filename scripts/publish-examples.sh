#!/usr/bin/env bash -ex

VERSION="0.0.56"

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT_DIR="$(dirname "${SCRIPTS_DIR}")"

${REPO_ROOT_DIR}/scripts/test-examples.sh --local-examples

cd ${REPO_ROOT_DIR}

# Check if the GitHub release exists; create it if missing
if ! gh release view v$VERSION >/dev/null 2>&1; then
	gh release create v$VERSION --title "v$VERSION" --notes "Release v$VERSION"
fi

gh release upload v$VERSION $TAR_NAME --clobber

cd ${REPO_ROOT_DIR}/examples

for EXAMPLE_DIR in *; do
	if [ -d "$EXAMPLE_DIR" ]; then
		TAR_NAME=${EXAMPLE_DIR}-examples.tar.gz
		cd ${EXAMPLE_DIR}
		tar -czvf ${TAR_NAME} *
		gh release upload v$VERSION ${TAR_NAME} --clobber
		rm -rf ${TAR_NAME}
		cd ..
	fi
done
