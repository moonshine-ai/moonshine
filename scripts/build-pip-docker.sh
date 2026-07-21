#!/bin/bash -ex

VERSION=0.0.70

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT_DIR=$(dirname $SCRIPTS_DIR)

docker build --platform linux/amd64 -t moonshine-ubuntu-amd64 .
docker build --platform linux/arm64 -t moonshine-ubuntu-arm64 .

docker run --rm -v ${REPO_ROOT_DIR}:/home/user/moonshine moonshine-ubuntu-amd64 \
	/bin/bash -c "cd /home/user/moonshine && scripts/build-pip.sh upload"

docker run --rm -v ${REPO_ROOT_DIR}:/home/user/moonshine moonshine-ubuntu-arm64 \
	/bin/bash -c "cd /home/user/moonshine && scripts/build-pip.sh upload"

# Build the arm64 Linux C++ library archive (moonshine-voice-linux-arm64.tar.gz)
# in the native-arm64 Docker container. This used to run on the Raspberry Pi
# (see scripts/build-all-platforms.sh stage_pi), which is much slower than the
# arm64 Docker instance on an Apple Silicon host. publish-binary.sh (without the
# `upload` argument) leaves the tarball in the repo root, which is bind-mounted,
# so it is visible on the host for the upload step below. gh is not installed or
# authenticated inside the container, so we upload from the host, which already
# has an authenticated gh.
docker run --rm -v ${REPO_ROOT_DIR}:/home/user/moonshine moonshine-ubuntu-arm64 \
	/bin/bash -c "cd /home/user/moonshine && scripts/publish-binary.sh"

"${SCRIPTS_DIR}/gh-upload-retry.sh" "${VERSION}" \
	"${REPO_ROOT_DIR}/moonshine-voice-linux-arm64.tar.gz"
