#!/bin/bash -ex

VERSION=0.0.71

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT_DIR=$(dirname $SCRIPTS_DIR)

docker build --platform linux/amd64 -t moonshine-ubuntu-amd64 .
docker build --platform linux/arm64 -t moonshine-ubuntu-arm64 .

docker run --rm -v ${REPO_ROOT_DIR}:/home/user/moonshine moonshine-ubuntu-amd64 \
	/bin/bash -c "cd /home/user/moonshine && scripts/build-pip.sh upload"

docker run --rm -v ${REPO_ROOT_DIR}:/home/user/moonshine moonshine-ubuntu-arm64 \
	/bin/bash -c "cd /home/user/moonshine && scripts/build-pip.sh upload"

# Build BOTH Linux C++ library archives (moonshine-voice-linux-x86_64.tar.gz and
# moonshine-voice-linux-arm64.tar.gz) inside their pinned Debian bookworm Docker
# containers. Building in the container (rather than natively on a build host) is
# what keeps the resulting libmoonshine.so's glibc floor low and portable: the
# base image (python:3.12-slim-bookworm) is deliberately old-glibc, so the .so
# only references glibc symbols available on any current distro. Building the
# x86_64 archive natively on a bleeding-edge host baked in a GLIBC_2.43
# requirement that no released distro satisfies (see issue #206); doing it here
# keeps x86_64 in lockstep with the already-working arm64 archive.
#
# publish-binary.sh (without the `upload` argument) leaves the tarball in the
# repo root, which is bind-mounted, so it is visible on the host for the upload
# steps below. Each run does a clean rebuild (rm -rf build) for its own arch, so
# running them sequentially does not cross-contaminate. gh is not installed or
# authenticated inside the container, so we upload from the host, which already
# has an authenticated gh.
docker run --rm -v ${REPO_ROOT_DIR}:/home/user/moonshine moonshine-ubuntu-amd64 \
	/bin/bash -c "cd /home/user/moonshine && scripts/publish-binary.sh"

docker run --rm -v ${REPO_ROOT_DIR}:/home/user/moonshine moonshine-ubuntu-arm64 \
	/bin/bash -c "cd /home/user/moonshine && scripts/publish-binary.sh"

"${SCRIPTS_DIR}/gh-upload-retry.sh" "${VERSION}" \
	"${REPO_ROOT_DIR}/moonshine-voice-linux-x86_64.tar.gz" \
	"${REPO_ROOT_DIR}/moonshine-voice-linux-arm64.tar.gz"
