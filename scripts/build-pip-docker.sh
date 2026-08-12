#!/bin/bash -ex

VERSION=0.1.2

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT_DIR=$(dirname $SCRIPTS_DIR)

# Arguments:
#   upload - publish the wheels to PyPI and the C++ archives to the GitHub
#            release. Without it the wheels and tarballs are built inside the
#            containers and left on the (bind-mounted) host, which is the dry-run
#            form used by scripts/build-all-platforms.sh.
DO_UPLOAD=""
for arg in "$@"; do
	case "$arg" in
	upload) DO_UPLOAD=1 ;;
	*)
		echo "Unknown argument: '$arg'" >&2
		exit 1
		;;
	esac
done

if [[ -n "${DO_UPLOAD}" ]]; then
	PIP_ARGS="upload"
else
	PIP_ARGS=""
fi

# The build context has to stay the repo root: the Dockerfile copies .pypirc
# from there. Passing it explicitly also removes the old dependency on the
# caller's working directory.
docker build --platform linux/amd64 -f "${SCRIPTS_DIR}/Dockerfile" \
	-t moonshine-ubuntu-amd64 "${REPO_ROOT_DIR}"
docker build --platform linux/arm64 -f "${SCRIPTS_DIR}/Dockerfile" \
	-t moonshine-ubuntu-arm64 "${REPO_ROOT_DIR}"
# An older base for the arm64 wheel only. Raspberry Pi OS bullseye is still
# widely installed and has glibc 2.31, so a wheel built on bookworm will not
# install there at all -- and a Pi is exactly where an arm64 Linux wheel gets
# used. There is no matching x86_64 image because that side has only ever
# shipped a 2_34 wheel.
docker build --platform linux/arm64 \
	--build-arg BASE_IMAGE=python:3.12-slim-bullseye \
	-f "${SCRIPTS_DIR}/Dockerfile" \
	-t moonshine-debian-bullseye-arm64 "${REPO_ROOT_DIR}"

docker run --rm -v ${REPO_ROOT_DIR}:/home/user/moonshine moonshine-ubuntu-amd64 \
	/bin/bash -c "cd /home/user/moonshine && scripts/build-pip.sh ${PIP_ARGS}"

docker run --rm -v ${REPO_ROOT_DIR}:/home/user/moonshine moonshine-ubuntu-arm64 \
	/bin/bash -c "cd /home/user/moonshine && scripts/build-pip.sh ${PIP_ARGS}"

# Last of the wheel builds, because each one clears dist/ before it starts and
# the Pi stage tests whatever is left there: the Pi runs bullseye, so this is the
# only one of these wheels it can install.
docker run --rm -v ${REPO_ROOT_DIR}:/home/user/moonshine \
	-e MOONSHINE_MANYLINUX_VERSION=2_31 moonshine-debian-bullseye-arm64 \
	/bin/bash -c "cd /home/user/moonshine && scripts/build-pip.sh ${PIP_ARGS}"

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

if [[ -n "${DO_UPLOAD}" ]]; then
	"${SCRIPTS_DIR}/gh-upload-retry.sh" "${VERSION}" \
		"${REPO_ROOT_DIR}/moonshine-voice-linux-x86_64.tar.gz" \
		"${REPO_ROOT_DIR}/moonshine-voice-linux-arm64.tar.gz"
else
	echo "No 'upload' argument; leaving the Linux archives in ${REPO_ROOT_DIR}:"
	ls -la "${REPO_ROOT_DIR}"/moonshine-voice-linux-*.tar.gz
fi
