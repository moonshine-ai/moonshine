#!/bin/bash -ex

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT_DIR=$(dirname $SCRIPTS_DIR)

docker build --platform linux/amd64 -t moonshine-ubuntu-amd64 .
docker build --platform linux/arm64 -t moonshine-ubuntu-arm64 .

docker run --rm -v ${REPO_ROOT_DIR}:/home/user/moonshine moonshine-ubuntu-amd64 \
	/bin/bash -c "cd /home/user/moonshine && scripts/build-pip.sh upload"

docker run --rm -v ${REPO_ROOT_DIR}:/home/user/moonshine moonshine-ubuntu-arm64 \
	/bin/bash -c "cd /home/user/moonshine && scripts/build-pip.sh upload"
