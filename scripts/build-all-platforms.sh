#! /bin/bash -ex

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT_DIR=$(dirname $SCRIPTS_DIR)

if [ -f "${REPO_ROOT_DIR}/.env" ]; then
    set -o allexport
    source "${REPO_ROOT_DIR}/.env"
    set +o allexport
fi

cd ${REPO_ROOT_DIR}
scripts/run-core-tests.sh
scripts/build-swift.sh
scripts/publish-swift.sh
scripts/publish-android.sh
scripts/build-pip.sh upload
scripts/build-pip-docker.sh
scripts/publish-binary.sh upload

ssh ${LINUX_CLOUD_HOST} 'cd moonshine \
  && git pull origin main \
  && scripts/run-core-tests.sh \
  && scripts/publish-binary.sh upload' || exit 1

ssh -p ${RPI_CLOUD_PORT} ${RPI_CLOUD_HOST} 'cd moonshine \
  && git pull origin main \
  && scripts/run-core-tests.sh \
  && scripts/build-pip.sh upload \
  && scripts/publish-binary.sh upload' || exit 1

ssh ${WINDOWS_CLOUD_USER}@${WINDOWS_CLOUD_HOST} 'cd moonshine `
  ; git pull origin main `
  ; scripts/run-core-tests.bat `
  ; scripts/publish-binary.bat upload `
  ; scripts/build-pip.bat upload' \
  || exit 1