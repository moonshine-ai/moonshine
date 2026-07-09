#!/bin/bash -ex

# Runs the Python module tests (python/tests/test_modules.py), which drive
# the __main__ sections of the most significant moonshine_voice modules
# against a freshly built wheel.
#
# Usage:
#   ./scripts/test-python.sh                Build the wheel first, then test.
#   ./scripts/test-python.sh --skip-build   Reuse the wheel already in
#                                           python/dist/ (faster iteration).

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT_DIR=$(dirname $SCRIPTS_DIR)
PYTHON_DIR=${REPO_ROOT_DIR}/python

if [[ "$1" != "--skip-build" ]]; then
    ${SCRIPTS_DIR}/build-pip.sh
fi

WHEEL=$(ls ${PYTHON_DIR}/dist/moonshine_voice-*.whl | head -n 1)
if [[ -z "${WHEEL}" ]]; then
    echo "No wheel found in ${PYTHON_DIR}/dist/. Run scripts/build-pip.sh first."
    exit 1
fi

WORK_DIR=$(mktemp -d)
cleanup() {
    rm -rf "${WORK_DIR}"
}
trap cleanup EXIT

# A throwaway venv so the modules are tested against exactly the wheel that
# will be uploaded, not whatever happens to be installed on this machine.
python3 -m venv "${WORK_DIR}/venv"
source "${WORK_DIR}/venv/bin/activate"
pip install --upgrade pip
pip install "${WHEEL}"
pip install -r "${PYTHON_DIR}/tests/requirements.txt"

pytest -v "${PYTHON_DIR}/tests/test_modules.py"
