#!/bin/bash -ex

# Tests that the code in the documentation still works, by executing the
# fenced code blocks in README.md and python/README.md (see
# python/tests/test_docs.py) and running the getting-started notebook end to
# end, all against a freshly built wheel.
#
# The Python module tests (python/tests/test_modules.py) are run separately
# by scripts/test-python.sh.
#
# Usage:
#   ./scripts/test-docs.sh                Build the wheel first, then test.
#   ./scripts/test-docs.sh --skip-build   Reuse the wheel already in
#                                         python/dist/ (faster iteration).

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

# A throwaway venv so the docs are tested against exactly the wheel that will
# be uploaded, not whatever happens to be installed on this machine.
python3 -m venv "${WORK_DIR}/venv"
source "${WORK_DIR}/venv/bin/activate"
pip install --upgrade pip
pip install "${WHEEL}"
pip install -r "${PYTHON_DIR}/tests/requirements.txt"

# Run the notebook from a scratch directory so the files it downloads don't
# end up in the repository.
cp "${PYTHON_DIR}/getting-started-with-moonshine-voice.ipynb" "${WORK_DIR}/"

pytest -v \
    --nbmake --nbmake-timeout=900 \
    "${PYTHON_DIR}/tests/test_docs.py" \
    "${WORK_DIR}/getting-started-with-moonshine-voice.ipynb"
