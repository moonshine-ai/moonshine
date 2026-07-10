#! /bin/bash -ex

# Expected to be run on macOS.
#
# Usage:
#   ./scripts/build-all-platforms.sh
#
# Environment:
#   LINUX_CLOUD_HOST       - SSH host for Linux cloud
#   LINUX_CLOUD_INSTANCE   - GCP instance name for the Linux VM (optional)
#   LINUX_CLOUD_ZONE       - GCP zone for the Linux VM (e.g. us-central1-b)
#   LINUX_CLOUD_PROJECT    - GCP project ID for the Linux VM
#   WINDOWS_CLOUD_USER     - SSH user for Windows cloud
#   WINDOWS_CLOUD_HOST     - SSH host for Windows cloud
#   WINDOWS_CLOUD_INSTANCE - GCP instance name for the Windows VM (optional)
#   WINDOWS_CLOUD_ZONE     - GCP zone for the Windows VM (e.g. us-central1-b)
#   WINDOWS_CLOUD_PROJECT  - GCP project ID for the Windows VM
#
# When the LINUX_CLOUD_INSTANCE / WINDOWS_CLOUD_INSTANCE variables are set the
# script will start the corresponding GCP VM before connecting and stop it
# again on exit (including on error) to minimize compute costs.

# Resume a GCP compute instance and wait for SSH to become available.
gcp_resume_instance() {
    local instance="$1"
    local zone="$2"
    local project="$3"
    local ssh_target="$4"

    echo "Resuming GCP instance ${instance} in ${zone} (project ${project})..."
    gcloud compute instances resume "${instance}" \
        --zone="${zone}" \
        --project="${project}"

    echo "Waiting for SSH on ${ssh_target} to be ready..."
    local attempt=0
    until ssh -o BatchMode=yes \
              -o ConnectTimeout=5 \
              -o StrictHostKeyChecking=accept-new \
              "${ssh_target}" exit 2>/dev/null; do
        attempt=$((attempt + 1))
        if [ ${attempt} -ge 60 ]; then
            echo "Timed out waiting for SSH on ${ssh_target}." >&2
            return 1
        fi
        sleep 5
    done
    echo "SSH on ${ssh_target} is ready."
}

# Suspend a GCP compute instance. Failures here are reported but do not abort
# cleanup of any other instances.
gcp_suspend_instance() {
    local instance="$1"
    local zone="$2"
    local project="$3"

    echo "Suspending GCP instance ${instance} in ${zone} (project ${project})..."
    gcloud compute instances suspend "${instance}" \
        --zone="${zone}" \
        --project="${project}" \
        || echo "Warning: failed to suspend ${instance}." >&2
}

cleanup() {
    local exit_code=$?
    set +e
    if [ -n "${LINUX_CLOUD_INSTANCE:-}" ]; then
        gcp_suspend_instance \
            "${LINUX_CLOUD_INSTANCE}" \
            "${LINUX_CLOUD_ZONE}" \
            "${LINUX_CLOUD_PROJECT}"
    fi
    if [ -n "${WINDOWS_CLOUD_INSTANCE:-}" ]; then
        gcp_suspend_instance \
            "${WINDOWS_CLOUD_INSTANCE}" \
            "${WINDOWS_CLOUD_ZONE}" \
            "${WINDOWS_CLOUD_PROJECT}"
    fi
    exit ${exit_code}
}
# All imperative work lives inside main() so that bash parses the entire
# script before it starts executing the long-running build steps. Without
# this, editing/saving the file mid-run shifts bash's byte offsets and can
# corrupt an in-flight run (e.g. turning "exit 1" into "xit 1").
main() {
    if [[ "$OSTYPE" != "darwin"* ]]; then
        echo "This script is expected to be run on macOS."
        exit 1
    fi

    SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
    REPO_ROOT_DIR=$(dirname $SCRIPTS_DIR)

    if [ -f "${REPO_ROOT_DIR}/.env" ]; then
        set -o allexport
        source "${REPO_ROOT_DIR}/.env"
        set +o allexport
    fi

    trap cleanup EXIT

    cd ${REPO_ROOT_DIR}
    scripts/test-core.sh
    scripts/test-python.sh
    scripts/test-docs.sh --skip-build
    scripts/build-swift.sh
    scripts/publish-swift.sh
    # Run the Android instrumentation tests on a local arm64 emulator before
    # publishing the AAR. x86_64 coverage runs on the Linux cloud host below
    # (Apple Silicon can't run an x86_64 emulator). Override the AVD name with
    # ANDROID_ARM64_AVD if yours differs.
    scripts/test-android.sh --avd "${ANDROID_ARM64_AVD:-moonshine_api26_arm64}"
    scripts/build-android.sh publish
    scripts/build-pip.sh upload
    scripts/build-pip-docker.sh
    scripts/publish-binary.sh upload
    scripts/publish-examples.sh

    if [ -n "${LINUX_CLOUD_INSTANCE:-}" ]; then
        gcp_resume_instance \
            "${LINUX_CLOUD_INSTANCE}" \
            "${LINUX_CLOUD_ZONE}" \
            "${LINUX_CLOUD_PROJECT}" \
            "${LINUX_CLOUD_HOST}"
    fi

    # The Linux host (LINUX_CLOUD_HOST in .env) is x86_64, so it runs the x86_64
    # Android instrumentation tests (Apple Silicon can't run an x86_64 emulator).
    # One-time host setup via scripts/setup-android-ci.sh: Android SDK +
    # platform-tools + emulator, an x86_64 system image, KVM, an AVD named
    # moonshine_api26_x86_64 (override with ANDROID_X86_64_AVD), and JAVA_HOME/
    # ANDROID_HOME exported in ~/.bashrc (sourced by non-interactive ssh). The
    # checkout path comes from LINUX_CLOUD_REPO_PATH. LINUX_CLOUD_REPO_PATH and the
    # AVD default are expanded here on the local side before the command is sent.
    ssh ${LINUX_CLOUD_HOST} "cd '${LINUX_CLOUD_REPO_PATH}' \
      && git pull origin main \
      && scripts/test-core.sh \
      && scripts/test-android.sh --avd '${ANDROID_X86_64_AVD:-moonshine_api26_x86_64}' \
      && scripts/publish-binary.sh upload" || exit 1

    ssh -p ${RPI_CLOUD_PORT} ${RPI_CLOUD_HOST} 'cd moonshine \
      && git pull origin main \
      && scripts/test-core.sh \
      && scripts/build-pip.sh upload \
      && scripts/publish-binary.sh upload' || exit 1

    if [ -n "${WINDOWS_CLOUD_INSTANCE:-}" ]; then
        gcp_resume_instance \
            "${WINDOWS_CLOUD_INSTANCE}" \
            "${WINDOWS_CLOUD_ZONE}" \
            "${WINDOWS_CLOUD_PROJECT}" \
            "${WINDOWS_CLOUD_USER}@${WINDOWS_CLOUD_HOST}"
    fi

    ssh ${WINDOWS_CLOUD_USER}@${WINDOWS_CLOUD_HOST} 'cd moonshine `
      ; git pull origin main `
      ; scripts/test-core.bat `
      ; scripts/publish-binary.bat upload `
      ; scripts/publish-examples.bat upload `
      ; scripts/build-pip.bat upload' \
      || exit 1
}

main "$@"
