#! /bin/bash -ex

# Expected to be run on macOS.
#
# Usage:
#   ./scripts/build-all-platforms.sh [RELEASE_REF]
#
# This builds and publishes a *frozen* release ref (a tag or branch), NOT your
# live working tree or origin/main. That way you can keep editing `main` while
# this long-running build is in flight without corrupting it. Normally you cut
# the ref first with scripts/prepare-release.sh (or scripts/patch-release.sh),
# then run this with no arguments and it builds the most recent v* tag.
#
# The local platforms build from an isolated git worktree checked out at the
# release ref, and each remote host checks out the same ref (instead of pulling
# main), so nothing you push to main afterwards leaks into the release.
#
# Environment:
#   RELEASE_REF            - git ref (tag/branch/sha) to build. Defaults to the
#                            first argument, or the most recent v* tag.
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
    if [ -n "${RELEASE_DIR:-}" ] && [ -d "${RELEASE_DIR}" ]; then
        echo "Removing release worktree ${RELEASE_DIR}..."
        git -C "${REPO_ROOT_DIR}" worktree remove --force "${RELEASE_DIR}" \
            2>/dev/null || rm -rf "${RELEASE_DIR}"
    fi
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

    # Freeze the release to an immutable ref so we can keep working on `main`
    # while this long build runs. RELEASE_REF defaults to the most recent v*
    # tag (what prepare-release.sh / patch-release.sh just pushed); override it
    # via the RELEASE_REF env var or the first argument.
    RELEASE_REF="${RELEASE_REF:-${1:-}}"
    git -C "${REPO_ROOT_DIR}" fetch origin --tags --prune
    if [ -z "${RELEASE_REF}" ]; then
        RELEASE_REF="$(git -C "${REPO_ROOT_DIR}" describe --tags --abbrev=0 --match 'v*')"
    fi
    if ! git -C "${REPO_ROOT_DIR}" rev-parse -q --verify "${RELEASE_REF}^{commit}" >/dev/null; then
        echo "Release ref '${RELEASE_REF}' does not resolve to a commit." >&2
        exit 1
    fi
    echo "Building release ref: ${RELEASE_REF}"

    # Build the local platforms from an isolated worktree checked out at the
    # release ref, so editing files in the main checkout can't corrupt the
    # in-flight build. Sub-scripts derive their repo root from their own
    # location, so running the worktree's copies roots everything in the
    # worktree; .env vars were already exported above and are inherited here.
    RELEASE_DIR="$(cd "${REPO_ROOT_DIR}/.." && pwd)/moonshine-release-${RELEASE_REF//\//-}"
    git -C "${REPO_ROOT_DIR}" worktree remove --force "${RELEASE_DIR}" 2>/dev/null || true
    rm -rf "${RELEASE_DIR}"
    git -C "${REPO_ROOT_DIR}" worktree add --detach "${RELEASE_DIR}" "${RELEASE_REF}"

    cd "${RELEASE_DIR}"
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
      && git fetch origin --tags --prune \
      && git checkout -f '${RELEASE_REF}' \
      && scripts/test-core.sh \
      && scripts/test-android.sh --avd '${ANDROID_X86_64_AVD:-moonshine_api26_x86_64}' \
      && scripts/publish-binary.sh upload" || exit 1

    ssh -p ${RPI_CLOUD_PORT} ${RPI_CLOUD_HOST} "cd moonshine \
      && git fetch origin --tags --prune \
      && git checkout -f '${RELEASE_REF}' \
      && scripts/test-core.sh \
      && scripts/build-pip.sh upload \
      && scripts/publish-binary.sh upload" || exit 1

    if [ -n "${WINDOWS_CLOUD_INSTANCE:-}" ]; then
        gcp_resume_instance \
            "${WINDOWS_CLOUD_INSTANCE}" \
            "${WINDOWS_CLOUD_ZONE}" \
            "${WINDOWS_CLOUD_PROJECT}" \
            "${WINDOWS_CLOUD_USER}@${WINDOWS_CLOUD_HOST}"
    fi

    # Keepalives so a brief network stall doesn't tear down the session. A
    # dropped connection kills the remote build outright, because Windows
    # OpenSSH terminates the session's process tree on disconnect. With these,
    # the client tolerates ~2 minutes (15s * 8) of silence before giving up.
    windows_ssh_opts=(
        -o ServerAliveInterval=15
        -o ServerAliveCountMax=8
        -o TCPKeepAlive=yes
    )

    # The Windows login shell is PowerShell. Check out the release ref first
    # (that also refreshes run-windows-ci.ps1 itself at that ref), then hand off
    # to the orchestrator, which runs each step with heavy, disconnect-surviving
    # logging (see the script header and build-logs/ on the box). Using a single
    # orchestrator also makes the run abort on the first failing step rather
    # than masking it behind the exit code of the last chained command. The
    # RELEASE_REF is expanded locally (via the single-quote break) so PowerShell
    # variables like $LASTEXITCODE stay intact for the remote shell.
    windows_remote_cmd='cd moonshine `
      ; git fetch origin --tags --prune `
      ; git checkout -f '"${RELEASE_REF}"' `
      ; if ($LASTEXITCODE -ne 0) { Write-Host "git checkout failed"; exit 1 } `
      ; pwsh -NoProfile -ExecutionPolicy Bypass -File scripts\run-windows-ci.ps1 -Upload'

    # Transient SSH/network disconnects (not build defects) have killed runs
    # mid-compile. The remote build is a clean rebuild and therefore idempotent,
    # so retry the whole invocation on a connection failure before giving up.
    # ssh exits 255 for transport-level errors (dropped connection, broken
    # pipe); any other non-zero code is the remote command's own exit status,
    # i.e. a genuine build/test failure that retrying won't fix -- fail fast on
    # those. Each attempt leaves a disconnect-surviving log on the box under
    # build-logs/.
    windows_attempts=3
    windows_attempt=1
    while true; do
        echo "Windows build attempt ${windows_attempt}/${windows_attempts}..."
        set +e
        ssh "${windows_ssh_opts[@]}" \
            "${WINDOWS_CLOUD_USER}@${WINDOWS_CLOUD_HOST}" \
            "${windows_remote_cmd}"
        windows_ssh_rc=$?
        set -e

        if [ ${windows_ssh_rc} -eq 0 ]; then
            break
        fi
        if [ ${windows_ssh_rc} -ne 255 ]; then
            echo "Windows build failed (remote exit ${windows_ssh_rc}); not a" \
                 "connection error, not retrying." >&2
            exit 1
        fi
        if [ ${windows_attempt} -ge ${windows_attempts} ]; then
            echo "Windows build aborted after ${windows_attempts} connection" \
                 "failures (ssh exit 255)." >&2
            exit 1
        fi
        windows_attempt=$((windows_attempt + 1))
        echo "SSH connection dropped (exit 255); retrying in 15s..."
        sleep 15
    done
}

main "$@"
