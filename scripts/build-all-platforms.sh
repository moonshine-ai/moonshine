#! /bin/bash -ex

# Expected to be run on macOS.
#
# Usage:
#   ./scripts/build-all-platforms.sh [RELEASE_REF]
#
# This builds and publishes a per-release branch (release-v<version>), NOT your
# live working tree or origin/main, so you can keep editing `main` while this
# long-running build is in flight. Normally you cut the branch first with
# scripts/prepare-release.sh, then run this with no arguments and it builds the
# newest release-v* branch.
#
# The branch is the source of truth. build-all refreshes the v<version> tag to
# the branch HEAD (GitHub Releases / SwiftPM need a tag) -- you never move tags
# by hand. The local platforms build from an isolated git worktree at the branch
# HEAD; each remote host resets to the same branch HEAD. To fold a fix mid-
# release, add a commit to the branch (see scripts/patch-release.sh) and re-run.
#
# Resumable: each stage drops a breadcrumb under .release-state/<version>/ when
# it completes, so re-running the script (e.g. after a failure, or after folding
# in a fix) skips the stages that already finished and picks up where it left
# off. Set RELEASE_FRESH=1 to discard the breadcrumbs and rebuild every stage.
#
# Environment:
#   RELEASE_REF            - release branch (or tag/sha) to build. Defaults to
#                            the first argument, or the newest release-v* branch.
#   RELEASE_FRESH          - if non-empty, ignore/clear resume breadcrumbs and
#                            rebuild every stage.
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

# Per-release resume support: each stage drops a breadcrumb file in STATE_DIR
# when it finishes, so re-running the script skips any stage that already
# completed for the same release ref. Because the release is pinned to an
# immutable ref, a resumed run rebuilds identical code. Set RELEASE_FRESH=1 to
# clear the breadcrumbs and rebuild every stage from scratch.
run_stage() {
    local name="$1"
    shift
    local marker="${STATE_DIR}/${name}.done"
    if [ -f "${marker}" ]; then
        echo "[resume] Skipping stage '${name}' (already completed for ${RELEASE_REF})."
        return 0
    fi
    echo "[stage] ===== ${name}: starting ====="
    "$@"
    touch "${marker}"
    echo "[stage] ===== ${name}: done ====="
}

# The x86_64 Linux cloud host runs the x86_64 Android instrumentation tests
# (Apple Silicon can't run an x86_64 emulator). One-time host setup via
# scripts/setup-android-ci.sh: Android SDK + platform-tools + emulator, an
# x86_64 system image, KVM, an AVD named moonshine_api26_x86_64 (override with
# ANDROID_X86_64_AVD), and JAVA_HOME/ANDROID_HOME exported in ~/.bashrc (sourced
# by non-interactive ssh). The checkout path comes from LINUX_CLOUD_REPO_PATH.
# LINUX_CLOUD_REPO_PATH and the AVD default are expanded here on the local side
# before the command is sent.
#
# NOTE: this host does NOT build the moonshine-voice-linux-x86_64.tar.gz C++
# archive. That is built in the pinned Debian bookworm Docker container in the
# build-pip-docker stage, alongside the arm64 archive, so both share the same
# low, portable glibc floor. Building the x86_64 archive natively here (as we
# used to) baked in a GLIBC_2.43 floor from this host's bleeding-edge glibc that
# no released distro satisfies -- see issue #206.
stage_linux() {
    if [ -n "${LINUX_CLOUD_INSTANCE:-}" ]; then
        gcp_resume_instance \
            "${LINUX_CLOUD_INSTANCE}" \
            "${LINUX_CLOUD_ZONE}" \
            "${LINUX_CLOUD_PROJECT}" \
            "${LINUX_CLOUD_HOST}"
    fi

    ssh ${LINUX_CLOUD_HOST} "cd '${LINUX_CLOUD_REPO_PATH}' \
      && ${REMOTE_GIT_SYNC} \
      && scripts/test-core.sh \
      && scripts/test-android.sh --avd '${ANDROID_X86_64_AVD:-moonshine_api26_x86_64}'" || exit 1
}

# The Raspberry Pi cloud host checks out the release ref and publishes the arm64
# wheel. The arm64 C++ library archive (moonshine-voice-linux-arm64.tar.gz) is
# NOT built here anymore -- it moved to the native-arm64 Docker instance in the
# build-pip-docker stage, which is much faster than the Pi.
stage_pi() {
    ssh -p ${RPI_CLOUD_PORT} ${RPI_CLOUD_HOST} "cd moonshine \
      && ${REMOTE_GIT_SYNC} \
      && scripts/test-core.sh \
      && scripts/build-pip.sh upload" || exit 1
}

# The Windows cloud host runs the CI orchestrator over SSH with
# disconnect-surviving retries.
stage_windows() {
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
    local windows_ssh_opts=(
        -o ServerAliveInterval=15
        -o ServerAliveCountMax=8
        -o TCPKeepAlive=yes
    )

    # The Windows login shell is PowerShell. Sync to the build point first (that
    # also refreshes run-windows-ci.ps1 itself), then hand off to the
    # orchestrator, which runs each step with heavy, disconnect-surviving logging
    # (see the script header and build-logs/ on the box). Using a single
    # orchestrator also makes the run abort on the first failing step rather
    # than masking it behind the exit code of the last chained command. The
    # sync command is expanded locally (via the single-quote break) so PowerShell
    # variables like $LASTEXITCODE stay intact for the remote shell.
    local windows_remote_cmd='cd moonshine `
      ; '"${WIN_GIT_SYNC}"' `
      ; if ($LASTEXITCODE -ne 0) { Write-Host "git sync failed"; exit 1 } `
      ; pwsh -NoProfile -ExecutionPolicy Bypass -File scripts\run-windows-ci.ps1 -Upload'

    # Transient SSH/network disconnects (not build defects) have killed runs
    # mid-compile. The remote build is a clean rebuild and therefore idempotent,
    # so retry the whole invocation on a connection failure before giving up.
    # ssh exits 255 for transport-level errors (dropped connection, broken
    # pipe); any other non-zero code is the remote command's own exit status,
    # i.e. a genuine build/test failure that retrying won't fix -- fail fast on
    # those. Each attempt leaves a disconnect-surviving log on the box under
    # build-logs/.
    local windows_attempts=3
    local windows_attempt=1
    local windows_ssh_rc
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

    # A release is built from its per-release branch's HEAD; build-all is the
    # single place that manages the matching v<version> tag. RELEASE_REF is the
    # branch to build (default: newest release-v* branch); it may also be an
    # explicit tag/sha for rebuilding an older release.
    RELEASE_REF="${RELEASE_REF:-${1:-}}"
    git -C "${REPO_ROOT_DIR}" fetch origin --tags --prune --force
    if [ -z "${RELEASE_REF}" ]; then
        RELEASE_REF="$( { git -C "${REPO_ROOT_DIR}" for-each-ref \
                --format='%(refname:short)' \
                'refs/heads/release-v*' 'refs/remotes/origin/release-v*'; } 2>/dev/null \
            | sed -E 's#^origin/##' | sort -u -V | tail -n1 )"
        if [ -z "${RELEASE_REF}" ]; then
            echo "No release-v* branch found. Run scripts/prepare-release.sh first." >&2
            exit 1
        fi
    fi

    # Resolve what to build. For a release branch, build the pushed HEAD
    # (origin/<branch>) and manage the v<version> tag; for an explicit tag/sha,
    # build it as-is and leave tags alone.
    RELEASE_BRANCH=""
    VERSION=""
    if [[ "${RELEASE_REF}" == release-v* ]]; then
        RELEASE_BRANCH="${RELEASE_REF}"
        VERSION="${RELEASE_REF#release-v}"
        if git -C "${REPO_ROOT_DIR}" rev-parse -q --verify \
                "refs/remotes/origin/${RELEASE_BRANCH}^{commit}" >/dev/null; then
            BUILD_COMMITISH="origin/${RELEASE_BRANCH}"
        else
            BUILD_COMMITISH="${RELEASE_BRANCH}"
        fi
    else
        BUILD_COMMITISH="${RELEASE_REF}"
        VERSION="${RELEASE_REF#v}"
    fi
    if ! BUILD_COMMIT="$(git -C "${REPO_ROOT_DIR}" rev-parse -q --verify "${BUILD_COMMITISH}^{commit}")"; then
        echo "Release ref '${RELEASE_REF}' does not resolve to a commit." >&2
        exit 1
    fi
    echo "Building ${RELEASE_REF} at ${BUILD_COMMIT}"

    # Breadcrumbs and worktree are keyed by version (stable across branch/tag) so
    # switching a build between the branch and its tag reuses the same state.
    if [[ "${VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        STATE_KEY="${VERSION}"
    else
        STATE_KEY="${RELEASE_REF//\//-}"
    fi

    # Refresh the v<version> tag to the branch HEAD and push it, so the publish
    # stages (and the GitHub Releases they create) have a tag pointing at exactly
    # what we're building. Only for release branches with a real version; an
    # explicit tag/sha build leaves tags untouched. This repo's tag is not
    # consumed by SwiftPM (that keys off the separate moonshine-swift tag), so
    # moving it is safe.
    if [ -n "${RELEASE_BRANCH}" ] && [[ "${VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        TAG="v${VERSION}"
        echo "Refreshing tag ${TAG} -> ${BUILD_COMMIT} (from ${RELEASE_BRANCH} HEAD)."
        git -C "${REPO_ROOT_DIR}" tag -f -a "${TAG}" -m "Release ${TAG}" "${BUILD_COMMIT}"
        git -C "${REPO_ROOT_DIR}" push --force origin "refs/tags/${TAG}"
    fi

    # How each remote host syncs to the build point: for a release branch, reset
    # to the pushed branch HEAD; for a tag/sha, check it out detached. Built here
    # (bash + PowerShell variants) and injected into the ssh commands below.
    #
    # The branch checkout uses -f (matching the tag/sha path below) so a remote
    # CI host force-syncs to the pushed HEAD, discarding any drift in its
    # working tree: stale local modifications or in-the-way untracked files
    # (e.g. left behind by manual debugging) would otherwise make a plain
    # `git checkout` abort with "local changes would be overwritten" and fail the
    # whole release. -f overwrites only the conflicting paths, so unrelated
    # untracked build caches are left intact.
    if [ -n "${RELEASE_BRANCH}" ]; then
        REMOTE_GIT_SYNC="git fetch origin --tags --prune --force && git checkout -f -B '${RELEASE_BRANCH}' 'origin/${RELEASE_BRANCH}'"
        WIN_GIT_SYNC="git fetch origin --tags --prune --force ; git checkout -f -B ${RELEASE_BRANCH} origin/${RELEASE_BRANCH}"
    else
        REMOTE_GIT_SYNC="git fetch origin --tags --prune --force && git checkout -f '${RELEASE_REF}'"
        WIN_GIT_SYNC="git fetch origin --tags --prune --force ; git checkout -f ${RELEASE_REF}"
    fi

    # Resume breadcrumbs live in the main checkout (NOT the worktree, which is
    # recreated every run), keyed by version so different releases don't collide.
    # Completed stages are skipped on a re-run; RELEASE_FRESH=1 forces a full
    # rebuild.
    STATE_DIR="${REPO_ROOT_DIR}/.release-state/${STATE_KEY}"
    if [ -n "${RELEASE_FRESH:-}" ]; then
        echo "RELEASE_FRESH set; clearing resume breadcrumbs in ${STATE_DIR}."
        rm -rf "${STATE_DIR}"
    fi
    mkdir -p "${STATE_DIR}"
    echo "Resume breadcrumbs: ${STATE_DIR}"
    if compgen -G "${STATE_DIR}/*.done" >/dev/null; then
        echo "Already-completed stages that will be skipped:"
        for done_marker in "${STATE_DIR}"/*.done; do
            echo "  - $(basename "${done_marker}" .done)"
        done
    fi

    # Build the local platforms from an isolated worktree at the build commit, so
    # editing files in the main checkout can't corrupt the in-flight build.
    # Sub-scripts derive their repo root from their own location, so running the
    # worktree's copies roots everything in the worktree; .env vars were already
    # exported above and are inherited here.
    RELEASE_DIR="$(cd "${REPO_ROOT_DIR}/.." && pwd)/moonshine-release-${STATE_KEY}"
    git -C "${REPO_ROOT_DIR}" worktree remove --force "${RELEASE_DIR}" 2>/dev/null || true
    rm -rf "${RELEASE_DIR}"
    git -C "${REPO_ROOT_DIR}" worktree add --detach "${RELEASE_DIR}" "${BUILD_COMMIT}"

    # A fresh worktree only contains tracked files, but the build relies on a
    # few gitignored, repo-root credential/config files that live in the main
    # checkout (e.g. .pypirc, which build-pip-docker.sh COPYs into its image).
    # Copy them across so the worktree build behaves like an in-place one.
    for cfg in .env .pypirc local.properties; do
        if [ -f "${REPO_ROOT_DIR}/${cfg}" ]; then
            cp "${REPO_ROOT_DIR}/${cfg}" "${RELEASE_DIR}/${cfg}"
        fi
    done

    cd "${RELEASE_DIR}"
    run_stage test-core          scripts/test-core.sh
    run_stage test-python        scripts/test-python.sh
    run_stage test-docs          scripts/test-docs.sh --skip-build
    run_stage build-swift        scripts/build-swift.sh
    run_stage publish-swift      scripts/publish-swift.sh
    run_stage test-android-arm64 scripts/test-android.sh --avd "${ANDROID_ARM64_AVD:-moonshine_api26_arm64}"
    run_stage build-android      scripts/build-android.sh publish
    run_stage build-pip          scripts/build-pip.sh upload
    run_stage build-pip-docker   scripts/build-pip-docker.sh
    run_stage publish-binary     scripts/publish-binary.sh upload
    run_stage build-wasm         scripts/build-wasm.sh upload
    run_stage publish-examples   scripts/publish-examples.sh

    run_stage linux   stage_linux
    run_stage pi      stage_pi
    run_stage windows stage_windows

    if [ -n "${RELEASE_BRANCH}" ] && [[ "${VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "All stages complete for ${RELEASE_REF} (tag v${VERSION} at ${BUILD_COMMIT})."
        echo "You can now merge ${RELEASE_BRANCH} into main to sync version strings."
    else
        echo "All stages complete for ${RELEASE_REF} (${BUILD_COMMIT})."
    fi
}

main "$@"
