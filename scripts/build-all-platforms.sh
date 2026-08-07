#! /bin/bash -ex

# Expected to be run on macOS.
#
# Usage:
#   ./scripts/build-all-platforms.sh [RELEASE_REF] [publish]
#
# Publishing is opt-in. Without the `publish` argument this is a DRY RUN: every
# stage still builds, tests and packages exactly as it would for a real release,
# but nothing leaves the machine. Specifically, a dry run does NOT:
#
#   - move or push the v<version> tag
#   - upload to PyPI, Maven Central or GitHub Releases
#   - publish the Swift package (scripts/publish-swift.sh is skipped entirely)
#   - fast-forward main or delete the candidate branch (finish-release is skipped)
#
# The Android stage uses publishToMavenLocal instead of publishAndReleaseToMaven-
# Central in a dry run, so the Gradle publication is still assembled, into ~/.m2
# rather than Maven Central -- which is also where publish-examples.sh picks it
# up when it builds the examples. Dry-run breadcrumbs and the build
# worktree are kept under separate, -dryrun-suffixed paths, so a dry run can
# never trick a subsequent real release into skipping an upload stage.
#
# This builds and publishes the development candidate branch (dev-v<version>),
# NOT your live working tree, from an isolated worktree so you can keep editing
# while the long-running build is in flight. The branch is cut by
# scripts/start-candidate.sh; run this with no arguments and it builds the
# newest dev-v* branch.
#
# The branch is the source of truth. build-all refreshes the v<version> tag to
# the branch HEAD (GitHub Releases / SwiftPM need a tag) -- you never move tags
# by hand. The local platforms build from an isolated git worktree at the branch
# HEAD; each remote host resets to the same branch HEAD. To fold in a late fix,
# add a commit to the branch and re-run.
#
# The final stage fast-forwards main to the commit it just published, which is
# what keeps main's README describing the binaries users can actually install.
# See scripts/start-candidate.sh for the full process.
#
# Resumable: each stage drops a breadcrumb under .release-state/<version>/ when
# it completes, so re-running the script (e.g. after a failure, or after folding
# in a fix) skips the stages that already finished and picks up where it left
# off. Set RELEASE_FRESH=1 to discard the breadcrumbs and rebuild every stage.
#
# Environment:
#   RELEASE_REF            - candidate branch (or tag/sha) to build. Defaults to
#                            the first non-`publish` argument, or the newest
#                            dev-v* branch.
#   RELEASE_PUBLISH        - if non-empty, publish as if `publish` was passed.
#   RELEASE_FRESH          - if non-empty, ignore/clear resume breadcrumbs and
#                            rebuild every stage.
#   RELEASE_SKIP_PREFLIGHT - if non-empty, skip scripts/preflight-release.sh.
#                            Only for deliberately unusual rebuilds.
#   LINUX_CLOUD_HOST       - SSH host for Linux cloud
#   LINUX_CLOUD_INSTANCE   - GCP instance name for the Linux VM (optional)
#   LINUX_CLOUD_ZONE       - GCP zone for the Linux VM (e.g. us-central1-b)
#   LINUX_CLOUD_PROJECT    - GCP project ID for the Linux VM
#   WINDOWS_CLOUD_USER     - SSH user for Windows cloud
#   WINDOWS_CLOUD_HOST     - SSH host for Windows cloud
#   WINDOWS_CLOUD_INSTANCE - GCP instance name for the Windows VM (optional)
#   WINDOWS_CLOUD_ZONE     - GCP zone for the Windows VM (e.g. us-central1-b)
#   WINDOWS_CLOUD_PROJECT  - GCP project ID for the Windows VM
#   WINDOWS_LIBVIRT_DOMAIN - libvirt domain name for a Windows KVM guest hosted
#                            on LINUX_CLOUD_HOST (e.g. moonshine-win). Used when
#                            WINDOWS_CLOUD_INSTANCE is unset. The guest is
#                            started before the windows stage and shut down on
#                            exit so qemu does not keep burning host cores.
#   GCP_SERVICE_ACCOUNT_KEY - path to a service account key file used for the
#                            gcloud calls below. Optional, but see the note.
#
# When the LINUX_CLOUD_INSTANCE / WINDOWS_CLOUD_INSTANCE variables are set the
# script will start the corresponding GCP VM before connecting and stop it
# again on exit (including on error) to minimize compute costs. The local
# Windows KVM guest (WINDOWS_LIBVIRT_DOMAIN) gets the same start/stop treatment
# via virsh on LINUX_CLOUD_HOST.
#
# Those gcloud calls are why GCP_SERVICE_ACCOUNT_KEY exists. A personal login's
# session expires part-way through a multi-hour release, and the next gcloud
# command then sits at an interactive reauthentication prompt with nobody there
# to answer it, hanging the run. Service account credentials do not expire that
# way. The key is activated in a gcloud configuration of its own, so an
# interactive gcloud in another terminal keeps using whatever account you had.
#
# The account in use is moonshine-release-ci@useful-sensors-website.iam
# .gserviceaccount.com, which can do nothing but suspend and resume the release
# VMs: a custom moonshineVmControl role bound on each VM individually, a
# moonshineVmOperations role at project level so gcloud can poll the resulting
# operation, and serviceAccountUser on the VM's own attached service account,
# without which Compute Engine refuses to start it. Recreate it with
# `gcloud iam service-accounts create` plus those bindings, and note that its
# key never expires -- delete the key in IAM if the laptop holding it is lost.

# Name of the throwaway gcloud configuration the service account is activated in.
GCLOUD_RELEASE_CONFIG=moonshine-release

# Authenticate gcloud as the release service account, if one is configured.
# Everything this script runs inherits CLOUDSDK_ACTIVE_CONFIG_NAME, so the
# credential applies to the whole run without touching the configuration an
# interactive gcloud uses.
gcp_use_service_account() {
    if [ -z "${GCP_SERVICE_ACCOUNT_KEY:-}" ]; then
        echo "GCP_SERVICE_ACCOUNT_KEY is not set; gcloud will use your own login," \
             "which can stall this run at a reauthentication prompt."
        return 0
    fi
    if [ ! -r "${GCP_SERVICE_ACCOUNT_KEY}" ]; then
        echo "GCP_SERVICE_ACCOUNT_KEY is set to '${GCP_SERVICE_ACCOUNT_KEY}'," \
             "which cannot be read." >&2
        exit 1
    fi

    gcloud config configurations describe "${GCLOUD_RELEASE_CONFIG}" >/dev/null 2>&1 \
        || gcloud config configurations create "${GCLOUD_RELEASE_CONFIG}" --no-activate
    CLOUDSDK_ACTIVE_CONFIG_NAME="${GCLOUD_RELEASE_CONFIG}" \
        gcloud auth activate-service-account --key-file="${GCP_SERVICE_ACCOUNT_KEY}"
    export CLOUDSDK_ACTIVE_CONFIG_NAME="${GCLOUD_RELEASE_CONFIG}"
    echo "gcloud authenticated as $(gcloud config get-value account 2>/dev/null)."
}

# Read each configured VM before the build starts. The stages that need these
# run hours in, and an expired credential or a missing permission is much
# cheaper to discover now than after a night of building.
gcp_check_instance_access() {
    local instance="$1"
    local zone="$2"
    local project="$3"

    if [ -z "${instance}" ]; then
        return 0
    fi
    if ! gcloud compute instances describe "${instance}" \
            --zone="${zone}" \
            --project="${project}" \
            --format="value(status)" >/dev/null; then
        echo "Cannot read GCP instance ${instance} in ${zone} (project ${project})." \
             "Fix the gcloud credentials before starting a release." >&2
        exit 1
    fi
}

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

# True when Windows CI runs against a libvirt KVM guest on LINUX_CLOUD_HOST
# rather than a GCP VM. The guest is expensive to leave idle (8 vCPUs / 16 GB),
# so the windows stage starts it and cleanup shuts it down.
windows_uses_libvirt_guest() {
    [ -z "${WINDOWS_CLOUD_INSTANCE:-}" ] \
        && [ -n "${WINDOWS_LIBVIRT_DOMAIN:-}" ] \
        && [ -n "${LINUX_CLOUD_HOST:-}" ]
}

# Run a virsh command on the Linux host that owns the Windows KVM guest.
libvirt_windows_virsh() {
    # shellcheck disable=SC2029 # intentional remote expansion of the domain name
    ssh -o BatchMode=yes "${LINUX_CLOUD_HOST}" \
        "sudo virsh $*"
}

# Boot the local Windows KVM guest and wait until SSH answers.
libvirt_windows_start() {
    local domain="${WINDOWS_LIBVIRT_DOMAIN}"
    local ssh_target="${WINDOWS_CLOUD_USER}@${WINDOWS_CLOUD_HOST}"
    local state

    state="$(libvirt_windows_virsh domstate "${domain}" 2>/dev/null || true)"
    if [ "${state}" != "running" ]; then
        echo "Starting libvirt guest ${domain} on ${LINUX_CLOUD_HOST}..."
        libvirt_windows_virsh start "${domain}"
    else
        echo "Libvirt guest ${domain} is already running."
    fi

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

# Shut down the local Windows KVM guest. Prefer ACPI shutdown, then destroy.
# Failures are reported but do not abort cleanup of anything else.
libvirt_windows_shutdown() {
    local domain="${WINDOWS_LIBVIRT_DOMAIN}"
    local state
    local i

    state="$(libvirt_windows_virsh domstate "${domain}" 2>/dev/null || true)"
    if [ -z "${state}" ] || [ "${state}" = "shut off" ]; then
        echo "Libvirt guest ${domain} is already shut off."
        return 0
    fi

    echo "Shutting down libvirt guest ${domain} on ${LINUX_CLOUD_HOST}..."
    libvirt_windows_virsh shutdown "${domain}" \
        || echo "Warning: virsh shutdown ${domain} failed." >&2

    for i in $(seq 1 24); do
        state="$(libvirt_windows_virsh domstate "${domain}" 2>/dev/null || true)"
        if [ "${state}" = "shut off" ]; then
            echo "Libvirt guest ${domain} is shut off."
            return 0
        fi
        sleep 5
    done

    echo "Guest ${domain} still '${state}' after graceful shutdown; destroying..." >&2
    libvirt_windows_virsh destroy "${domain}" \
        || echo "Warning: virsh destroy ${domain} failed." >&2
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
    elif windows_uses_libvirt_guest; then
        libvirt_windows_shutdown
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

# A stage that exists only to publish, and so has no meaningful dry-run form:
# unlike the build stages, it can't be run with its upload withheld. Skipped
# without a breadcrumb, so a later real release still runs it.
run_publish_stage() {
    local name="$1"
    shift
    if [ -z "${PUBLISH}" ]; then
        echo "[dry-run] Skipping publish-only stage '${name}'."
        return 0
    fi
    run_stage "${name}" "$@"
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
      && scripts/build-pip.sh ${UPLOAD_ARGS[*]}" || exit 1
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
    elif windows_uses_libvirt_guest; then
        libvirt_windows_start
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

    # The Windows guest has no persistent gh auth or PyPI credentials. For
    # upload runs, push secrets from this Mac into a short-lived env script on
    # the guest (via SSH stdin — never argv, so `ps` cannot leak them), then
    # dot-source it for the CI session and delete it afterwards.
    local windows_env_bootstrap=""
    if [ -n "${WINDOWS_UPLOAD_FLAG}" ]; then
        local gh_token="${GH_TOKEN:-${GITHUB_TOKEN:-}}"
        if [ -z "${gh_token}" ] && command -v gh >/dev/null 2>&1; then
            gh_token="$(gh auth token 2>/dev/null || true)"
        fi
        if [ -z "${gh_token}" ]; then
            echo "Windows upload requires GH_TOKEN/GITHUB_TOKEN or a logged-in" \
                "gh on this Mac (the guest has no gh auth of its own)." >&2
            exit 1
        fi
        local pypirc="${REPO_ROOT_DIR}/.pypirc"
        if [ ! -f "${pypirc}" ]; then
            echo "Windows upload requires ${pypirc} for twine (PyPI)." >&2
            exit 1
        fi

        echo "Installing short-lived upload credentials on Windows guest..."
        # Build a PowerShell env script locally and stream it over SSH stdin so
        # neither GH nor PyPI secrets appear in process argv.
        python3 - "${gh_token}" "${pypirc}" <<'PY' | ssh "${windows_ssh_opts[@]}" \
            "${WINDOWS_CLOUD_USER}@${WINDOWS_CLOUD_HOST}" \
            "powershell -NoProfile -Command \"Set-Content -LiteralPath (Join-Path \$env:USERPROFILE '.moonshine-release-env.ps1') -Value ([Console]::In.ReadToEnd()) -Encoding utf8\""
import configparser, sys
from pathlib import Path

def ps_single(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"

gh_token = sys.argv[1]
cfg = configparser.ConfigParser()
cfg.read(Path(sys.argv[2]))
user = cfg["pypi"]["username"]
password = cfg["pypi"]["password"]
print("$env:GH_TOKEN = " + ps_single(gh_token))
print("$env:GITHUB_TOKEN = $env:GH_TOKEN")
print("$env:TWINE_USERNAME = " + ps_single(user))
print("$env:TWINE_PASSWORD = " + ps_single(password))
PY

        windows_env_bootstrap='. (Join-Path $env:USERPROFILE ".moonshine-release-env.ps1"); '
        echo "Forwarding GitHub + PyPI credentials into Windows session for uploads."
    fi

    # The Windows login shell is PowerShell. Sync to the build point first (that
    # also refreshes run-windows-ci.ps1 itself), then hand off to the
    # orchestrator, which runs each step with heavy, disconnect-surviving logging
    # (see the script header and build-logs/ on the box). Using a single
    # orchestrator also makes the run abort on the first failing step rather
    # than masking it behind the exit code of the last chained command. The
    # sync command is expanded locally (via the single-quote break) so PowerShell
    # variables like $LASTEXITCODE stay intact for the remote shell.
    local windows_remote_cmd="${windows_env_bootstrap}"'try { cd moonshine `
      ; '"${WIN_GIT_SYNC}"' `
      ; if ($LASTEXITCODE -ne 0) { Write-Host "git sync failed"; exit 1 } `
      ; pwsh -NoProfile -ExecutionPolicy Bypass -File scripts\run-windows-ci.ps1'"${WINDOWS_UPLOAD_FLAG}"' `
      } finally { Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $env:USERPROFILE ".moonshine-release-env.ps1") }'

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

    # A release takes hours and is meant to be left alone, but a Mac that sleeps
    # part-way through takes the build down with it. With the lid shut it only
    # surfaces for brief maintenance dark wakes, and in those the audio hardware
    # stays powered down, so any stage that plays audio (the Swift TTS tests) runs
    # against a device that never starts. Hold a power assertion for exactly as long
    # as this script lives: caffeinate -w exits when our pid does. The system-sleep
    # part of the assertion only holds on AC power, so keep an unattended run
    # plugged in.
    caffeinate -dims -w $$ &

    gcp_use_service_account
    gcp_check_instance_access \
        "${LINUX_CLOUD_INSTANCE:-}" "${LINUX_CLOUD_ZONE:-}" "${LINUX_CLOUD_PROJECT:-}"
    gcp_check_instance_access \
        "${WINDOWS_CLOUD_INSTANCE:-}" "${WINDOWS_CLOUD_ZONE:-}" "${WINDOWS_CLOUD_PROJECT:-}"

    # Arguments are order-independent: `publish` opts in to publishing, and any
    # other bare word is the release ref. Publishing is opt-in so that the
    # default invocation is a rehearsal -- the expensive mistakes in a release
    # are all uploads, and no registry lets you take one back.
    PUBLISH="${RELEASE_PUBLISH:-}"
    local ref_arg=""
    for arg in "$@"; do
        case "${arg}" in
            publish) PUBLISH=1 ;;
            "") ;;
            -*) echo "Unknown option: '${arg}'" >&2; exit 1 ;;
            *)
                if [ -n "${ref_arg}" ]; then
                    echo "Unexpected extra argument: '${arg}' (release ref is" \
                         "already '${ref_arg}')." >&2
                    exit 1
                fi
                ref_arg="${arg}"
                ;;
        esac
    done

    # A release is built from the candidate branch's HEAD; build-all is the
    # single place that manages the matching v<version> tag. RELEASE_REF is the
    # branch to build (default: newest dev-v* branch); it may also be an
    # explicit tag/sha for rebuilding an older release.
    RELEASE_REF="${RELEASE_REF:-${ref_arg}}"
    git -C "${REPO_ROOT_DIR}" fetch origin --tags --prune --force
    if [ -z "${RELEASE_REF}" ]; then
        RELEASE_REF="$( { git -C "${REPO_ROOT_DIR}" for-each-ref \
                --format='%(refname:short)' \
                'refs/heads/dev-v*' 'refs/remotes/origin/dev-v*'; } 2>/dev/null \
            | sed -E 's#^origin/##' | sort -u -V | tail -n1 )"
        if [ -z "${RELEASE_REF}" ]; then
            echo "No dev-v* branch found. Run scripts/start-candidate.sh first." >&2
            exit 1
        fi
    fi

    # Resolve what to build. For a release branch, build the pushed HEAD
    # (origin/<branch>) and manage the v<version> tag; for an explicit tag/sha,
    # build it as-is and leave tags alone.
    RELEASE_BRANCH=""
    VERSION=""
    if [[ "${RELEASE_REF}" == dev-v* ]]; then
        RELEASE_BRANCH="${RELEASE_REF}"
        VERSION="${RELEASE_REF#dev-v}"
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
    if [ -n "${PUBLISH}" ]; then
        echo "Mode: PUBLISH -- artifacts will be uploaded and main will advance."
    else
        echo "Mode: DRY RUN -- nothing will be uploaded, no remote refs moved."
        echo "      Re-run with 'publish' to ship."
    fi

    # Withheld from every stage that gates its upload on an argument, so a dry
    # run exercises the same build and packaging code paths and stops short of
    # the upload. Stages whose only job is publishing are skipped by
    # run_publish_stage instead.
    if [ -n "${PUBLISH}" ]; then
        UPLOAD_ARGS=(upload)
        ANDROID_ARGS=(publish)
        WINDOWS_UPLOAD_FLAG=" -Upload"
    else
        UPLOAD_ARGS=()
        ANDROID_ARGS=(local)
        WINDOWS_UPLOAD_FLAG=""
    fi

    # Breadcrumbs and worktree are keyed by version (stable across branch/tag) so
    # switching a build between the branch and its tag reuses the same state.
    if [[ "${VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        STATE_KEY="${VERSION}"
    else
        STATE_KEY="${RELEASE_REF//\//-}"
    fi
    # A dry run's breadcrumbs must never be mistaken for a real release's: a
    # build-pip stage that completed without uploading would otherwise be
    # skipped by the subsequent publish run, and the wheel would never ship.
    # Preflight reads the unsuffixed directory for the same reason -- a dry run
    # is not an in-progress release.
    if [ -z "${PUBLISH}" ]; then
        STATE_KEY="${STATE_KEY}-dryrun"
    fi

    # Check everything that has to be true before the first artifact goes out.
    # This runs before the tag is moved, so a failed preflight leaves no trace.
    # Skipped for explicit tag/sha rebuilds, which are deliberately reproducing
    # an old state rather than shipping a new one.
    if [ -n "${RELEASE_BRANCH}" ] && [ -z "${RELEASE_SKIP_PREFLIGHT:-}" ]; then
        "${SCRIPTS_DIR}/preflight-release.sh" "${RELEASE_BRANCH}" "${BUILD_COMMIT}"
    fi

    # Refresh the v<version> tag to the branch HEAD and push it, so the publish
    # stages (and the GitHub Releases they create) have a tag pointing at exactly
    # what we're building. Only for release branches with a real version; an
    # explicit tag/sha build leaves tags untouched. This repo's tag is not
    # consumed by SwiftPM (that keys off the separate moonshine-swift tag), so
    # moving it is safe.
    # A dry run leaves tags alone, locally and on the remote: the tag exists to
    # anchor the published artifacts, and a dry run has none. Every host syncs to
    # BUILD_COMMIT by sha, so no stage needs the tag to be in place.
    if [ -n "${RELEASE_BRANCH}" ] && [[ "${VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        TAG="v${VERSION}"
        if [ -n "${PUBLISH}" ]; then
            echo "Refreshing tag ${TAG} -> ${BUILD_COMMIT} (from ${RELEASE_BRANCH} HEAD)."
            git -C "${REPO_ROOT_DIR}" tag -f -a "${TAG}" -m "Release ${TAG}" "${BUILD_COMMIT}"
            git -C "${REPO_ROOT_DIR}" push --force origin "refs/tags/${TAG}"
        else
            echo "[dry-run] Would refresh tag ${TAG} -> ${BUILD_COMMIT} and push it."
        fi
    fi

    # How each remote host syncs to the build point. Every host checks out the
    # single BUILD_COMMIT resolved above, detached, rather than tracking the
    # branch HEAD. The candidate branch is also the development branch, so a
    # commit pushed while this multi-hour build is in flight would otherwise be
    # picked up by whichever stages had not started yet, silently producing a
    # release built from a mix of commits. Pinning the sha means a run is
    # reproducible and development never has to freeze; a re-run re-resolves
    # the branch HEAD and the remaining stages move to the new commit together.
    #
    # The checkout uses -f so a remote CI host discards any drift in its working
    # tree: stale local modifications or in-the-way untracked files (e.g. left
    # behind by manual debugging) would otherwise make a plain `git checkout`
    # abort with "local changes would be overwritten" and fail the whole
    # release. -f overwrites only the conflicting paths, so unrelated untracked
    # build caches are left intact.
    REMOTE_GIT_SYNC="git fetch origin --tags --prune --force && git checkout -f --detach '${BUILD_COMMIT}'"
    WIN_GIT_SYNC="git fetch origin --tags --prune --force ; git checkout -f --detach ${BUILD_COMMIT}"

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

    # Model/TTS binaries are gitignored (fetched from the CDN). A resumed run
    # recreates the worktree but may skip test-core, which is the stage that
    # normally bootstraps them — leaving later stages (build-swift, etc.) with
    # an empty test-assets tree. Prefer a fast copy from the main checkout when
    # those files are already present; otherwise fetch into the worktree.
    if [[ ! -f "${RELEASE_DIR}/test-assets/tiny-en/encoder_model.ort" ]] || \
       [[ ! -f "${RELEASE_DIR}/core/moonshine-tts/data/kokoro/model.ort" ]]; then
        if [[ -f "${REPO_ROOT_DIR}/test-assets/tiny-en/encoder_model.ort" ]] && \
           [[ -f "${REPO_ROOT_DIR}/core/moonshine-tts/data/kokoro/model.ort" ]]; then
            echo "Copying voice assets from main checkout into release worktree..."
            # test-assets: copy gitignored model blobs the tracked tree lacks.
            mkdir -p "${RELEASE_DIR}/test-assets"
            rsync -a --ignore-existing \
                "${REPO_ROOT_DIR}/test-assets/" "${RELEASE_DIR}/test-assets/"
            rsync -a --ignore-existing \
                "${REPO_ROOT_DIR}/core/moonshine-tts/data/" \
                "${RELEASE_DIR}/core/moonshine-tts/data/"
        else
            echo "Fetching voice assets into release worktree..."
            (
                cd "${RELEASE_DIR}"
                scripts/fetch-voice-assets.sh all
            )
        fi
    fi

    # The worktree is always recreated from the build commit, so build products
    # from prior stages (e.g. swift/Moonshine.xcframework) are gone even when
    # their breadcrumbs remain. Restore a cached copy when we have one; otherwise
    # drop the breadcrumb so build-swift re-runs.
    if [[ -f "${STATE_DIR}/build-swift.done" ]]; then
        if [[ ! -d "${RELEASE_DIR}/swift/Moonshine.xcframework" ]]; then
            if [[ -d "${STATE_DIR}/Moonshine.xcframework" ]]; then
                echo "Restoring cached Moonshine.xcframework into the fresh worktree..."
                mkdir -p "${RELEASE_DIR}/swift"
                rm -rf "${RELEASE_DIR}/swift/Moonshine.xcframework"
                cp -R "${STATE_DIR}/Moonshine.xcframework" \
                    "${RELEASE_DIR}/swift/Moonshine.xcframework"
            else
                echo "build-swift.done but no xcframework in the worktree or cache;" \
                     "clearing the breadcrumb so it rebuilds."
                rm -f "${STATE_DIR}/build-swift.done"
            fi
        fi
    fi

    cd "${RELEASE_DIR}"
    run_stage test-core          scripts/test-core.sh
    run_stage test-python        scripts/test-python.sh
    run_stage test-docs          scripts/test-docs.sh --skip-build
    run_stage build-swift        scripts/build-swift.sh
    # Keep a copy outside the disposable worktree so resumed runs can skip the
    # multi-platform Swift rebuild when only later stages still need to run.
    if [[ -d "${RELEASE_DIR}/swift/Moonshine.xcframework" ]]; then
        rm -rf "${STATE_DIR}/Moonshine.xcframework"
        cp -R "${RELEASE_DIR}/swift/Moonshine.xcframework" \
            "${STATE_DIR}/Moonshine.xcframework"
    fi
    run_publish_stage publish-swift scripts/publish-swift.sh
    run_stage test-android-arm64 scripts/test-android.sh --avd "${ANDROID_ARM64_AVD:-moonshine_api26_arm64}"
    # Physical Pixel + iPad Tiny Streaming latency (same metric as the README
    # table). Requires the devices to be plugged into this Mac; set
    # MOBILE_LATENCY_OPTIONAL=1 to skip when hardware is absent. Does not rewrite
    # the README here (this is a disposable worktree) -- refresh figures with
    # scripts/test-mobile-latency.sh --update-readme on the candidate branch.
    run_stage test-mobile-latency scripts/test-mobile-latency.sh --skip-build-swift
    run_stage build-android      scripts/build-android.sh "${ANDROID_ARGS[@]}"
    run_stage build-pip          scripts/build-pip.sh "${UPLOAD_ARGS[@]}"
    run_stage build-pip-docker   scripts/build-pip-docker.sh "${UPLOAD_ARGS[@]}"
    run_stage publish-binary     scripts/publish-binary.sh "${UPLOAD_ARGS[@]}"
    run_stage build-wasm         scripts/build-wasm.sh "${UPLOAD_ARGS[@]}"
    run_stage publish-examples   scripts/publish-examples.sh "${UPLOAD_ARGS[@]}"

    run_stage linux   stage_linux
    run_stage pi      stage_pi
    run_stage windows stage_windows

    if [ -z "${PUBLISH}" ]; then
        echo "Dry run complete for ${RELEASE_REF} (${BUILD_COMMIT})."
        echo "Everything built and tested; nothing was published and main was" \
             "left alone."
        echo "Ship it with: scripts/build-all-platforms.sh ${RELEASE_REF} publish"
        return 0
    fi

    # Everything is published, so main can now advance to what shipped. This is
    # a stage rather than a manual follow-up because skipping it is what let
    # main's docs drift away from the released binaries in the first place.
    if [ -n "${RELEASE_BRANCH}" ] && [[ "${VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        run_stage finish-release \
            scripts/finish-release.sh "${RELEASE_BRANCH}" "${BUILD_COMMIT}"
        echo "All stages complete for ${RELEASE_REF} (tag v${VERSION} at ${BUILD_COMMIT})."
        echo "main now points at the released commit."
        echo "Start the next cycle with scripts/start-candidate.sh <next_version>."
    else
        echo "All stages complete for ${RELEASE_REF} (${BUILD_COMMIT})."
        echo "Rebuild of an explicit ref: main was left alone."
    fi
}

main "$@"
