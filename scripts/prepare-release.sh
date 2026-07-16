#! /bin/bash

# =============================================================================
# Making a Release
# =============================================================================
#
# Releases are built from a *frozen* git ref rather than the live working tree
# or `main`, so you can keep working on `main` while the (long-running) build
# and publish steps run without any risk of your in-progress changes leaking
# into the release. The workflow is three steps:
#
#   1. Cut the release. From a clean, up-to-date `main`, run this script with
#      the new version number. It creates a `release-v<version>` branch off
#      origin/main, rewrites every version string via update-version.sh,
#      commits, tags `v<version>`, pushes the branch and tag, and then drops
#      you back on `main`.
#
#        scripts/prepare-release.sh 0.0.69
#
#   2. Build and publish the frozen tag. Run scripts/build-all-platforms.sh
#      with no arguments and it builds the most recent v* tag (override with
#      the RELEASE_REF environment variable or a first argument). The local
#      platforms build inside an isolated `git worktree` checked out at that
#      tag, and each remote host checks out the same tag instead of pulling
#      `main`. Because the whole build is pinned to the tag, you are free to
#      keep committing to `main` while it runs.
#
#        scripts/build-all-platforms.sh
#
#   3. Fold in fixes if needed. If you discover a problem partway through,
#      commit the fix (usually on `main`) and use scripts/patch-release.sh to
#      cherry-pick it onto a new `release-v<version>` branch, bump to a new
#      patch version, tag, and push. A new version number is required because
#      package registries (PyPI, Maven, GitHub release assets) reject
#      re-uploading an existing version. Then re-run build-all-platforms.sh.
#
#        scripts/patch-release.sh 0.0.70 <commit-sha>
#
# After a release has been published successfully you can merge its
# `release-v<version>` branch back into `main` to keep the version strings in
# sync.
#
# =============================================================================
# This script (prepare-release.sh)
# =============================================================================
#
# Prepare a new release by freezing the current top of `main` onto an immutable
# release branch + tag, so the (long-running) build/publish step can build from
# that frozen snapshot while you keep working on `main`.
#
# Usage:
#   scripts/prepare-release.sh <new_version> [<old_version>]
#
# Example:
#   scripts/prepare-release.sh 0.0.69
#
# What it does:
#   1. Verifies you are on a clean `main` that is in sync with origin/main.
#   2. Derives <old_version> from the most recent v* tag (unless you pass one).
#   3. Creates branch `release-v<new_version>` from origin/main.
#   4. Runs update-version.sh to rewrite every version string on that branch.
#   5. Commits, annotates a `v<new_version>` tag, and pushes the branch + tag.
#   6. Switches you back to `main` so you can immediately keep working.
#
# After this succeeds, run:
#   scripts/build-all-platforms.sh
# which builds/publishes the frozen `v<new_version>` tag (see that script). Any
# further commits you make to `main` will NOT affect the in-flight release.
#
# If you need to fold a fix into an already-prepared release, use
# scripts/patch-release.sh instead of re-running this script.

set -euo pipefail

# All imperative work lives inside main() so bash parses the whole script
# before executing anything (mirrors build-all-platforms.sh).
main() {
    if [ $# -lt 1 ] || [ $# -gt 2 ]; then
        echo "Usage: $0 <new_version> [<old_version>]" >&2
        exit 1
    fi

    NEW_VERSION="$1"
    OLD_VERSION="${2:-}"

    # Versions are bare (no leading v), e.g. 0.0.69. Tags are v-prefixed.
    if ! [[ "${NEW_VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "New version '${NEW_VERSION}' is not in X.Y.Z form (no leading 'v')." >&2
        exit 1
    fi

    SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
    REPO_ROOT_DIR="$(dirname "${SCRIPTS_DIR}")"
    cd "${REPO_ROOT_DIR}"

    local current_branch
    current_branch="$(git rev-parse --abbrev-ref HEAD)"
    if [ "${current_branch}" != "main" ]; then
        echo "You are on '${current_branch}', but releases must start from 'main'." >&2
        echo "Switch to main (git checkout main) and try again." >&2
        exit 1
    fi

    if ! git diff --quiet || ! git diff --cached --quiet; then
        echo "Your working tree has uncommitted changes to tracked files." >&2
        echo "Commit or stash them before preparing a release." >&2
        exit 1
    fi

    echo "Fetching latest refs from origin..."
    git fetch origin --tags --prune

    if [ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]; then
        echo "Local main is not in sync with origin/main." >&2
        echo "Pull/push so they match, then try again." >&2
        exit 1
    fi

    if [ -z "${OLD_VERSION}" ]; then
        local latest_tag
        # Highest-version v* tag, NOT `git describe` (which only sees tags
        # reachable from a commit). Release tags live on release-v* branches
        # that aren't ancestors of main, so describe would miss the latest one.
        latest_tag="$(git tag -l 'v*' --sort=-v:refname | head -n1)"
        if [ -z "${latest_tag}" ]; then
            echo "Could not find a previous v* tag to derive the old version." >&2
            echo "Pass it explicitly: $0 ${NEW_VERSION} <old_version>" >&2
            exit 1
        fi
        OLD_VERSION="${latest_tag#v}"
    fi

    if [ "${OLD_VERSION}" = "${NEW_VERSION}" ]; then
        echo "Old and new versions are both '${NEW_VERSION}'; nothing to bump." >&2
        exit 1
    fi

    local tag="v${NEW_VERSION}"
    local branch="release-v${NEW_VERSION}"

    if git rev-parse -q --verify "refs/tags/${tag}" >/dev/null \
        || git ls-remote --exit-code --tags origin "${tag}" >/dev/null 2>&1; then
        echo "Tag '${tag}' already exists (locally or on origin)." >&2
        echo "Pick a new version, or use scripts/patch-release.sh for fixes." >&2
        exit 1
    fi
    if git rev-parse -q --verify "refs/heads/${branch}" >/dev/null \
        || git ls-remote --exit-code --heads origin "${branch}" >/dev/null 2>&1; then
        echo "Branch '${branch}' already exists (locally or on origin)." >&2
        exit 1
    fi

    echo "Preparing release ${tag} (bumping ${OLD_VERSION} -> ${NEW_VERSION})..."
    echo "Creating branch '${branch}' from origin/main."
    git checkout -b "${branch}" origin/main

    # If anything below fails, drop the half-made branch and return to main so a
    # retry starts clean.
    restore_main() {
        git cherry-pick --abort >/dev/null 2>&1 || true
        git checkout -f main >/dev/null 2>&1 || true
        git branch -D "${branch}" >/dev/null 2>&1 || true
    }
    trap restore_main ERR

    echo "Rewriting version strings..."
    "${SCRIPTS_DIR}/update-version.sh" "${OLD_VERSION}" "${NEW_VERSION}"

    git commit -a -m "Update to version ${NEW_VERSION}"
    git tag -a "${tag}" -m "Release ${tag}"

    echo "Pushing branch and tag to origin..."
    git push origin "${branch}"
    git push origin "${tag}"

    trap - ERR
    git checkout main

    cat <<EOF

Release ${tag} is prepared and pushed.
  branch: ${branch}
  tag:    ${tag}

You are back on 'main' and can keep working. To build and publish the frozen
snapshot, run:

  scripts/build-all-platforms.sh

(That builds tag ${tag} by default. Override with RELEASE_REF=<ref> if needed.)

To fold a fix into this release later, commit it (typically on main) and run:

  scripts/patch-release.sh <next_version> <commit-ish>...
EOF
}

main "$@"
