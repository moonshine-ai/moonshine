#! /bin/bash

# =============================================================================
# Making a Release
# =============================================================================
#
# A release lives on a per-release branch, release-v<version>, and every
# build/checkout step tracks that branch's HEAD. You never move tags or
# force-push by hand: build-all-platforms.sh derives and refreshes the
# v<version> tag from the branch automatically (the tag exists only because
# GitHub Releases and SwiftPM need one). So folding a fix into an in-progress
# release is just "add a commit to the branch and re-run the build".
#
#   1. Cut the release:
#        scripts/prepare-release.sh 0.0.69
#      Creates branch release-v0.0.69 off origin/main, rewrites every version
#      string via update-version.sh, commits, and pushes the branch. No tag is
#      created here; you are dropped back on main so you can keep working.
#
#   2. Build and publish:
#        scripts/build-all-platforms.sh
#      Builds the newest release-v* branch (override with RELEASE_REF). The
#      local platforms build in an isolated worktree at the branch HEAD; each
#      remote host resets to the same branch HEAD. It is resumable: completed
#      stages are skipped on re-run. build-all refreshes the v<version> tag to
#      the branch HEAD so the publish steps / GitHub Releases have a tag.
#
#   3. Fold in a fix (same version, build incrementally):
#        # commit the fix somewhere (e.g. on main), then:
#        scripts/patch-release.sh <commit-ish>...
#      Cherry-picks those commits onto the current release branch and pushes.
#      Re-run scripts/build-all-platforms.sh; it resumes and the remaining
#      stages pick up the new HEAD -- no tag juggling required.
#
#      NOTE: to ship a fix for a version that is ALREADY fully published, cut a
#      NEW version with prepare-release.sh instead. Package registries (PyPI,
#      Maven, GitHub release assets) reject re-uploading an existing version.
#
# After a release is fully published you can merge release-v<version> back into
# main to keep the version strings in sync.
#
# =============================================================================
# This script (prepare-release.sh)
# =============================================================================
#
# Cut a new release branch off origin/main with all version strings bumped, so
# build-all-platforms.sh can build it while you keep working on main.
#
# Usage:
#   scripts/prepare-release.sh <new_version> [<old_version>]
#
# Example:
#   scripts/prepare-release.sh 0.0.69
#
# What it does:
#   1. Verifies you are on a clean `main` that is in sync with origin/main.
#   2. Derives <old_version> from the latest release (unless you pass one).
#   3. Creates branch release-v<new_version> from origin/main.
#   4. Runs update-version.sh to rewrite every version string on that branch.
#   5. Commits and pushes the branch (no tag).
#   6. Switches you back to main so you can immediately keep working.

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

    # Versions are bare (no leading v), e.g. 0.0.69.
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
    git fetch origin --tags --prune --force

    if [ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]; then
        echo "Local main is not in sync with origin/main." >&2
        echo "Pull/push so they match, then try again." >&2
        exit 1
    fi

    if [ -z "${OLD_VERSION}" ]; then
        # Highest existing version across both release-v* branches and v* tags,
        # so derivation is correct regardless of whether past releases were cut
        # as branches (hybrid) or older tag-only releases.
        OLD_VERSION="$( { git tag -l 'v*'; \
            git for-each-ref --format='%(refname:short)' \
                'refs/heads/release-v*' 'refs/remotes/origin/release-v*'; } 2>/dev/null \
            | sed -E 's#^origin/##; s#^release-##; s#^v##' \
            | grep -E '^[0-9]+\.[0-9]+\.[0-9]+$' \
            | sort -V | tail -n1 )"
        if [ -z "${OLD_VERSION}" ]; then
            echo "Could not find a previous release to derive the old version." >&2
            echo "Pass it explicitly: $0 ${NEW_VERSION} <old_version>" >&2
            exit 1
        fi
    fi

    if [ "${OLD_VERSION}" = "${NEW_VERSION}" ]; then
        echo "Old and new versions are both '${NEW_VERSION}'; nothing to bump." >&2
        exit 1
    fi

    local branch="release-v${NEW_VERSION}"
    local tag="v${NEW_VERSION}"

    if git rev-parse -q --verify "refs/heads/${branch}" >/dev/null \
        || git ls-remote --exit-code --heads origin "${branch}" >/dev/null 2>&1; then
        echo "Branch '${branch}' already exists (locally or on origin)." >&2
        echo "That version is already in progress; use scripts/patch-release.sh" >&2
        echo "to fold fixes into it, or pick a new version." >&2
        exit 1
    fi
    if git rev-parse -q --verify "refs/tags/${tag}" >/dev/null \
        || git ls-remote --exit-code --tags origin "${tag}" >/dev/null 2>&1; then
        echo "Tag '${tag}' already exists (locally or on origin)." >&2
        echo "That version has already been built; pick a new version." >&2
        exit 1
    fi

    echo "Preparing release ${branch} (bumping ${OLD_VERSION} -> ${NEW_VERSION})..."
    echo "Creating branch '${branch}' from origin/main."
    git checkout -b "${branch}" origin/main

    # If anything below fails, drop the half-made branch and return to main so a
    # retry starts clean.
    restore_main() {
        git checkout -f main >/dev/null 2>&1 || true
        git branch -D "${branch}" >/dev/null 2>&1 || true
    }
    trap restore_main ERR

    echo "Rewriting version strings..."
    "${SCRIPTS_DIR}/update-version.sh" "${OLD_VERSION}" "${NEW_VERSION}"

    git commit -a -m "Update to version ${NEW_VERSION}"

    echo "Pushing branch to origin..."
    git push -u origin "${branch}"

    trap - ERR
    git checkout main

    cat <<EOF

Release branch ${branch} is prepared and pushed.

You are back on 'main' and can keep working. To build and publish it, run:

  scripts/build-all-platforms.sh

(That builds the newest release-v* branch by default -- ${branch}. It refreshes
the ${tag} tag from the branch HEAD automatically; no manual tagging needed.)

To fold a fix into this release later, commit it (typically on main) and run:

  scripts/patch-release.sh <commit-ish>...

then re-run scripts/build-all-platforms.sh (it resumes where it left off).
EOF
}

main "$@"
