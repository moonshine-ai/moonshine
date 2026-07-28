#! /bin/bash

# =============================================================================
# The release process
# =============================================================================
#
# `main` is the released branch: it only ever contains code that has been fully
# published to PyPI, Maven, npm, SwiftPM and GitHub Releases. GitHub renders
# main's README on the repo landing page, so keeping main frozen at the last
# release is what makes the documentation match the binaries people actually
# install.
#
# Nothing is committed directly to main. All development happens on a single
# candidate branch, dev-v<version>, which is created by this script and folded
# back into main by scripts/finish-release.sh when the release ships. Because
# main only ever moves by fast-forwarding to the candidate, the two never
# diverge and the fold-in can never conflict.
#
#   1. Start a development cycle:
#        scripts/start-candidate.sh 0.1.0
#      Creates dev-v0.1.0 off origin/main, rewrites every version string to the
#      version this cycle will ship, commits, pushes, and leaves you on the
#      branch. Bumping here (rather than just before the release) means the
#      docs and samples reference the right version all cycle long, and can be
#      tested against it.
#
#   2. Develop. Commit to dev-v0.1.0 as you normally would to main.
#
#   3. Build and publish:
#        scripts/build-all-platforms.sh          # dry run: builds, uploads nothing
#        scripts/build-all-platforms.sh publish  # the real thing
#      Builds the newest dev-v* branch (override with RELEASE_REF). With
#      `publish` it also refreshes the v<version> tag from its HEAD, runs every
#      publish stage, and finishes by fast-forwarding main to what it just
#      shipped.
#
#   4. Start the next cycle with step 1 again.
#
# To ship a fix for a version that is ALREADY fully published, start a new
# candidate at the next patch version: package registries (PyPI, Maven, GitHub
# release assets) reject re-uploading an existing version. If a candidate is
# already in flight when that happens, rebase it onto main afterwards so the
# fast-forward invariant still holds.
#
# =============================================================================
# This script (start-candidate.sh)
# =============================================================================
#
# Usage:
#   scripts/start-candidate.sh <new_version> [<old_version>]
#
# Example:
#   scripts/start-candidate.sh 0.1.0
#
# What it does:
#   1. Verifies your working tree is clean and no other candidate is open.
#   2. Derives <old_version> from core/CMakeLists.txt (the repo's own idea of
#      its version) unless you pass one.
#   3. Switches to main and fast-forwards it to origin/main.
#   4. Creates branch dev-v<new_version> from main, or reuses it if you are
#      already standing on it.
#   5. Runs update-version.sh to rewrite every version string on that branch.
#   6. Commits and pushes the branch (no tag), leaving you on it.

set -euo pipefail

main() {
    if [ $# -lt 1 ] || [ $# -gt 2 ]; then
        echo "Usage: $0 <new_version> [<old_version>]" >&2
        exit 1
    fi

    NEW_VERSION="$1"
    OLD_VERSION="${2:-}"

    # Versions are bare (no leading v), e.g. 0.1.0.
    if ! [[ "${NEW_VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "New version '${NEW_VERSION}' is not in X.Y.Z form (no leading 'v')." >&2
        exit 1
    fi

    SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
    REPO_ROOT_DIR="$(dirname "${SCRIPTS_DIR}")"
    cd "${REPO_ROOT_DIR}"

    if ! git diff --quiet || ! git diff --cached --quiet; then
        echo "Your working tree has uncommitted changes to tracked files." >&2
        echo "Commit or stash them before starting a candidate: the version" >&2
        echo "bump is committed with 'git commit -a' and would sweep them up." >&2
        exit 1
    fi

    local branch="dev-v${NEW_VERSION}"
    local tag="v${NEW_VERSION}"

    echo "Fetching latest refs from origin..."
    git fetch origin --tags --prune --force

    if git rev-parse -q --verify "refs/tags/${tag}" >/dev/null \
        || git ls-remote --exit-code --tags origin "${tag}" >/dev/null 2>&1; then
        echo "Tag '${tag}' already exists (locally or on origin)." >&2
        echo "That version has already been built; pick a new version." >&2
        exit 1
    fi

    # The repo's own version, not the newest tag or branch. Deriving it from a
    # tag breaks whenever main is behind the last release, which is exactly the
    # state this process exists to prevent but has to cope with while adopting.
    if [ -z "${OLD_VERSION}" ]; then
        OLD_VERSION="$(sed -n 's/^set(MOONSHINE_VERSION "\([^"]*\)").*/\1/p' \
            core/CMakeLists.txt | head -n1)"
        if [ -z "${OLD_VERSION}" ]; then
            echo "Could not read MOONSHINE_VERSION from core/CMakeLists.txt." >&2
            echo "Pass the old version explicitly: $0 ${NEW_VERSION} <old_version>" >&2
            exit 1
        fi
    fi

    if [ "${OLD_VERSION}" = "${NEW_VERSION}" ]; then
        echo "The repo is already at '${NEW_VERSION}'; nothing to bump." >&2
        exit 1
    fi
    # Versions only ever go up. Going backwards would make the release look
    # older than what is already on the registries.
    if [ "$(printf '%s\n%s\n' "${OLD_VERSION}" "${NEW_VERSION}" \
        | sort -V | tail -n1)" != "${NEW_VERSION}" ]; then
        echo "New version '${NEW_VERSION}' is lower than the current" \
             "'${OLD_VERSION}'." >&2
        exit 1
    fi

    # Two entry points: the normal one cuts the branch from main, and the
    # bootstrap one picks up a branch you already created by hand (e.g. to park
    # in-flight work off main before adopting this process).
    local current_branch
    current_branch="$(git rev-parse --abbrev-ref HEAD)"
    if [ "${current_branch}" = "${branch}" ]; then
        echo "Already on '${branch}'; bumping it in place."
    else
        if git rev-parse -q --verify "refs/heads/${branch}" >/dev/null \
            || git ls-remote --exit-code --heads origin "${branch}" >/dev/null 2>&1; then
            echo "Branch '${branch}' already exists but you are on" \
                 "'${current_branch}'." >&2
            echo "Check it out (git checkout ${branch}) and re-run, or pick a" >&2
            echo "different version." >&2
            exit 1
        fi

        # Exactly one candidate is open at a time. Two would both be bumping
        # version strings and both expecting to fast-forward main, and only one
        # of them could be right. A candidate that is already contained in main
        # has shipped and is just leftover local state, so tidy it away instead
        # of complaining about it.
        local candidate leftover=()
        while IFS= read -r candidate; do
            [ -n "${candidate}" ] || continue
            if git merge-base --is-ancestor "${candidate}" origin/main 2>/dev/null; then
                leftover+=("${candidate}")
            else
                echo "Candidate branch '${candidate}' is still open." >&2
                echo "Ship it (scripts/build-all-platforms.sh) before starting" >&2
                echo "another, or delete it if it was abandoned:" >&2
                echo "  git push origin --delete ${candidate}; git branch -D ${candidate}" >&2
                exit 1
            fi
        done < <( { git for-each-ref --format='%(refname:short)' \
                'refs/heads/dev-v*' 'refs/remotes/origin/dev-v*'; } 2>/dev/null \
            | sed -E 's#^origin/##' | sort -u -V )

        # Get to a current main ourselves rather than making you do it: after a
        # release the local branch is whatever the last cycle used and main is
        # behind. The tree is already known clean, so this cannot lose work.
        if [ "${current_branch}" != "main" ]; then
            echo "Switching from '${current_branch}' to main."
            git checkout main
        fi
        if [ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]; then
            echo "Fast-forwarding main to origin/main."
            if ! git merge --ff-only origin/main; then
                echo "Local main has diverged from origin/main and cannot be" >&2
                echo "fast-forwarded. Nothing should ever commit to main; sort" >&2
                echo "that out before starting a candidate." >&2
                exit 1
            fi
        fi

        # After the checkout, so a leftover we were standing on is deletable.
        for candidate in "${leftover[@]:-}"; do
            [ -n "${candidate}" ] || continue
            echo "Cleaning up shipped candidate branch '${candidate}'."
            git branch -D "${candidate}" >/dev/null 2>&1 || true
        done

        echo "Creating branch '${branch}' from main."
        git checkout -b "${branch}"
    fi

    # If the bump fails, leave the branch as it was rather than half-rewritten.
    restore() {
        git checkout -- . >/dev/null 2>&1 || true
    }
    trap restore ERR

    echo "Rewriting version strings (${OLD_VERSION} -> ${NEW_VERSION})..."
    "${SCRIPTS_DIR}/update-version.sh" "${OLD_VERSION}" "${NEW_VERSION}"

    git commit -a -m "Update to version ${NEW_VERSION}"

    echo "Pushing branch to origin..."
    git push -u origin "${branch}"

    trap - ERR

    cat <<EOF

Candidate branch ${branch} is ready and you are standing on it.

All development for this cycle goes here, not on main. When it is time to
ship, rehearse the whole release first (builds and tests everything, uploads
nothing):

  scripts/build-all-platforms.sh

Then ship it for real:

  scripts/build-all-platforms.sh publish

That builds ${branch}, publishes every artifact, refreshes the ${tag} tag, and
finishes by fast-forwarding main to the released commit.
EOF
}

main "$@"
