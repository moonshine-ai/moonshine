#! /bin/bash

# Fold one or more fixes into the current in-progress release by cherry-picking
# them onto its release-v<version> branch. No version bump and no tag: the
# release branch is the mutable source of truth, and build-all-platforms.sh
# refreshes the v<version> tag from the branch HEAD on its next run.
#
# Usage:
#   scripts/patch-release.sh <commit-ish> [<commit-ish> ...]
#
# Example (fix already committed on main as abc1234):
#   scripts/patch-release.sh abc1234
#
# What it does:
#   1. Finds the newest release-v* branch (the release you're building).
#   2. Checks it out at origin's HEAD.
#   3. Cherry-picks the commits you name onto it.
#   4. Pushes the branch and returns you to main.
#
# Afterwards re-run:
#   scripts/build-all-platforms.sh
# It resumes (skipping completed stages) and the remaining stages pick up the
# new branch HEAD.
#
# IMPORTANT: this is for a release that is still in progress. To ship a fix for
# a version that is ALREADY fully published, cut a NEW version with
# scripts/prepare-release.sh instead -- package registries reject re-uploading
# an existing version.

set -euo pipefail

main() {
    if [ $# -lt 1 ]; then
        echo "Usage: $0 <commit-ish> [<commit-ish> ...]" >&2
        exit 1
    fi

    COMMITS=("$@")

    SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
    REPO_ROOT_DIR="$(dirname "${SCRIPTS_DIR}")"
    cd "${REPO_ROOT_DIR}"

    if ! git diff --quiet || ! git diff --cached --quiet; then
        echo "Your working tree has uncommitted changes to tracked files." >&2
        echo "Commit or stash them before folding in a fix." >&2
        exit 1
    fi

    echo "Fetching latest refs from origin..."
    git fetch origin --tags --prune --force

    # Resolve every commit-ish up front so a typo fails before we switch branch.
    local resolved=()
    local c sha
    for c in "${COMMITS[@]}"; do
        if ! sha="$(git rev-parse -q --verify "${c}^{commit}")"; then
            echo "Could not resolve commit '${c}'." >&2
            exit 1
        fi
        resolved+=("${sha}")
    done

    # Newest release-v* branch (highest version), from origin if present.
    local version
    version="$( { git for-each-ref --format='%(refname:short)' \
            'refs/heads/release-v*' 'refs/remotes/origin/release-v*'; } 2>/dev/null \
        | sed -E 's#^origin/##; s#^release-v##' \
        | grep -E '^[0-9]+\.[0-9]+\.[0-9]+$' \
        | sort -V | tail -n1 )"
    if [ -z "${version}" ]; then
        echo "No release-v* branch found to patch." >&2
        echo "Use scripts/prepare-release.sh to cut a release first." >&2
        exit 1
    fi
    local branch="release-v${version}"

    if ! git ls-remote --exit-code --heads origin "${branch}" >/dev/null 2>&1 \
        && ! git rev-parse -q --verify "refs/heads/${branch}" >/dev/null; then
        echo "Release branch '${branch}' not found on origin or locally." >&2
        exit 1
    fi

    echo "Folding ${#resolved[@]} commit(s) into ${branch}."
    echo "Cherry-picking: ${resolved[*]}"

    # Check the branch out at origin's HEAD (create/reset the local branch to
    # match what everything else builds).
    if git ls-remote --exit-code --heads origin "${branch}" >/dev/null 2>&1; then
        git checkout -B "${branch}" "origin/${branch}"
    else
        git checkout "${branch}"
    fi

    if ! git cherry-pick "${resolved[@]}"; then
        cat <<EOF >&2

Cherry-pick hit a conflict on branch '${branch}'. Nothing has been pushed.

To finish by hand:
  1. Resolve the conflicts, then: git cherry-pick --continue
     (repeat until all picks are applied)
  2. git push origin ${branch}
  3. git checkout main

Or to bail out entirely:
  git cherry-pick --abort && git checkout main
EOF
        exit 1
    fi

    # On any later failure, just return to main (the branch keeps the picks;
    # nothing destructive has happened).
    restore_main() {
        git checkout -f main >/dev/null 2>&1 || true
    }
    trap restore_main ERR

    echo "Pushing ${branch} to origin..."
    git push origin "${branch}"

    trap - ERR
    git checkout main

    cat <<EOF

Fix folded into ${branch} and pushed.

Re-run the build to continue where it left off:

  scripts/build-all-platforms.sh

It will resume (skipping completed stages) and the remaining stages will build
the updated branch HEAD.
EOF
}

main "$@"
