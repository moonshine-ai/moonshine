#! /bin/bash

# Fold one or more fixes into an already-prepared release by cherry-picking them
# onto a fresh release branch and cutting a new patch version.
#
# Usage:
#   scripts/patch-release.sh <new_version> <commit-ish> [<commit-ish> ...]
#
# Example (fix already committed on main as abc1234):
#   scripts/patch-release.sh 0.0.70 abc1234
#
# Why a new version number: package registries (PyPI, Maven, GitHub release
# assets, etc.) reject re-uploading an existing version, so a re-publish always
# needs a fresh version.
#
# What it does:
#   1. Uses the most recent v* tag as the base (the last release you cut).
#   2. Creates branch `release-v<new_version>` from that tag.
#   3. Cherry-picks the commits you name onto it.
#   4. Runs update-version.sh to bump the version strings.
#   5. Commits, tags `v<new_version>`, and pushes the branch + tag.
#   6. Switches you back to `main`.
#
# Afterwards re-run:
#   scripts/build-all-platforms.sh
# which will build/publish the new frozen tag.

set -euo pipefail

main() {
    if [ $# -lt 2 ]; then
        echo "Usage: $0 <new_version> <commit-ish> [<commit-ish> ...]" >&2
        exit 1
    fi

    NEW_VERSION="$1"
    shift
    COMMITS=("$@")

    if ! [[ "${NEW_VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "New version '${NEW_VERSION}' is not in X.Y.Z form (no leading 'v')." >&2
        exit 1
    fi

    SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
    REPO_ROOT_DIR="$(dirname "${SCRIPTS_DIR}")"
    cd "${REPO_ROOT_DIR}"

    if ! git diff --quiet || ! git diff --cached --quiet; then
        echo "Your working tree has uncommitted changes to tracked files." >&2
        echo "Commit or stash them before cutting a patch release." >&2
        exit 1
    fi

    echo "Fetching latest refs from origin..."
    git fetch origin --tags --prune

    # Resolve every commit-ish up front so a typo fails before we branch.
    local resolved=()
    local c sha
    for c in "${COMMITS[@]}"; do
        if ! sha="$(git rev-parse -q --verify "${c}^{commit}")"; then
            echo "Could not resolve commit '${c}'." >&2
            exit 1
        fi
        resolved+=("${sha}")
    done

    local base_tag
    # Highest-version v* tag, NOT `git describe` (which only sees tags reachable
    # from HEAD); release tags live on release-v* branches off main.
    base_tag="$(git tag -l 'v*' --sort=-v:refname | head -n1)"
    if [ -z "${base_tag}" ]; then
        echo "Could not find a previous v* tag to base the patch on." >&2
        echo "Use scripts/prepare-release.sh to cut the first release." >&2
        exit 1
    fi
    local old_version="${base_tag#v}"

    if [ "${old_version}" = "${NEW_VERSION}" ]; then
        echo "New version '${NEW_VERSION}' matches the base tag '${base_tag}'." >&2
        echo "Pick a higher version number." >&2
        exit 1
    fi

    local tag="v${NEW_VERSION}"
    local branch="release-v${NEW_VERSION}"

    if git rev-parse -q --verify "refs/tags/${tag}" >/dev/null \
        || git ls-remote --exit-code --tags origin "${tag}" >/dev/null 2>&1; then
        echo "Tag '${tag}' already exists (locally or on origin)." >&2
        exit 1
    fi
    if git rev-parse -q --verify "refs/heads/${branch}" >/dev/null \
        || git ls-remote --exit-code --heads origin "${branch}" >/dev/null 2>&1; then
        echo "Branch '${branch}' already exists (locally or on origin)." >&2
        exit 1
    fi

    echo "Cutting patch release ${tag} from base ${base_tag}."
    echo "Cherry-picking: ${resolved[*]}"
    git checkout -b "${branch}" "${base_tag}"

    if ! git cherry-pick "${resolved[@]}"; then
        cat <<EOF >&2

Cherry-pick hit a conflict on branch '${branch}'. Nothing has been pushed.

To finish by hand:
  1. Resolve the conflicts, then: git cherry-pick --continue
     (repeat until all picks are applied)
  2. scripts/update-version.sh ${old_version} ${NEW_VERSION}
  3. git commit -a -m "Update to version ${NEW_VERSION}"
  4. git tag -a ${tag} -m "Release ${tag}"
  5. git push origin ${branch} && git push origin ${tag}
  6. git checkout main

Or to bail out entirely:
  git cherry-pick --abort && git checkout main && git branch -D ${branch}
EOF
        exit 1
    fi

    # From here on, undo the branch on failure and return to main.
    restore_main() {
        git checkout -f main >/dev/null 2>&1 || true
        git branch -D "${branch}" >/dev/null 2>&1 || true
    }
    trap restore_main ERR

    echo "Rewriting version strings..."
    "${SCRIPTS_DIR}/update-version.sh" "${old_version}" "${NEW_VERSION}"

    git commit -a -m "Update to version ${NEW_VERSION}"
    git tag -a "${tag}" -m "Release ${tag}"

    echo "Pushing branch and tag to origin..."
    git push origin "${branch}"
    git push origin "${tag}"

    trap - ERR
    git checkout main

    cat <<EOF

Patch release ${tag} is prepared and pushed.
  base:   ${base_tag}
  branch: ${branch}
  tag:    ${tag}

You are back on 'main'. To build and publish it, run:

  scripts/build-all-platforms.sh
EOF
}

main "$@"
