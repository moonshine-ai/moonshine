#! /bin/bash

# Fold a fully published candidate branch into main, so main once again matches
# the binaries people can actually install. Normally run as the last stage of
# scripts/build-all-platforms.sh rather than by hand.
#
# Usage:
#   scripts/finish-release.sh <branch> [<commit>]
#
# Example:
#   scripts/finish-release.sh dev-v0.1.0
#
# main only ever moves forward to a commit that already contains all of it, so
# this is a fast-forward and never a merge. That is enforced twice: locally by
# an ancestry check, and by pushing a plain (non-force) update that the remote
# will reject if it is not a fast-forward. A rejection means something was
# committed directly to main, which the process forbids.
#
# It runs safely from a detached worktree (as build-all-platforms.sh does): it
# pushes refs rather than checking main out, since main is already checked out
# in the primary worktree.

set -euo pipefail

main() {
    if [ $# -lt 1 ] || [ $# -gt 2 ]; then
        echo "Usage: $0 <branch> [<commit>]" >&2
        exit 1
    fi

    local branch="$1"
    local commit="${2:-}"

    SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
    REPO_ROOT_DIR="$(dirname "${SCRIPTS_DIR}")"
    cd "${REPO_ROOT_DIR}"

    git fetch origin --tags --prune --force

    if [ -z "${commit}" ]; then
        if ! commit="$(git rev-parse -q --verify "origin/${branch}^{commit}")"; then
            echo "Branch '${branch}' not found on origin." >&2
            exit 1
        fi
    fi
    if ! commit="$(git rev-parse -q --verify "${commit}^{commit}")"; then
        echo "Could not resolve the commit to fold into main." >&2
        exit 1
    fi

    local main_commit
    main_commit="$(git rev-parse origin/main)"

    if [ "${main_commit}" = "${commit}" ]; then
        echo "main is already at ${commit}; nothing to fold in."
        return 0
    fi

    if ! git merge-base --is-ancestor "${main_commit}" "${commit}"; then
        cat <<EOF >&2
Cannot fast-forward main to ${commit}.

origin/main (${main_commit}) is not an ancestor of the released commit, which
means something was committed directly to main. Nothing under this process
should ever do that.

Reconcile by rebasing the candidate onto main and re-running the release:

  git checkout ${branch}
  git rebase origin/main
  git push --force-with-lease origin ${branch}
EOF
        exit 1
    fi

    echo "Fast-forwarding main: ${main_commit} -> ${commit}"
    git push origin "${commit}:refs/heads/main"

    echo "Deleting candidate branch ${branch} now that it is merged."
    git push origin --delete "${branch}" || \
        echo "Warning: could not delete origin/${branch}; remove it by hand." >&2
    git branch -D "${branch}" >/dev/null 2>&1 || true

    cat <<EOF

main now points at the released commit, so the landing-page docs match the
published binaries.

Update your local checkout and start the next cycle:

  git checkout main && git pull --ff-only
  scripts/start-candidate.sh <next_version>
EOF
}

main "$@"
