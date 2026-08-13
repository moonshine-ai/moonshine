#! /bin/bash

# Everything that must be true before a release build starts. Run automatically
# by scripts/build-all-platforms.sh; you can also run it by hand at any point in
# a development cycle to confirm the branch is shippable.
#
# Usage:
#   scripts/preflight-release.sh [--dry-run] <branch> [<commit>]
#
# <commit> defaults to the branch's pushed tip and may be any commit-ish.
#
# With --dry-run, credentials that only the upload stages use are reported as
# warnings instead of failures, because a dry run never reaches the code that
# needs them. Blocking a rehearsal on a credential it will not touch just
# teaches you to skip preflight altogether.
#
# These checks exist because the failures they catch are expensive: the publish
# stages push to PyPI, Maven Central and GitHub Releases, none of which let
# you re-upload a version. npm (@moonshine-ai/moonshine-wasm) is a separate
# manual step via scripts/publish-wasm-npm.sh. A mistake found four hours in,
# after half the registries have accepted an artifact, costs a burned version
# number. Every check here is one that can be made cheaply up front.

set -uo pipefail

FAILURES=0
DRY_RUN=""

fail() {
    echo "  FAIL: $1" >&2
    shift
    while [ $# -gt 0 ]; do
        echo "        $1" >&2
        shift
    done
    FAILURES=$((FAILURES + 1))
}

warn() {
    echo "  WARN: $1" >&2
    shift
    while [ $# -gt 0 ]; do
        echo "        $1" >&2
        shift
    done
}

# For problems that only matter once something is actually uploaded.
publish_only_fail() {
    if [ -n "${DRY_RUN}" ]; then
        warn "$@" "(not fatal for a dry run, but fix it before publishing)"
    else
        fail "$@"
    fi
}

pass() {
    echo "  ok: $1"
}

main() {
    if [ "${1:-}" = "--dry-run" ]; then
        DRY_RUN=1
        shift
    fi
    if [ $# -lt 1 ] || [ $# -gt 2 ]; then
        echo "Usage: $0 [--dry-run] <branch> [<commit>]" >&2
        exit 1
    fi

    local branch="$1"
    local version="${branch#dev-v}"

    if [ "${branch}" = "${version}" ]; then
        echo "'${branch}' is not a dev-v<version> candidate branch." >&2
        exit 1
    fi

    SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
    REPO_ROOT_DIR="$(dirname "${SCRIPTS_DIR}")"
    cd "${REPO_ROOT_DIR}"

    # Resolve to a sha up front. Several checks compare against one, and a
    # symbolic ref would silently compare unequal to its own resolved tip.
    local commit
    if ! commit="$(git rev-parse -q --verify "${2:-origin/${branch}}^{commit}")"; then
        echo "Could not resolve '${2:-origin/${branch}}' to a commit." >&2
        exit 1
    fi

    echo "Preflight checks for ${branch} at ${commit}:"

    # The single most costly mistake: building a candidate whose version strings
    # were never bumped. It publishes artifacts labelled with the previous
    # version, which the registries then refuse to overwrite, and force-moves
    # the new tag onto them.
    local file_version
    file_version="$(git show "${commit}:core/CMakeLists.txt" 2>/dev/null \
        | sed -n 's/^set(MOONSHINE_VERSION "\([^"]*\)").*/\1/p' | head -n1)"
    if [ -z "${file_version}" ]; then
        fail "could not read MOONSHINE_VERSION from core/CMakeLists.txt at ${commit}."
    elif [ "${file_version}" != "${version}" ]; then
        fail "branch says ${version} but the tree says ${file_version}." \
            "The version bump never ran. Fix with:" \
            "  git checkout ${branch} && scripts/start-candidate.sh ${version}"
    else
        pass "branch name and tree agree on version ${version}"
    fi

    # Catches a version string added during the cycle that update-version.sh
    # does not know about, which would otherwise ship pointing at the old
    # release.
    if "${SCRIPTS_DIR}/update-version.sh" --verify "${version}" >/dev/null 2>&1; then
        pass "all version-bearing files are at ${version}"
    else
        fail "version strings are inconsistent across the repo. Details:" \
            "  scripts/update-version.sh --verify ${version}"
    fi

    # Registries reject re-uploading a version, so a fresh run against an
    # already-published one cannot succeed. A resumed run legitimately finds the
    # release already there, so only complain when there are no breadcrumbs.
    local state_dir="${REPO_ROOT_DIR}/.release-state/${version}"
    if command -v gh >/dev/null 2>&1; then
        if gh release view "v${version}" >/dev/null 2>&1; then
            if compgen -G "${state_dir}/*.done" >/dev/null; then
                pass "v${version} release exists (resuming an in-progress build)"
            else
                fail "v${version} is already published on GitHub Releases and" \
                    "this is a fresh run. PyPI, Maven and npm will reject the" \
                    "re-upload. Start a new version instead:" \
                    "  scripts/start-candidate.sh <next_version>"
            fi
        else
            pass "v${version} has not been published yet"
        fi
        if gh auth status >/dev/null 2>&1; then
            pass "gh is authenticated"
        else
            fail "gh is not authenticated; the upload stages will fail." \
                "  gh auth login"
        fi
        if command -v npm >/dev/null 2>&1; then
            if npm whoami >/dev/null 2>&1; then
                pass "npm is authenticated as $(npm whoami 2>/dev/null)"
            else
                # npm publish is no longer part of build-all-platforms; it is a
                # manual follow-up (scripts/publish-wasm-npm.sh) so a missing
                # login must not block the multi-hour release.
                warn "npm is not authenticated; run scripts/publish-wasm-npm.sh" \
                    "after the release (and npm login) so jsDelivr can serve" \
                    "@moonshine-ai/moonshine-wasm."
            fi
        else
            warn "npm is not installed; scripts/publish-wasm-npm.sh will need it" \
                "after the release to publish @moonshine-ai/moonshine-wasm."
        fi
    else
        fail "gh is not installed; the release upload stages need it."
    fi

    # The build is from the pushed commit, so anything local and uncommitted or
    # unpushed silently will not be in the release.
    local remote_tip
    remote_tip="$(git rev-parse -q --verify "refs/remotes/origin/${branch}")"
    if [ -z "${remote_tip}" ]; then
        fail "branch ${branch} does not exist on origin. Push it first:" \
            "  git push -u origin ${branch}"
    elif [ "${remote_tip}" != "${commit}" ]; then
        fail "origin/${branch} is ${remote_tip} but the build commit is" \
            "${commit}. Push the branch so the remote hosts build the same code."
    else
        pass "origin/${branch} matches the build commit"
    fi

    # finish-release.sh fast-forwards main at the very end. If that is going to
    # be impossible, say so now rather than after everything is published.
    local main_commit
    main_commit="$(git rev-parse -q --verify origin/main)"
    if [ -z "${main_commit}" ]; then
        fail "could not resolve origin/main."
    elif git merge-base --is-ancestor "${main_commit}" "${commit}"; then
        pass "main can fast-forward to the release commit"
    else
        fail "origin/main is not an ancestor of ${commit}, so the final" \
            "fold-in into main would fail. Something was committed directly to" \
            "main. Rebase the candidate first:" \
            "  git checkout ${branch} && git rebase origin/main" \
            "  git push --force-with-lease origin ${branch}"
    fi

    # Credentials the publish stages read from the repo root. They are
    # gitignored, so a fresh machine silently lacks them.
    local cfg
    for cfg in .env .pypirc; do
        if [ -f "${REPO_ROOT_DIR}/${cfg}" ]; then
            pass "${cfg} is present"
        else
            fail "${cfg} is missing from the repo root; publish stages need it."
        fi
    done

    if [ ${FAILURES} -gt 0 ]; then
        echo >&2
        echo "Preflight found ${FAILURES} problem(s); not starting the release." >&2
        exit 1
    fi
    echo "Preflight passed."
}

main "$@"
