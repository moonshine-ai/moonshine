# Release Process

`main` only ever contains released code. All development happens on a
`dev-v<version>` candidate branch, which is folded into `main` when the release
ships. There are exactly two commands in the whole cycle: one to start it, and
one to build it — run once as a rehearsal, then again with `publish`.

## Step 1 — start a cycle

Pick the version this cycle will ship, then:

```bash
scripts/start-candidate.sh 0.1.1
```

This syncs `main`, cuts `dev-v0.1.1` from it, rewrites every version string in
the repo, commits, pushes, and leaves you standing on the new branch. You can
run it from any branch as long as your working tree is clean; it handles getting
to `main` itself.

## Step 2 — develop

Commit to `dev-v0.1.1` exactly as you used to commit to `main`. Pull requests
target the candidate branch. Nothing else to do.

Optional, any time: check the branch is still shippable without starting a
build.

```bash
scripts/preflight-release.sh dev-v0.1.1
```

## Step 3 — rehearse

```bash
scripts/build-all-platforms.sh
```

Publishing is opt-in, so this is a dry run: every stage builds, tests and
packages on every platform, and nothing leaves the machine. No tag is moved, no
registry is touched, `main` stays where it is, and the Swift publish and
`finish-release` stages are skipped outright because they have no non-publishing
form. Android builds via `publishToMavenLocal`, so the Gradle publication is
still assembled, into `~/.m2` rather than Maven Central — which is where the
example builds then pick it up.

## Step 4 — ship

```bash
scripts/build-all-platforms.sh publish
```

This runs preflight, builds the newest `dev-v*` branch, publishes to PyPI, Maven
Central, npm, SwiftPM and GitHub Releases, and finishes by fast-forwarding
`main` to what it just shipped and deleting the candidate branch.

It is resumable — completed stages are skipped on a re-run, so if something
fails, fix it, commit to the candidate branch, and run it again. Dry-run
breadcrumbs are kept separately (`.release-state/<version>-dryrun/`), so a
rehearsal can never make the real release skip an upload stage.

Then go back to step 1 for the next version.

## What the checks catch

`start-candidate.sh` refuses to run if your working tree is dirty, if the
version already has a tag, if a candidate branch is already open, if the new
version is lower than the current one, or if local `main` has diverged from
`origin/main`.

`preflight-release.sh` runs before the first artifact is published — the point
where mistakes stop being cheap, because no registry lets you re-upload a
version. It fails the release if:

- the branch name and the version in the tree disagree, i.e. the bump never ran
- any version-bearing file is out of sync with the rest
- that version is already published and this is a fresh (non-resumed) run
- `gh` is missing or not authenticated
- the branch has unpushed commits, so the remote hosts would build other code
- `origin/main` is not an ancestor of the build commit, so the final fold-in
  into `main` would fail
- `.env` or `.pypirc` is missing from the repo root

`finish-release.sh` pushes a plain, non-force update to `main`, so the remote
rejects it if it is not a fast-forward.

## Why it works this way

Most people never build from source; they install a pinned binary that only
changes when a release is minted. GitHub renders the default branch's README on
the repo landing page, so when `main` is a development trunk the landing page
documents APIs that don't exist in anything users can install. Freezing `main`
at the last release makes the docs and the binaries describe the same software.

Bumping the version when the candidate is *cut* rather than just before
publishing means every doc, sample and download URL on the branch names the
version it will actually ship with, and gets tested against it all cycle.

## Rules

**Never commit to `main`.** The fold-in is a fast-forward, so a direct commit to
`main` breaks it and preflight will block the next release until you rebase the
candidate.

**Only one candidate branch at a time.** Two would both be bumping version
strings and both expecting to fast-forward `main`.

**The candidate is never named `vX.Y.Z`.** Git resolves an ambiguous `vX.Y.Z` to
the *tag*, not the branch, so a branch sharing the release tag's name silently
gives you a detached checkout of the tag. Hence the `dev-` prefix.

**Fixes for an already-published version get a new patch version.** PyPI, Maven
and GitHub release assets all reject re-uploading an existing version.

**You can keep developing during a release build.** It resolves the branch HEAD
to a single commit up front and pins every local and remote host to that sha, so
commits pushed mid-build are not picked up by later stages.

## Caveats

`releases/latest` URLs point at the *previous* release while you are on a
candidate branch, so running `scripts/test-examples.sh` without
`--local-examples` builds new example sources against old binaries. The release
path is unaffected; `publish-examples.sh` passes `--local-examples`.

In a dry run it passes `--local-library` instead. The examples pin an exact
library version — Android from Maven Central, iOS from the `moonshine-swift`
repo — which does not exist yet when nothing has been published, so a rehearsal
builds them against this checkout's AAR and Swift package.

Docs on the candidate branch still describe unreleased software. That's by
design — it's `main` that has to be trustworthy.
