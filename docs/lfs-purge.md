# Purging Git LFS history (storage billing)

Forward-only deletes of model/TTS binaries shrink future clones but **do not**
reduce GitHub LFS storage charges. GitHub bills for every unique LFS object still
reachable from the remote until those objects are purged.

Do this only after the fetch-based workflow
([`scripts/fetch-voice-assets.sh`](../scripts/fetch-voice-assets.sh)) is on the
default branch and the team agrees to a force-push + re-clone.

> **Status:** a purge was performed on 2026-08-07. It rewrote all 13 branches and
> 20 tags, taking the LFS payload from 7.07 GB (596 objects) to 1.33 GB
> (61 objects). The remaining 1.33 GB is described under
> [What cannot be reclaimed](#what-cannot-be-reclaimed).

## Preconditions

1. All removed LFS paths are gone from `main` (or your default branch).
2. Contributors can bootstrap with `scripts/fetch-voice-assets.sh all`.
3. Announce a freeze: no pushes until rewrite + purge complete.
4. Ensure you have org admin rights and a backup clone (`git clone --mirror`).

## Do not use `git lfs migrate export`

`git lfs migrate export` does not remove anything from history. It converts LFS
pointers back into ordinary Git blobs, inlining the payload into the packfiles.
That both bloats the repository and makes the push fail outright, because GitHub
hard-rejects any non-LFS blob over 100 MB and several of these assets are larger
than that. Use `git filter-repo` path removal instead.

## Derive the path list mechanically

Do not hand-maintain a glob list; it will be incomplete. In particular **do not**
build the list from `git lfs ls-files`, which deduplicates by object ID and
reports only one representative path per object. Identical assets committed under
two paths (for example the same model under `examples/android/...` and
`examples/ios/...`) share one object, so removing the single path that
`ls-files` reports leaves the object alive under the path it hid.

Walk every commit's tree instead. The rule is: delete every path that ever held
an LFS object and is not present on `main`.

```bash
git clone --mirror git@github.com:moonshine-ai/moonshine.git moonshine-lfs-purge.git
cd moonshine-lfs-purge.git
python3 ../scripts/lfs-purge-paths.py > /tmp/lfs_remove.txt
```

Sanity-check before rewriting — the list must not intersect `main`, and the
ONNX Runtime libraries and embedded `.cpp` sources must survive:

```bash
git ls-tree -r --name-only main | sort -u > /tmp/main_tree.txt
comm -12 <(sort -u /tmp/lfs_remove.txt) /tmp/main_tree.txt | wc -l   # must be 0
```

## Rewrite history

```bash
git filter-repo --invert-paths --paths-from-file /tmp/lfs_remove.txt --force
```

This rewrites pointer files only, so it never downloads LFS payloads and finishes
in seconds. Commits left empty by the removal are pruned, so the commit count
drops slightly.

Verify that the rewrite changed history without changing content — the root tree
of `main` must be unchanged:

```bash
git rev-parse main^{tree}   # must equal the pre-rewrite value
```

Force-push all refs (coordinate with the org). `filter-repo` removes the `origin`
remote, so add it back first. Push branches and tags only; `refs/pull/*` is
managed by GitHub and cannot be written:

```bash
git remote add origin git@github.com:moonshine-ai/moonshine.git
git push --force --all origin
git push --force --tags origin
```

Every clone and fork must re-clone; old remotes will disagree on history. In an
existing clean checkout you can instead
`git fetch origin --prune --prune-tags --force && git reset --hard origin/main`.

## Request a GitHub LFS purge

Rewriting history orphans LFS objects but **does not** free billed storage until
GitHub deletes them. Open a ticket with GitHub Support asking to purge orphaned
LFS objects for `moonshine-ai/moonshine` after the force-push.

Include:

- Repository name
- Approximate time of the force-push
- Confirmation that history was rewritten and old LFS pointers are gone

Pull request refs (`refs/pull/*`) keep the pre-rewrite commits reachable on
GitHub's side, which is another reason the Support purge is required.

## What cannot be reclaimed

Path deletion cannot free stale versions of files that still exist on `main`,
because removing the path would remove the current file too. That accounts for
the residual 1.33 GB, dominated by superseded ONNX Runtime builds — a single
obsolete `core/third-party/onnxruntime/lib/android/arm64/libonnxruntime.so` is
708 MB. Only 0.28 GB of the remainder is content `main` actually uses. Freeing
the rest means rewriting those historical pointers in place, which is
substantially more invasive than path removal.

Separately, the push surfaced large files that were committed to history outside
LFS entirely (for example `test-assets/slinkier-en/decoder.onnx` at 87 MB). Those
are ordinary Git blobs, so they do not affect the LFS bill, but they do inflate
clone size and would need the same `filter-repo` treatment to remove.

## Verify

```bash
git lfs ls-files | wc -l   # should be ~ORT libs + embedded .cpp only (11)
du -sh .git/lfs            # local cache; prune with: git lfs prune
```

Check the org's GitHub LFS storage graph after Support confirms the purge.
