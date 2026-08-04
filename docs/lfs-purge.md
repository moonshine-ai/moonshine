# Purging Git LFS history (storage billing)

Forward-only deletes of model/TTS binaries shrink future clones but **do not**
reduce GitHub LFS storage charges. GitHub bills for every unique LFS object still
reachable from the remote until those objects are purged.

Do this only after the fetch-based workflow
([`scripts/fetch-voice-assets.sh`](../scripts/fetch-voice-assets.sh)) is on the
default branch and the team agrees to a force-push + re-clone.

## Preconditions

1. All removed LFS paths are gone from `main` (or your default branch).
2. Contributors can bootstrap with `scripts/fetch-voice-assets.sh all`.
3. Announce a freeze: no pushes until rewrite + purge complete.
4. Ensure you have org admin rights and a backup clone (`git clone --mirror`).

## Rewrite history (strip LFS payloads)

From a fresh mirror clone:

```bash
git clone --mirror git@github.com:moonshine-ai/moonshine.git moonshine-lfs-purge.git
cd moonshine-lfs-purge.git

# Export listed paths out of LFS / remove their blobs from history.
# Adjust the path list to match what was deleted from HEAD.
git lfs migrate export --everything \
  --include="core/moonshine-tts/data/**,\
examples/**/tts-data/**,\
test-assets/**/*.ort,\
test-assets/tiny-en/tokenizer.bin,\
test-assets/tiny-streaming-en/**,\
python/src/moonshine_voice/assets/tiny-en/*.ort,\
android/java/androidTest/assets/tiny-en/*"
```

Alternatively use [`git filter-repo`](https://github.com/newren/git-filter-repo)
with `--path` / `--path-glob` removals for the same trees, then
`git lfs prune`.

Force-push all refs (coordinate with the org):

```bash
git push --force --all
git push --force --tags
```

Every clone and fork must re-clone; old remotes will disagree on history.

## Request a GitHub LFS purge

Rewriting history orphans LFS objects but **does not** free billed storage until
GitHub deletes them. Open a ticket with GitHub Support asking to purge orphaned
LFS objects for `moonshine-ai/moonshine` after the force-push.

Include:

- Repository name
- Approximate time of the force-push
- Confirmation that history was rewritten and old LFS pointers are gone

## Verify

```bash
git lfs ls-files | wc -l   # should be ~ORT libs + embedded .cpp only
du -sh .git/lfs            # local cache; prune with: git lfs prune
```

Check the org’s GitHub LFS storage graph after Support confirms the purge.
