#!/usr/bin/env python3
"""List the paths a Git LFS history purge should delete.

Prints every path that ever held an LFS object and is not present on the keep
ref (`main` by default), one per line, for feeding to
`git filter-repo --invert-paths --paths-from-file`. See docs/lfs-purge.md for
the surrounding procedure.

The path set is derived by walking the tree of every commit rather than by
reading `git lfs ls-files`. `ls-files` deduplicates by object ID and reports a
single representative path per object, so any asset committed under two paths
(the same model under `examples/android/...` and `examples/ios/...`, say)
appears only once. Removing the reported path leaves the object reachable
through the path that was hidden, and it keeps being billed. Walking trees
recovers every (path, object) pair; on this repo that was the difference
between 577 paths and the true 714.

Anything present on the keep ref is excluded, so the current checkout is
untouched: the rewrite changes history without changing content, and the root
tree of the keep ref is identical before and after.

Usage:
    # From inside a bare mirror clone (see docs/lfs-purge.md):
    python3 /path/to/scripts/lfs-purge-paths.py > /tmp/lfs_remove.txt
    python3 /path/to/scripts/lfs-purge-paths.py --keep-ref main --stats
"""

import argparse
import subprocess
import sys

LFS_POINTER_PREFIX = "version https://git-lfs"

# LFS pointer files are a few short lines; this bounds how many blobs we read.
MAX_POINTER_BYTES = 300


def git(*args, check=True):
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, errors="replace", check=False
    )
    if check and result.returncode != 0:
        sys.exit(f"git {' '.join(args)} failed:\n{result.stderr.strip()}")
    return result.stdout


def git_batch(command, payload):
    result = subprocess.run(
        ["git", "cat-file", command],
        input=payload,
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    return result.stdout


def find_lfs_pointer_blobs():
    """Return {blob sha: payload size} for every LFS pointer in the object db."""
    catalog = subprocess.run(
        [
            "git",
            "cat-file",
            "--batch-all-objects",
            "--batch-check=%(objectname) %(objecttype) %(objectsize)",
        ],
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    ).stdout

    candidates = []
    for line in catalog.splitlines():
        fields = line.split()
        is_small_blob = (
            len(fields) == 3
            and fields[1] == "blob"
            and int(fields[2]) < MAX_POINTER_BYTES
        )
        if is_small_blob:
            candidates.append(fields[0])
    if not candidates:
        return {}

    body = git_batch("--batch", "\n".join(candidates) + "\n").split("\n")
    pointers = {}
    for index, header in enumerate(body):
        fields = header.split()
        if len(fields) != 3 or fields[1] != "blob":
            continue
        if index + 1 >= len(body):
            continue
        if not body[index + 1].startswith(LFS_POINTER_PREFIX):
            continue
        size = 0
        for offset in range(index + 1, min(index + 5, len(body))):
            if body[offset].startswith("size "):
                size = int(body[offset].split()[1])
                break
        pointers[fields[0]] = size
    return pointers


def paths_holding(pointers):
    """Map every LFS pointer blob to all paths it occupies in any commit."""
    commits = git("rev-list", "--all").split()
    trees = git_batch(
        "--batch-check=%(objectname)", "".join(f"{c}^{{tree}}\n" for c in commits)
    ).split()

    paths = {}
    for tree in sorted(set(trees)):
        for line in git("ls-tree", "-r", "--full-tree", tree).split("\n"):
            if not line:
                continue
            meta, _, path = line.partition("\t")
            fields = meta.split()
            if len(fields) >= 3 and fields[2] in pointers:
                paths.setdefault(fields[2], set()).add(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep-ref",
        default="main",
        help="paths present on this ref are never removed (default: main)",
    )
    parser.add_argument(
        "--stats", action="store_true", help="write a summary to stderr"
    )
    args = parser.parse_args()

    pointers = find_lfs_pointer_blobs()
    if not pointers:
        sys.exit("No LFS pointer blobs found. Is this a mirror clone of the repo?")

    occupied = paths_holding(pointers)
    missing = len(pointers) - len(occupied)
    if missing:
        print(
            f"warning: {missing} LFS blobs are unreachable from any commit",
            file=sys.stderr,
        )

    keep = set(git("ls-tree", "-r", "--name-only", args.keep_ref).splitlines())
    every_path = set()
    for found in occupied.values():
        every_path |= found
    remove = sorted(every_path - keep)

    if args.stats:
        kept_bytes = sum(
            size for sha, size in pointers.items() if occupied.get(sha, set()) & keep
        )
        total = sum(pointers.values())
        duplicated = sum(1 for found in occupied.values() if len(found) > 1)
        print(
            f"LFS objects:        {len(pointers)}\n"
            f"  at >1 path:       {duplicated}\n"
            f"paths ever holding: {len(every_path)}\n"
            f"  kept ({args.keep_ref}): {len(every_path & keep)}\n"
            f"  removed:          {len(remove)}\n"
            f"payload total:      {total / 1024**3:.2f} GB\n"
            f"  irreducible:      {kept_bytes / 1024**3:.2f} GB",
            file=sys.stderr,
        )

    print("\n".join(remove))


if __name__ == "__main__":
    main()
