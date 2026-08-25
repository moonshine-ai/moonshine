#! /bin/bash
# Freeze the Hugging Face voice-asset mirror at the version this release ships.
#
# Usage:
#   scripts/tag-hf-voice-assets.sh
#
# Called as a publish-only stage of scripts/build-all-platforms.sh. Tags
# moonshine-ai/moonshine-voice-assets with v<version> pointing at current
# `main`, so a later checkout of this git revision can fetch the matching
# snapshot instead of whatever HF `main` has become. Dry runs skip this stage.
#
# Idempotent: if the tag already exists it is left alone. HF tags are not
# moved; a changed snapshot is a new patch version.
#
# Requires the `hf` CLI, authenticated to an account that can write the
# moonshine-ai org (hf auth login).

set -euo pipefail

VERSION="0.1.6"
HF_REPO="${MOONSHINE_HF_REPO:-moonshine-ai/moonshine-voice-assets}"
TAG="v${VERSION}"

if ! command -v hf >/dev/null 2>&1; then
  echo "error: hf CLI not found; install with: pip install huggingface_hub" >&2
  echo "       then: hf auth login" >&2
  exit 1
fi

whoami="$(hf auth whoami 2>/dev/null || true)"
if [[ -z "${whoami}" ]]; then
  echo "error: hf is not authenticated; run: hf auth login" >&2
  exit 1
fi
echo "Authenticated to Hugging Face as: ${whoami}"
echo "Tagging ${HF_REPO} with ${TAG} at revision main."

if hf repos tag list "${HF_REPO}" | grep -Eq "(^|[[:space:]])${TAG}($|[[:space:]])"; then
  echo "Tag ${TAG} already exists on ${HF_REPO}; leaving it in place."
  exit 0
fi

hf repos tag create "${HF_REPO}" "${TAG}" \
  --message "Moonshine Voice ${TAG}" \
  --revision main

echo "Created ${HF_REPO} tag ${TAG}."
