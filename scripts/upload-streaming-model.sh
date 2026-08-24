#!/usr/bin/env bash
# Publish one quantized streaming model to the Moonshine download bucket, so that
# the URL in core/moonshine-model-catalog.cpp resolves:
#   https://download.moonshine.ai/model/<name>/<dir>/encoder.ort
#
# Streaming models are eight files that only make sense together: five graphs
# (the frontend split into a model and a weights blob), the architecture that
# describes their state shapes, and the tokenizer that decodes their output. A
# directory holding seven of them is a model that loads and produces nonsense, so
# the list below is checked rather than globbed.
#
# The catalog pins a dated directory and this script refuses to write into one
# that already exists. That is the rollback story: older library versions keep
# resolving the bytes they were tested against, reverting is a one-line change to
# the catalog, and clients that key their download cache off the URL re-fetch
# cleanly instead of mixing old and new graphs. It also means no cache purge is
# needed, since every object written is new -- leave MOONSHINE_INVALIDATE_CDN
# alone.
#
# After running this, regenerate the integrity metadata, which reads sizes and
# checksums back off the CDN, and commit the result:
#   python3 scripts/generate-model-file-metadata.py
#
# Prerequisites: rclone, and the Cloudflare R2 credentials described in
# cdn-publish-common.sh.
#
#   scripts/upload-streaming-model.sh ~/dir small-streaming-ja quantized_26_08_23
#
# Environment:
#   MOONSHINE_R2_BUCKET            Bucket name (default: download-moonshine-ai)
#   MOONSHINE_MODEL_CACHE_CONTROL  Cache-Control for uploaded objects
#                                  (default: "public, max-age=2592000"; 30 days).
#   MOONSHINE_ALLOW_EXISTING_DIR   Write into a directory that already has
#                                  objects. Only for repairing a partial upload.
#                                  Safe to use: the per-file guard in
#                                  cdn_publish_file still refuses to change any
#                                  object that is already there with different
#                                  content, so this can fill gaps but cannot
#                                  rewrite a published graph.
#
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <source-dir> <model-name> <quantized-dir>" >&2
  echo "   eg: $0 ja_release/deploy_tiny tiny-streaming-ja quantized_26_08_23" >&2
  exit 2
fi

SRC="$(cd "$1" && pwd)"
MODEL_NAME="$2"
QUANTIZED_DIR="$3"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_CONTROL="${MOONSHINE_MODEL_CACHE_CONTROL:-public, max-age=2592000}"

# Must match stt_component_files() in core/moonshine-model-catalog.cpp for a
# streaming architecture. English additionally publishes
# decoder_kv_with_attention.ort for word timestamps; nothing else does yet.
FILES=(adapter.ort cross_kv.ort decoder_kv.ort encoder.ort
       frontend.model.ort frontend.weights.ort
       streaming_config.json tokenizer.bin)

if [[ ! "${QUANTIZED_DIR}" =~ ^quantized_[0-9]{2}_[0-9]{2}_[0-9]{2}$ ]]; then
  echo "Directory should look like quantized_YY_MM_DD, got '${QUANTIZED_DIR}'." >&2
  exit 1
fi

for file in "${FILES[@]}"; do
  if [[ ! -f "${SRC}/${file}" ]]; then
    echo "Missing ${SRC}/${file}; a partial streaming model is not publishable." >&2
    exit 1
  fi
  # ORT flatbuffers carry "ORTM" as the file identifier at offset 4. Catches a
  # truncated copy, and catches handing this script a directory of .onnx.
  if [[ "${file}" == *.ort ]]; then
    if [[ "$(dd if="${SRC}/${file}" bs=1 skip=4 count=4 2>/dev/null)" != "ORTM" ]]; then
      echo "${SRC}/${file} is not an ORT model (no ORTM magic)." >&2
      exit 1
    fi
  fi
done
# tokenizer.bin has no magic number to check, so check it is not empty: an
# unusable model with the right file list is the failure this whole script exists
# to prevent.
if [[ ! -s "${SRC}/tokenizer.bin" ]]; then
  echo "${SRC}/tokenizer.bin is empty." >&2
  exit 1
fi

source "${ROOT}/scripts/cdn-publish-common.sh"
cdn_setup_r2

REMOTE_DIR="model/${MODEL_NAME}/${QUANTIZED_DIR}"
DEST="r2:${CDN_R2_BUCKET}/${REMOTE_DIR}"

if [[ -z "${MOONSHINE_ALLOW_EXISTING_DIR:-}" ]]; then
  existing="$(rclone lsf "${DEST}/" 2>/dev/null | head -n 1 || true)"
  if [[ -n "${existing}" ]]; then
    echo "${REMOTE_DIR} already exists (found '${existing}')." >&2
    echo "Publish a new dated directory instead of overwriting a pinned one." >&2
    exit 1
  fi
fi

echo "Upload ${SRC} -> ${DEST} (Cache-Control: ${CACHE_CONTROL})" >&2
# --header-upload is how the caching header reaches R2; rclone drops metadata
# that is not asked for explicitly, and an object uploaded without it is served
# with no Cache-Control at all.
for file in "${FILES[@]}"; do
  cdn_publish_file "${SRC}/${file}" "${DEST}/${file}" "${CACHE_CONTROL}"
done

echo >&2
echo "Published https://${CDN_HOST}/${REMOTE_DIR}/" >&2
rclone lsl "${DEST}/" >&2
echo >&2
echo "Now: update kJapaneseStreamingQuantizedDir (or equivalent) and stt_catalog()" >&2
echo "in core/moonshine-model-catalog.cpp, MODEL_INFO in" >&2
echo "language-bindings/python/src/moonshine_voice/download.py, then run" >&2
echo "python3 scripts/generate-model-file-metadata.py and commit." >&2
