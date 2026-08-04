#!/usr/bin/env bash
# Bootstrap model/TTS binaries that used to live in Git LFS.
#
# Runtime SDKs still download from https://download.moonshine.ai. This script
# fills local trees used by offline tests, examples, and CDN publish:
#   test-assets/…              STT / diarization / embedding fixtures
#   core/moonshine-tts/data/…  TTS + G2P bundles (READMEs stay in git)
#   android/java/androidTest/assets/tiny-en/   (optional; mirrors CDN tiny-en)
#
# Primary source: CDN. Optional fallback for the full TTS tree: Hugging Face
#   moonshine-ai/moonshine-voice-assets  (paths match CDN under tts/ and model/).
#
# Usage:
#   scripts/fetch-voice-assets.sh              # test-assets + tts
#   scripts/fetch-voice-assets.sh test-assets
#   scripts/fetch-voice-assets.sh tts
#   scripts/fetch-voice-assets.sh android-test
#   scripts/fetch-voice-assets.sh all
#
# Environment:
#   MOONSHINE_CDN_BASE   default https://download.moonshine.ai
#   MOONSHINE_HF_REPO    default moonshine-ai/moonshine-voice-assets
#   MOONSHINE_FETCH_FORCE  if non-empty, re-download even when size matches
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CDN_BASE="${MOONSHINE_CDN_BASE:-https://download.moonshine.ai}"
HF_REPO="${MOONSHINE_HF_REPO:-moonshine-ai/moonshine-voice-assets}"
FORCE="${MOONSHINE_FETCH_FORCE:-}"

# Catalog pin for streaming fixtures (must match core/moonshine-model-catalog.cpp).
STREAMING_PIN="tiny-streaming-en/quantized_26_07_30"

usage() {
  sed -n '2,22p' "$0" | sed 's/^# \?//'
  exit "${1:-0}"
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "error: required command not found: $1" >&2
    exit 1
  }
}

# Download URL -> dest if missing, force set, or Content-Length differs.
fetch_url() {
  local url="$1"
  local dest="$2"
  mkdir -p "$(dirname "${dest}")"

  if [[ -z "${FORCE}" && -f "${dest}" ]]; then
    local remote_len=""
    remote_len="$(curl -sI -L --fail "${url}" | tr -d '\r' | awk -F': ' 'tolower($1)=="content-length"{print $2; exit}')" || true
    if [[ -n "${remote_len}" ]]; then
      local local_len
      local_len="$(wc -c <"${dest}" | tr -d ' ')"
      if [[ "${local_len}" == "${remote_len}" ]]; then
        echo "ok  ${dest} (${local_len} bytes)"
        return 0
      fi
      echo "refresh ${dest} (local ${local_len} != remote ${remote_len})"
    fi
  fi

  echo "get ${url}"
  local tmp
  tmp="$(mktemp "${dest}.tmp.XXXXXX")"
  if ! curl -fL --retry 3 --retry-delay 2 -o "${tmp}" "${url}"; then
    rm -f "${tmp}"
    echo "error: download failed: ${url}" >&2
    return 1
  fi
  mv -f "${tmp}" "${dest}"
  echo "wrote ${dest} ($(wc -c <"${dest}" | tr -d ' ') bytes)"
}

fetch_cdn() {
  local rel="$1" # path under CDN host, e.g. model/tiny-en/...
  local dest="$2"
  fetch_url "${CDN_BASE}/${rel}" "${dest}"
}

fetch_test_assets() {
  echo "=== test-assets (CDN) ==="
  local ta="${ROOT}/test-assets"

  fetch_cdn "model/tiny-en/quantized/tiny-en/encoder_model.ort" \
    "${ta}/tiny-en/encoder_model.ort"
  fetch_cdn "model/tiny-en/quantized/tiny-en/decoder_model_merged.ort" \
    "${ta}/tiny-en/decoder_model_merged.ort"
  fetch_cdn "model/tiny-en/quantized/tiny-en/decoder_with_attention.ort" \
    "${ta}/tiny-en/decoder_with_attention.ort"
  fetch_cdn "model/tiny-en/quantized/tiny-en/tokenizer.bin" \
    "${ta}/tiny-en/tokenizer.bin"

  for f in adapter.ort cross_kv.ort decoder_kv.ort decoder_kv_with_attention.ort \
           encoder.ort frontend.ort streaming_config.json tokenizer.bin; do
    fetch_cdn "model/${STREAMING_PIN}/${f}" "${ta}/tiny-streaming-en/${f}"
  done

  fetch_cdn "model/diarization-community1/segmentation.ort" \
    "${ta}/diarization/segmentation.ort"
  fetch_cdn "model/diarization-community1/embedding.ort" \
    "${ta}/diarization/embedding.ort"

  fetch_cdn "model/embeddinggemma-300m/model_q4.ort" \
    "${ta}/embeddinggemma-300m-ONNX/model_q4.ort"

  fetch_cdn "model/spelling-en/spelling_cnn.ort" \
    "${ta}/spelling_cnn.ort"
}

# androidTest historically bundled its own tiny-en; build.gradle.kts now uses
# test-assets as the androidTest asset root. Keep this target for anyone who
# still points at android/java/androidTest/assets explicitly.
fetch_android_test() {
  echo "=== android instrumented-test tiny-en (CDN) ==="
  local dest="${ROOT}/android/java/androidTest/assets/tiny-en"
  fetch_cdn "model/tiny-en/quantized/tiny-en/encoder_model.ort" \
    "${dest}/encoder_model.ort"
  fetch_cdn "model/tiny-en/quantized/tiny-en/decoder_model_merged.ort" \
    "${dest}/decoder_model_merged.ort"
  fetch_cdn "model/tiny-en/quantized/tiny-en/tokenizer.bin" \
    "${dest}/tokenizer.bin"
}

fetch_tts_via_hf() {
  need_cmd hf
  need_cmd rsync
  echo "=== TTS via Hugging Face (${HF_REPO}) ==="
  local dest="${ROOT}/core/moonshine-tts/data"
  mkdir -p "${dest}"
  # Download only the tts/ tree, then flatten into data/ (HF paths are tts/<key>).
  local staging
  staging="$(mktemp -d "${TMPDIR:-/tmp}/moonshine-tts-hf.XXXXXX")"
  if ! hf download "${HF_REPO}" --include "tts/**" --local-dir "${staging}"; then
    rm -rf "${staging}"
    return 1
  fi
  if [[ ! -d "${staging}/tts" ]]; then
    rm -rf "${staging}"
    echo "error: hf download did not produce ${staging}/tts" >&2
    return 1
  fi
  # Copy over binaries; do not clobber checked-in README.md files.
  rsync -a --exclude 'README.md' --exclude '**/README.md' \
    "${staging}/tts/" "${dest}/"
  rm -rf "${staging}"
  echo "TTS tree populated under ${dest}"
}

fetch_tts_via_cdn_inventory() {
  need_cmd curl
  need_cmd python3
  echo "=== TTS via CDN (inventory from HF FILES.tsv) ==="
  local dest="${ROOT}/core/moonshine-tts/data"
  mkdir -p "${dest}"
  local tsv
  tsv="$(mktemp)"
  if ! curl -fL -o "${tsv}" \
    "https://huggingface.co/${HF_REPO}/resolve/main/FILES.tsv"; then
    rm -f "${tsv}"
    return 1
  fi

  python3 - "${tsv}" "${CDN_BASE}" "${dest}" "${FORCE}" <<'PY'
import os, sys, urllib.request

tsv, cdn_base, dest, force = sys.argv[1:5]
force = bool(force)
count = 0
skipped = 0
with open(tsv, encoding="utf-8") as f:
    f.readline()  # header
    for line in f:
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        path = parts[0]
        if not path.startswith("tts/"):
            continue
        rel = path[len("tts/") :]
        size = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
        out = os.path.join(dest, rel)
        if not force and os.path.isfile(out) and size is not None:
            if os.path.getsize(out) == size:
                skipped += 1
                continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        url = f"{cdn_base}/{path}"
        print(f"get {url}", flush=True)
        tmp = out + ".tmp"
        try:
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, out)
        except Exception as e:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise SystemExit(f"download failed for {url}: {e}") from e
        count += 1
print(f"downloaded {count} TTS files ({skipped} already present)", flush=True)
PY
  local status=$?
  rm -f "${tsv}"
  return "${status}"
}

fetch_tts() {
  if command -v hf >/dev/null 2>&1; then
    fetch_tts_via_hf
  else
    echo "note: 'hf' CLI not found; falling back to per-file CDN downloads" >&2
    fetch_tts_via_cdn_inventory
  fi
}

TARGET="${1:-all}"
case "${TARGET}" in
  -h|--help) usage 0 ;;
  test-assets) need_cmd curl; fetch_test_assets ;;
  tts) fetch_tts ;;
  android-test) need_cmd curl; fetch_android_test ;;
  all)
    need_cmd curl
    fetch_test_assets
    fetch_tts
    ;;
  *)
    echo "unknown target: ${TARGET}" >&2
    usage 1
    ;;
esac

echo "Done."
