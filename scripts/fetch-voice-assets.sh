#!/usr/bin/env bash
# Bootstrap model/TTS binaries that used to live in Git LFS.
#
# Runtime SDKs still download from https://download.moonshine.ai. This script
# fills local trees used by offline tests, examples, and CDN publish:
#   test-assets/…              STT / diarization / embedding fixtures
#                              (tiny streaming for every published language)
#   core/moonshine-tts/data/…  TTS + G2P bundles (READMEs stay in git)
#   language-bindings/android/java/androidTest/assets/tiny-en/   (optional; mirrors CDN tiny-en)
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
#   MOONSHINE_HF_REVISION
#       Hugging Face revision for the TTS tree (branch, tag, or sha). When
#       unset, uses v<MOONSHINE_VERSION> if that tag exists on the Hub, else
#       `main` (the open candidate, before the release has tagged a snapshot).
#   MOONSHINE_FETCH_FORCE  if non-empty, re-download even when size matches
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CDN_BASE="${MOONSHINE_CDN_BASE:-https://download.moonshine.ai}"
HF_REPO="${MOONSHINE_HF_REPO:-moonshine-ai/moonshine-voice-assets}"
FORCE="${MOONSHINE_FETCH_FORCE:-}"
HF_REVISION=""

# Catalog pins for tiny streaming fixtures (must match
# core/moonshine-model-catalog.cpp). Tiny is fetched for every published
# streaming language so streaming-language-smoke-test can load each one
# offline. English also ships the attention decoder used by word-timestamp tests.
STREAMING_TINY_PINS=(
  tiny-streaming-ar/quantized_26_08_24
  tiny-streaming-de/quantized_26_08_24
  tiny-streaming-en/quantized_26_08_21
  tiny-streaming-es/quantized_26_08_24
  tiny-streaming-ja/quantized_26_08_23
  tiny-streaming-tl/quantized_26_08_24
  tiny-streaming-vi/quantized_26_08_24
  tiny-streaming-zh/quantized_26_08_24
)

usage() {
  sed -n '2,28p' "$0" | sed 's/^# \?//'
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

# Tiny streaming graphs shared by every language. English also has the
# attention decoder; other languages 404 if we ask for it.
fetch_tiny_streaming_model() {
  local ta="$1"
  local pin="$2"
  local dest="${ta}/${pin%%/*}"
  local f
  for f in adapter.ort cross_kv.ort decoder_kv.ort encoder.ort \
           frontend.model.ort frontend.weights.ort \
           streaming_config.json tokenizer.bin; do
    fetch_cdn "model/${pin}/${f}" "${dest}/${f}"
  done
  if [[ "${pin}" == tiny-streaming-en/* ]]; then
    fetch_cdn "model/${pin}/decoder_kv_with_attention.ort" \
      "${dest}/decoder_kv_with_attention.ort"
  fi
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

  local pin
  for pin in "${STREAMING_TINY_PINS[@]}"; do
    fetch_tiny_streaming_model "${ta}" "${pin}"
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

# androidTest historically bundled its own tiny-en; language-bindings/android/build.gradle.kts now uses
# test-assets as the androidTest asset root. Keep this target for anyone who
# still points at language-bindings/android/java/androidTest/assets explicitly.
fetch_android_test() {
  echo "=== android instrumented-test tiny-en (CDN) ==="
  local dest="${ROOT}/language-bindings/android/java/androidTest/assets/tiny-en"
  fetch_cdn "model/tiny-en/quantized/tiny-en/encoder_model.ort" \
    "${dest}/encoder_model.ort"
  fetch_cdn "model/tiny-en/quantized/tiny-en/decoder_model_merged.ort" \
    "${dest}/decoder_model_merged.ort"
  fetch_cdn "model/tiny-en/quantized/tiny-en/tokenizer.bin" \
    "${dest}/tokenizer.bin"
}

library_version() {
  sed -n 's/^set(MOONSHINE_VERSION "\([^"]*\)").*/\1/p' \
    "${ROOT}/core/CMakeLists.txt" | head -n1
}

# True if FILES.tsv exists at this Hub revision (tag, branch, or sha).
hf_revision_exists() {
  local rev="$1"
  local code
  code="$(curl -sI -o /dev/null -w "%{http_code}" \
    "https://huggingface.co/${HF_REPO}/resolve/${rev}/FILES.tsv" || true)"
  case "${code}" in
    200|301|302|303|307|308) return 0 ;;
    *) return 1 ;;
  esac
}

resolve_hf_revision() {
  if [[ -n "${MOONSHINE_HF_REVISION:-}" ]]; then
    printf '%s\n' "${MOONSHINE_HF_REVISION}"
    return
  fi
  local version tag
  version="$(library_version)"
  if [[ -n "${version}" ]]; then
    tag="v${version}"
    if hf_revision_exists "${tag}"; then
      printf '%s\n' "${tag}"
      return
    fi
  fi
  printf 'main\n'
}

ensure_hf_revision() {
  if [[ -z "${HF_REVISION}" ]]; then
    HF_REVISION="$(resolve_hf_revision)"
  fi
}

fetch_tts_via_hf() {
  need_cmd hf
  need_cmd rsync
  ensure_hf_revision
  echo "=== TTS via Hugging Face (${HF_REPO} @ ${HF_REVISION}) ==="
  local dest="${ROOT}/core/moonshine-tts/data"
  mkdir -p "${dest}"
  # Download only the tts/ tree, then flatten into data/ (HF paths are tts/<key>).
  local staging
  staging="$(mktemp -d "${TMPDIR:-/tmp}/moonshine-tts-hf.XXXXXX")"
  if ! hf download "${HF_REPO}" --revision "${HF_REVISION}" \
    --include "tts/**" --local-dir "${staging}"; then
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

# Percent-encode one path segment as UTF-8 bytes (works without Python).
urlencode_segment() {
  local s="$1"
  local i c o out=""
  local LC_ALL=C
  for ((i = 0; i < ${#s}; i++)); do
    c="${s:i:1}"
    case "${c}" in
      [A-Za-z0-9._~-]) out+="${c}" ;;
      *)
        # Mask to 8 bits: bash's "'$c" is signed for high UTF-8 bytes.
        printf -v o '%d' "'${c}"
        printf -v c '%%%02X' $((o & 0xFF))
        out+="${c}"
        ;;
    esac
  done
  printf '%s' "${out}"
}

# Encode each /-separated segment of a CDN-relative path.
urlencode_cdn_path() {
  local path="$1"
  local out="" seg rest="${path}/"
  while [[ -n "${rest}" ]]; do
    seg="${rest%%/*}"
    rest="${rest#*/}"
    [[ -n "${out}" ]] && out+="/"
    out+="$(urlencode_segment "${seg}")"
  done
  printf '%s' "${out}"
}

fetch_tts_via_cdn_inventory() {
  need_cmd curl
  ensure_hf_revision
  echo "=== TTS via CDN (inventory from HF FILES.tsv @ ${HF_REVISION}) ==="
  local dest="${ROOT}/core/moonshine-tts/data"
  mkdir -p "${dest}"
  local tsv
  tsv="$(mktemp)"
  if ! curl -fL -o "${tsv}" \
    "https://huggingface.co/${HF_REPO}/resolve/${HF_REVISION}/FILES.tsv"; then
    rm -f "${tsv}"
    return 1
  fi

  # Cloudflare WAF rejects Python-urllib's default User-Agent (HTTP 403) on
  # download.moonshine.ai, so fetch with curl. Enumerate FILES.tsv in bash so
  # this also works on Windows Git Bash (no working python3 on PATH).
  local path size rel dest_path enc_path url tmp
  local count=0 skipped=0 downloaded=0
  # Skip header line.
  {
    read -r _
    while IFS=$'\t' read -r path size _ || [[ -n "${path}" ]]; do
      [[ -z "${path}" || "${path}" == \#* ]] && continue
      [[ "${path}" == tts/* ]] || continue
      rel="${path#tts/}"
      dest_path="${dest}/${rel}"
      if [[ -z "${FORCE}" && -f "${dest_path}" && "${size}" =~ ^[0-9]+$ ]]; then
        if [[ "$(wc -c <"${dest_path}" | tr -d ' ')" == "${size}" ]]; then
          skipped=$((skipped + 1))
          continue
        fi
      fi
      enc_path="$(urlencode_cdn_path "${path}")"
      url="${CDN_BASE}/${enc_path}"
      echo "get ${url}"
      mkdir -p "$(dirname "${dest_path}")"
      tmp="${dest_path}.tmp"
      if ! curl -fL --retry 3 --retry-delay 2 -o "${tmp}" "${url}"; then
        rm -f "${tmp}" "${tsv}"
        echo "error: download failed: ${url}" >&2
        return 1
      fi
      mv -f "${tmp}" "${dest_path}"
      downloaded=$((downloaded + 1))
      count=$((count + 1))
    done
  } <"${tsv}"
  rm -f "${tsv}"
  echo "downloaded ${downloaded} TTS files (${skipped} already present)"
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
