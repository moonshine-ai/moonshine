#!/usr/bin/env bash
# Mirror one published model directory to the Hugging Face archive repo.
#
# moonshine-ai/moonshine-voice-assets holds every file the download manifests
# reference, at the same path it has under https://download.moonshine.ai, and
# its FILES.tsv records the size and checksums of each. The runtime never reads
# from it; it exists so a given release's exact bytes can be found and verified
# after the CDN has moved on. Run this after scripts/upload-streaming-model.sh
# (or the equivalent for another model type) has published to the CDN and the
# catalog has been pointed at the new directory, with the same local directory:
#
#   scripts/mirror-model-to-hf.sh ~/staged/tiny-streaming-ja tiny-streaming-ja quantized_26_11_07
#
# The files go up in one commit and FILES.tsv is updated in a second one, so a
# failed upload leaves the inventory describing only what is really there.
# Re-running is safe: files already present with identical content are skipped
# by the Hub, and FILES.tsv rows are replaced rather than duplicated.
#
# Requires the `hf` CLI, authenticated to an account that can write the
# moonshine-ai org (hf auth login), and the google-crc32c Python package.
#
# Environment:
#   MOONSHINE_HF_REPO   default moonshine-ai/moonshine-voice-assets
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <source-dir> <model-name> <quantized-dir>" >&2
  exit 2
fi

SRC="$(cd "$1" && pwd)"
MODEL_NAME="$2"
QUANTIZED_DIR="$3"
HF_REPO="${MOONSHINE_HF_REPO:-moonshine-ai/moonshine-voice-assets}"
REMOTE_DIR="model/${MODEL_NAME}/${QUANTIZED_DIR}"

for cmd in hf python3; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "error: $cmd not found" >&2; exit 1; }
done
whoami="$(hf auth whoami 2>/dev/null || true)"
if [[ -z "${whoami}" ]]; then
  echo "error: hf is not authenticated; run: hf auth login" >&2
  exit 1
fi

# Only the files a client can download belong in the mirror, so stray .onnx
# sources or tool output next to them are left behind.
INCLUDE=(--include "*.ort" --include "*.json" --include "*.bin")

echo "Mirroring ${SRC} -> ${HF_REPO}:${REMOTE_DIR}" >&2
hf upload "${HF_REPO}" "${SRC}" "${REMOTE_DIR}" "${INCLUDE[@]}" \
  --commit-message "Mirror ${REMOTE_DIR}" >&2

# FILES.tsv: path, size, md5, base64 big-endian crc32c (the GCS convention the
# inventory started with). Rows for this directory replace any existing ones.
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
hf download "${HF_REPO}" FILES.tsv --local-dir "${WORK}" >/dev/null 2>&1 \
  || printf 'path\tsize_bytes\tmd5\tcrc32c\n' > "${WORK}/FILES.tsv"

python3 - "${SRC}" "${REMOTE_DIR}" "${WORK}/FILES.tsv" <<'EOF'
import base64, hashlib, os, sys
import google_crc32c

src, remote_dir, tsv = sys.argv[1:4]
rows = []
with open(tsv) as f:
    header = f.readline()
    for line in f:
        line = line.rstrip("\n")
        if line and not line.startswith(remote_dir + "/"):
            rows.append(line)
for name in sorted(os.listdir(src)):
    if not name.endswith((".ort", ".json", ".bin")):
        continue
    path = os.path.join(src, name)
    md5 = hashlib.md5()
    crc = google_crc32c.Checksum()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            md5.update(chunk)
            crc.update(chunk)
    rows.append("\t".join([f"{remote_dir}/{name}", str(os.path.getsize(path)),
                           md5.hexdigest(), base64.b64encode(crc.digest()).decode()]))
rows.sort()
with open(tsv, "w") as f:
    f.write(header if header.strip() else "path\tsize_bytes\tmd5\tcrc32c\n")
    f.write("\n".join(rows) + "\n")
print(f"FILES.tsv: {len(rows)} rows", file=sys.stderr)
EOF

hf upload "${HF_REPO}" "${WORK}/FILES.tsv" FILES.tsv \
  --commit-message "Add ${REMOTE_DIR} to FILES.tsv" >&2
echo "Mirrored https://huggingface.co/${HF_REPO}/tree/main/${REMOTE_DIR}" >&2
