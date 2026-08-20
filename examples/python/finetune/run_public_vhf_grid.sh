#!/usr/bin/env bash
# Measure the public trainer on published Medium. Quote only these numbers.
# ATCOSIM sanity, then UWB decoder / both / full, each with ATCO2 + LibriSpeech.
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}
PYTHON=${PYTHON:-python3}
MODEL=${MODEL:-moonshine-ai/moonshine-streaming-medium}
OUT=${OUT:-./public_vhf_grid}
HOURS=${HOURS:-2.0}

run() {
  local tag="$1"; shift
  echo "=== $tag ==="
  "$PYTHON" -m moonshine_voice.lora \
    --model "$MODEL" \
    --train-hours "$HOURS" \
    --eval --eval-dataset atco2 --canary \
    --output-dir "$OUT/$tag" \
    --work-dir "$OUT/work" \
    "$@"
}

mkdir -p "$OUT"
run atcosim_dec --dataset atcosim --sites decoder
run uwb_dec --dataset uwb_atcc --sites decoder
run uwb_both --dataset uwb_atcc --sites both
run uwb_full --dataset uwb_atcc --adapt full

echo "=== summaries ==="
for d in atcosim_dec uwb_dec uwb_both uwb_full; do
  echo "-- $d"
  "$PYTHON" -c "import json; print(json.dumps(json.load(open('$OUT/$d/summary.json')), indent=2))"
done
