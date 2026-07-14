#!/usr/bin/env bash
# End-to-end pipeline: synthesize -> mine -> extract -> train -> evaluate -> export.
#
# Edit words.txt first, then run:  ./run_all.sh
# Almost every knob lives in config.sh (or override inline, e.g. PS_ROWS=50000 ./run_all.sh).
set -euo pipefail
cd "$(dirname "$0")"
. ./config.sh

echo "=================================================================="
echo " STT training pipeline"
echo "   words:      $WORDS"
echo "   data dir:   $DATA"
echo "=================================================================="

echo; echo "### 1/6  Synthesize command words with ZipVoice"
$PY tools/synthesize.py --words-file "$WORDS" --output-dir "$TTS_DIR" \
    ${TTS_VOICES:+--voices $TTS_VOICES}

echo; echo "### (optional) Download MUSAN noise + RIRs for augmentation"
if [ ! -d "$MUSAN_DIR" ] || [ ! -d "$RIR_DIR" ]; then
    $PY tools/download_musan_rirs.py --musan-dir "$MUSAN_DIR" --rir-dir "$RIR_DIR" \
        || echo "  (augmentation asset download failed; continuing with synthetic noise)"
fi

echo; echo "### 2/6  Mine People's Speech for command words"
$PY tools/mine_peoples_speech.py --words-file "$WORDS" --mined-dir "$MINED_DIR" \
    --config "$PS_CONFIG" --split "$PS_SPLIT" --limit "$PS_ROWS"

echo; echo "### 3/6  Mine People's Speech for _unknown_ (reject) clips"
$PY tools/mine_peoples_speech.py --words-file "$WORDS" --mined-dir "$MINED_DIR" \
    --config "$PS_CONFIG" --split "$PS_SPLIT" --limit "$PS_UNKNOWN_ROWS" --unknown

echo; echo "### 4/6  Force-align and cut clips"
$PY tools/extract_clips.py --words-file "$WORDS" --mined-dir "$MINED_DIR" \
    --output-dir "$PS_DIR"

echo; echo "### 5/6  Train"
$PY -m stt_training.train --words-file "$WORDS" \
    --tts-dir "$TTS_DIR" --ps-dir "$PS_DIR" \
    --checkpoints-dir "$CKPT_DIR" --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" \
    --musan-noise-dir "$MUSAN_DIR" --rir-dir "$RIR_DIR"

RUN_DIR=$(ls -dt "$CKPT_DIR"/run_* | head -1)
echo "   latest run: $RUN_DIR"

echo; echo "### 6/6  Export to int8 LiteRT and evaluate"
$PY -m stt_training.export --checkpoint "$RUN_DIR" \
    --tts-dir "$TTS_DIR" --ps-dir "$PS_DIR"
$PY -m stt_training.evaluate --checkpoint "$RUN_DIR" \
    --tts-dir "$TTS_DIR" --ps-dir "$PS_DIR" \
    --tflite "$RUN_DIR/spelling_cnn_mel_int8.tflite"

echo
echo "=================================================================="
echo " Done. Deploy artifacts:"
echo "   $RUN_DIR/spelling_cnn_mel_int8.tflite"
echo "   $RUN_DIR/spelling_cnn_meta.json"
echo " Next: copy both into moonshine-micro/models/ and see README step 8."
echo "=================================================================="
