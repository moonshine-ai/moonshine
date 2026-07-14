# Shared configuration for the STT training pipeline.
# Sourced by run_all.sh and safe to source in your own shell:
#     . ./config.sh
#
# Almost nothing here needs changing for a normal run -- edit words.txt instead.

# Python interpreter (use your venv).
PY="${PY:-python}"

# The vocabulary. This is the one file you normally edit.
WORDS="${WORDS:-words.txt}"

# Where generated audio, checkpoints and caches live (gitignored).
DATA="${DATA:-data}"
TTS_DIR="${TTS_DIR:-$DATA/tts}"                    # ZipVoice output, <voice>/<word>/*.wav
PS_DIR="${PS_DIR:-$DATA/peoples_speech}"           # extracted PS clips, <word>/*.wav
MINED_DIR="${MINED_DIR:-$DATA/mined}"              # mining JSONL + audio cache
CKPT_DIR="${CKPT_DIR:-checkpoints}"

# Optional augmentation assets (downloaded by tools/download_musan_rirs.py).
MUSAN_DIR="${MUSAN_DIR:-$DATA/musan/noise}"
RIR_DIR="${RIR_DIR:-$DATA/rirs}"

# --- data-gathering knobs --------------------------------------------------
# ZipVoice speakers to synthesize with (empty = all 15 built-in voices).
TTS_VOICES="${TTS_VOICES:-}"

# People's Speech: how many rows to scan (train split is ~1.5M). Start small.
PS_CONFIG="${PS_CONFIG:-clean}"
PS_SPLIT="${PS_SPLIT:-train}"
PS_ROWS="${PS_ROWS:-200000}"          # rows to scan for command words
PS_UNKNOWN_ROWS="${PS_UNKNOWN_ROWS:-20000}"   # rows to scan for _unknown_ negatives

# --- training knobs --------------------------------------------------------
EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
