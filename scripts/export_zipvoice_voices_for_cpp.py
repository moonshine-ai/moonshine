#!/usr/bin/env python3
# Copyright 2026 Useful Sensors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate the compiled-in ZipVoice reference voices C++ source.

Reads the VCTK ``voice_bank_vctk`` bank produced by the ZipVoice repo, selects one masculine and one
feminine speaker per accent (where available), trims/pads each clip to a fixed 4-second, 24 kHz, mono
16-bit PCM buffer, and writes ``core/moonshine-tts/src/zipvoice-voices-data.cpp`` with the PCM arrays,
transcripts and metadata table consumed by ``zipvoice-voices.h``.

Usage:

  python3 scripts/export_zipvoice_voices_for_cpp.py \
      --voice-bank ~/projects/ZipVoice/voice_bank_vctk \
      --out core/moonshine-tts/src/zipvoice-voices-data.cpp
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 24000
CLIP_SECONDS = 4
CLIP_SAMPLES = SAMPLE_RATE * CLIP_SECONDS

# Accents that only ship a single gender in VCTK are kept as-is; "Unknown" is skipped.
SKIP_ACCENTS = {"Unknown"}

# Built-in voice slugs (accent_gender) to exclude, e.g. low-quality renders removed after review.
EXCLUDE_SLUGS = {
    "welsh_female",
    "scottish_male",
    "scottish_female",
    "northern_irish_male",
    "british_female",
}


def slug_from_voice_id(voice_id: str) -> str:
    """``voice_000_american_female`` -> ``american_female``."""
    return re.sub(r"^voice_\d+_", "", voice_id)


def select_voices(voices):
    """One masculine + one feminine speaker per accent (first of each in file order)."""
    chosen = []
    seen = set()  # (accent, gender)
    for v in voices:
        accent = v["accent"]
        if accent in SKIP_ACCENTS:
            continue
        gender = "female" if v["gender_presentation"].startswith("f") else "male"
        key = (accent, gender)
        if key in seen:
            continue
        seen.add(key)
        chosen.append(v)
    # Drop excluded slugs *after* selection so an excluded voice is removed entirely rather than
    # replaced by another speaker of the same accent/gender.
    chosen = [v for v in chosen if slug_from_voice_id(v["voice_id"]) not in EXCLUDE_SLUGS]
    chosen.sort(key=lambda v: slug_from_voice_id(v["voice_id"]))
    return chosen


def _clean_cut_index(wav: np.ndarray, sr: int, max_samples: int) -> int:
    """End index (<= max_samples) that ends at a *strong* natural pause when possible.

    A hard cut at exactly 4 s often lands mid-word, and ZipVoice then "continues" the reference clip
    at the start of the generated speech (audible leakage). Even cutting at a weak inter-word gap can
    leave the clip ending on a dangling function word (e.g. "...slabs of"), which also leaks. So we
    prefer the last *phrase-level* pause (comma / sentence boundary, >= ~220 ms of silence), falling
    back to any short pause, then to a hard cut with a fade-out.
    """
    if len(wav) <= max_samples:
        return len(wav)
    seg = np.abs(wav[:max_samples]).astype(np.float32)
    peak = float(seg.max()) or 1.0
    win = max(1, int(0.02 * sr))  # 20 ms smoothing envelope
    env = np.convolve(seg, np.ones(win, dtype=np.float32) / win, mode="same")
    thresh = 0.04 * peak
    silent = env < thresh
    min_keep = int(1.0 * sr)   # keep at least ~1 s of reference speech
    strong_sil = int(0.22 * sr)  # phrase/sentence boundary (comma, full stop)
    weak_sil = int(0.10 * sr)    # any inter-word gap

    # Collect (start, length) of every silence run whose start is past ``min_keep``.
    runs = []
    i = 0
    n = len(silent)
    while i < n:
        if silent[i]:
            j = i
            while j < n and silent[j]:
                j += 1
            if i >= min_keep:
                runs.append((i, j - i))
            i = j
        else:
            i += 1

    # Prefer the last strong (phrase-level) pause; otherwise the last weak pause.
    strong = [start for start, length in runs if length >= strong_sil]
    if strong:
        return min(max_samples, strong[-1] + int(0.05 * sr))
    weak = [start for start, length in runs if length >= weak_sil]
    if weak:
        return min(max_samples, weak[-1] + int(0.05 * sr))
    return max_samples


def load_clip_pcm16(path: Path) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = data.mean(axis=1).astype(np.float32)  # to mono
    if sr != SAMPLE_RATE:
        raise SystemExit(f"{path} is {sr} Hz; expected {SAMPLE_RATE} Hz")
    cut = _clean_cut_index(wav, sr, CLIP_SAMPLES)
    wav = np.array(wav[:cut], dtype=np.float32)
    # Short fade-out softens any residual mid-word cut so it doesn't leak into generation.
    fade = min(len(wav), int(0.03 * sr))
    if fade > 0:
        wav[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    if len(wav) < CLIP_SAMPLES:
        wav = np.pad(wav, (0, CLIP_SAMPLES - len(wav)))
    else:
        wav = wav[:CLIP_SAMPLES]
    pcm = np.clip(np.rint(wav * 32768.0), -32768, 32767).astype(np.int16)
    return pcm


class _CloneTranscriber:
    """Transcribes the *exact* embedded clip with Moonshine ASR.

    ZipVoice needs the clone transcript to match the reference audio; since clips are trimmed to a fixed
    4 seconds, the original VCTK sentence no longer corresponds to what is actually spoken. Transcribing
    the trimmed clip (like ZipVoice's own auto-transcription) keeps them aligned and avoids reference-clip leakage.
    """

    def __init__(self):
        from moonshine_voice import (  # noqa: F401
            ModelArch,
            Transcriber,
            get_model_for_language,
        )

        self._ModelArch = ModelArch
        model_path, model_arch = get_model_for_language("en", wanted_model_arch=ModelArch.BASE)
        self._t = Transcriber(model_path=model_path, model_arch=model_arch)

    def transcribe_pcm16(self, pcm: np.ndarray) -> str:
        audio = pcm.astype(np.float32) / 32768.0
        tr = self._t.transcribe_without_streaming(audio, sample_rate=SAMPLE_RATE)
        return " ".join(line.text.strip() for line in tr.lines).strip()

    def close(self):
        try:
            self._t.close()
        except Exception:
            pass


def c_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice-bank", required=True, help="Path to voice_bank_vctk directory")
    parser.add_argument("--out", required=True, help="Output .cpp path")
    parser.add_argument(
        "--transcribe",
        type=lambda s: s.lower() not in ("0", "false", "no"),
        default=True,
        help="Transcribe each trimmed clip with Moonshine ASR for an audio-matched clone transcript "
        "(default true). When false, uses the original VCTK sentence (may not match a trimmed clip).",
    )
    args = parser.parse_args()

    bank = Path(args.voice_bank).expanduser()
    with open(bank / "voices.json", "r", encoding="utf-8") as f:
        voices = json.load(f)

    chosen = select_voices(voices)

    transcriber = None
    if args.transcribe:
        transcriber = _CloneTranscriber()

    lines = []
    lines.append("// GENERATED by scripts/export_zipvoice_voices_for_cpp.py -- do not edit by hand.")
    lines.append("// Built-in ZipVoice reference voices: one masculine + one feminine VCTK speaker per")
    lines.append("// accent, each a 4-second, 24 kHz, mono 16-bit PCM clip plus its exact transcript.")
    lines.append('#include "zipvoice-voices.h"')
    lines.append("")
    lines.append("namespace moonshine_tts {")
    lines.append("namespace {")
    lines.append("")

    entries = []
    for v in chosen:
        slug = slug_from_voice_id(v["voice_id"])
        sym = "kPcm_" + re.sub(r"[^A-Za-z0-9_]", "_", slug)
        pcm = load_clip_pcm16(bank / f"clips/{Path(v['clip']).name}")
        gender = "female" if v["gender_presentation"].startswith("f") else "male"
        # Transcript must match the *trimmed* audio, not the original VCTK sentence.
        clone_transcript = v["prompt_text"]  # voices.json key predates the clone_* naming
        if transcriber is not None:
            asr = transcriber.transcribe_pcm16(pcm)
            if asr:
                clone_transcript = asr
        lines.append(f"const int16_t {sym}[] = {{")
        row = []
        for i, s in enumerate(pcm.tolist()):
            row.append(str(s))
            if len(row) == 20:
                lines.append("  " + ",".join(row) + ",")
                row = []
        if row:
            lines.append("  " + ",".join(row) + ",")
        lines.append("};")
        lines.append("")
        entries.append((slug, v, gender, sym, len(pcm), clone_transcript))

    lines.append("const ZipVoiceBuiltinVoice kZipVoiceBuiltinVoices[] = {")
    for slug, v, gender, sym, n, clone_transcript in entries:
        lines.append("  {")
        lines.append(f'    "{c_escape(slug)}",')
        lines.append(f'    "{c_escape(v["accent"])}",')
        lines.append(f'    "{gender}",')
        lines.append(f'    "{c_escape(v.get("vctk_speaker", ""))}",')
        lines.append(f'    "{c_escape(clone_transcript)}",')
        lines.append(f"    {sym},")
        lines.append(f"    {n}u,")
        lines.append(f"    {SAMPLE_RATE}u,")
        lines.append("  },")
    lines.append("};")
    lines.append("")
    lines.append("}  // namespace")
    lines.append("")
    lines.append("const ZipVoiceBuiltinVoice* zipvoice_builtin_voices(size_t* count) {")
    lines.append("  if (count != nullptr) {")
    lines.append("    *count = sizeof(kZipVoiceBuiltinVoices) / sizeof(kZipVoiceBuiltinVoices[0]);")
    lines.append("  }")
    lines.append("  return kZipVoiceBuiltinVoices;")
    lines.append("}")
    lines.append("")
    lines.append("}  // namespace moonshine_tts")
    lines.append("")

    if transcriber is not None:
        transcriber.close()

    out_path = Path(args.out)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(entries)} voices to {out_path}")
    for slug, v, gender, _sym, n, clone_transcript in entries:
        print(f"  {slug:<24} {v['accent']:<14} {gender:<7} {n} samples  \"{clone_transcript[:48]}\"")


if __name__ == "__main__":
    main()
