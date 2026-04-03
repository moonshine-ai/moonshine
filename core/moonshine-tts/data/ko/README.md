# Korean — `ko`

## Contents

- **`dict.tsv`** — Hangul word → IPA (broad Seoul-style inventory after normalization), used by C++ `KoreanRuleG2p` (lexicon + internal Hangul rule pipeline).

- **`piper-voices/ko_KR-melotts-medium.onnx`** (+ **`.onnx.json`**) — Piper VITS (22.05 kHz, medium) trained on MeloTTS-style synthetic Korean. Used by C++ `PiperTTS` (`--lang ko`), `moonshine_tts` (Piper fallback), and `speak.py --engine piper --lang ko`. Repo root symlinks `data/ko/piper-voices/ko_KR-melotts-medium.onnx*` point here.

Weights may be **int8-packed** with `onnx-shrink-ray` (see below); the JSON sidecar is unchanged.

### MeloTTS

**[MeloTTS](https://github.com/myshell-ai/MeloTTS)** (MIT) is a multi-lingual text-to-speech library from [MyShell](https://myshell.ai). Korean is among its supported locales. It works well as a **fast, open-source reference synthesizer** for bootstrapping synthetic speech or style targets when building Piper-style datasets.

The file `ko_KR-melotts-medium.onnx` is **not** the MeloTTS model: it is a **Piper VITS** checkpoint trained (or continued) on MeloTTS-style Korean material and exported for `piper_train` / ONNX Runtime. The `.onnx.json` sidecar holds Piper metadata (`phoneme_id_map`, sample rate, inference scales, etc.).

### Re-export ONNX from a MeloTTS training checkpoint

Use `training/piper_korean/export_melotts_checkpoint_to_cpp.sh` (writes `data/ko/piper-voices/ko_KR-melotts-medium.onnx` + JSON and refreshes `data/ko/piper-voices` symlinks).

### Optional: int8 weight packing (`onnx-shrink-ray`)

After export (or to re-pack an FP32 checkout):

```bash
pip install onnx onnx-shrink-ray onnx-graphsurgeon
# From the parent monorepo root (this tree is the `moonshine-tts` submodule):
python scripts/shrink_piper_voice_onnx_weights.py --root moonshine-tts/data --name-contains melotts
```

## Provenance

| Asset | Source |
|--------|--------|
| `dict.tsv` | [open-dict-data/ipa-dict](https://github.com/open-dict-data/ipa-dict) — `data/ko.txt` (MIT). Fetched via `scripts/download_multilingual_ipa_lexicons.py`. |

**Note:** A separate Korean **morph + UPOS** ONNX export exists for tagging tests (`data/ko/roberta_korean_morph_upos_onnx/`, HF [KoichiYasuoka/roberta-base-korean-morph-upos](https://huggingface.co/KoichiYasuoka/roberta-base-korean-morph-upos)), produced by `scripts/export_korean_ud_onnx.py`. It is **not** wired into the main Korean G2P path in C++ or Python `korean_rule_g2p`—only this lexicon is required for pronunciation in the shipped pipeline.

## Recreating

```bash
python scripts/download_multilingual_ipa_lexicons.py --only ko
```

Writes `data/ko/dict.tsv` and `data/ko/source.txt`. Copy `dict.tsv` into `data/ko/` if you maintain this tree separately.

### Optional: Korean morph ONNX (for `KoreanTokPosOnnx` / tests only)

```bash
python scripts/export_korean_ud_onnx.py
```

Default: `data/ko/roberta_korean_morph_upos_onnx/`.
