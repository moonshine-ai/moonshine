# ZipVoice TTS assets

ZipVoice is a zero-shot voice-cloning text-to-speech model (k2-fsa/ZipVoice). Moonshine ships an
ONNX Runtime port as an optional TTS engine selected with a `zipvoice_` voice prefix (see
`core/moonshine-c-api.h`).

## Files in this directory

Runtime assets, resolved under `g2p_root` (or supplied in memory via the C API). `.ort` is preferred
over `.onnx` when both are present.

| Key                          | Role                                                   |
|------------------------------|--------------------------------------------------------|
| `text_encoder.onnx` / `.ort` | Embeds phoneme tokens, computes duration/upsampling.   |
| `fm_decoder.onnx` / `.ort`   | Flow-matching velocity decoder (Euler ODE).            |
| `vocoder.onnx` / `.ort`      | Vocos mel -> waveform.                                  |
| `tokens.txt`                 | Phoneme/pinyin -> id lexicon (espeak IPA + pinyin).    |
| `model.json`                 | Feature/architecture config (feat_dim, sample rate).   |

These weights are **not** committed to git in bulk; regenerate them from the upstream checkpoint (they
are tracked with Git LFS when present). The Moonshine binary release tarballs do not contain them —
they are downloaded/cached at runtime like the Kokoro/Piper assets.

## Regenerating

Requires a local checkout of the ZipVoice repo with its dependencies installed:

```bash
python3 scripts/export_zipvoice_model.py \
    --zipvoice-repo ~/projects/ZipVoice \
    --model-name zipvoice_distill
```

This runs the ZipVoice tooling (`scripts/export_onnx.py`, `scripts/rewrite_swoosh.py`,
`scripts/onnx_vocoder.py`, `scripts/check_onnx_parity.py`) and converts the result to `.ort`.

### Size vs quality

By default the **smallest** parity-passing set is shipped: mixed int8 acoustic models (per-channel
int8 weights, with activation-limited output projections kept in fp32) plus the fp32 vocoder. The
fm_decoder drops from ~455 MB (fp32) to ~119 MB (mixed int8) with no audible quality loss.

Pass `--swoosh` to instead ship the fp32 fm_decoder rewritten to use the `ai.zipvoice` custom ops
(`SwooshL`/`SwooshR`/`GluGate`/`DepthwiseConv1d`/`BiasNorm`/`Bypass`). Those ops are compiled directly
into `libmoonshine` (see `core/moonshine-tts/src/zipvoice-custom-ops.cpp`) — no shared library is
loaded at runtime — so the custom-op graph runs everywhere, including iOS. The int8 graph does not
reference the custom ops; they remain available for the fp32 path.

## Distill vs full model

`zipvoice_distill` (default) uses 8 sampling steps / guidance 3.0; the full `zipvoice` model uses 16
steps / guidance 1.0. Inference is otherwise identical (the classifier-free-guidance logic is baked
into the exported graph), so both deploy through the same files here; select with the
`zipvoice_model` option.

## Reference voices

The built-in reference voices (`zipvoice_american_female`, `zipvoice_indian_male`, …) are VCTK
clips compiled directly into `libmoonshine` (`core/moonshine-tts/src/zipvoice-voices-data.cpp`),
regenerated with `scripts/export_zipvoice_voices_for_cpp.py`. Callers can also pass their own
reference clip as in-memory PCM (`zipvoice/prompt_audio`).
