# Phonon TTS assets

[Phonon](https://gradium.ai/phonon) is [Gradium](https://gradium.ai)'s family of small streaming
text-to-speech models, built on Kyutai's [Pocket TTS](https://github.com/kyutai-labs/pocket-tts)
and Mimi codec. Moonshine ships the 31M-parameter English model,
[`gradium/phonon-31m-en`](https://huggingface.co/gradium/phonon-31m-en), as a C++ port of the
generation loop in Gradium's [`xn-ptts`](https://github.com/gradium-ai/xn-ptts) runtime, on ONNX
Runtime, selected with a `phonon_` voice prefix (see `core/moonshine-c-api.h`). The engine lives in
`core/moonshine-tts/src/phonon-tts.cpp`; the text preparation ported from `xn-ptts` is in
`phonon-text.cpp`.

## Provenance and licenses

| Asset | Source | License |
|-------|--------|---------|
| Flow LM weights, config, tokenizer (`embed_*.ort`, `backbone.ort`, `flow_head.ort`, `tokenizer.bin`) | [gradium/phonon-31m-en](https://huggingface.co/gradium/phonon-31m-en) | Apache 2.0 |
| Mimi decoder (`mimi_transformer.ort`, `mimi_seanet.ort`) | [gradium/phonon-mimi](https://huggingface.co/gradium/phonon-mimi), Gradium's build of Kyutai's Mimi codec decoder | Apache 2.0 |
| Voice catalog (`voices.json`, `voices/*.safetensors`) | [gradium/phonon-voices](https://huggingface.co/gradium/phonon-voices): the 37 English voices of its 102 (American, British, and character voices) | Apache 2.0 |
| Generation loop, text normalization and prompt planning (C++ port) | [gradium-ai/xn-ptts](https://github.com/gradium-ai/xn-ptts) `ptts` crate | MIT or Apache 2.0 |
| Model definition the graphs are cut from | Hugging Face `transformers` (`GradiumPhononTTSForConditionalGeneration`, `PocketTTSAudioTokenizer`) | Apache 2.0 |

Gradium's larger 40M models (`phonon-40m-*`) are CC BY-NC 4.0 and are deliberately not included;
the exporter is pinned to the 31M English checkpoint. The model card asks that Phonon not be used
to impersonate a real person or deceive listeners about who is speaking.

## Voices and cloning

A Phonon voice is a `[1, 512, frames]` tensor of speaker latents (the `speaker_wavs` tensor of a
`.safetensors` file, about 256 KB), fed to the model as a prompt. The encoder that produces those
latents from a recording is not part of the released weights, so Moonshine cannot clone into a
Phonon voice on the device; the on-device cloning path (`clone_from()` / `cloning()` /
`zipvoice/clone_audio`) remains ZipVoice.

Beyond the catalog, Gradium provides voices through [Gradium Studio](https://studio.gradium.ai/),
a hosted, account-based service: **voice cloning** from about ten seconds of clear speech and
**voice design** from a text description, each downloadable as a Phonon `.safetensors` file
("Download for Phonon" in the Studio, or the `clone-voice.py` / `download-voice.py` scripts in
`xn-ptts`, which need a Gradium API key). Such a file loads here unchanged: pass the bare voice
`phonon` with `phonon_voice_file=<path>` (or the `phonon/voice` in-memory key). Voices made that
way are under Gradium's terms of service rather than the catalog's Apache 2.0 license, and should
only be used with the speaker's consent.

## Files in this directory

Runtime assets, resolved under `g2p_root` (or supplied in memory via the C API with the same keys
prefixed `phonon/`).

| Key                          | Role                                                                 |
|------------------------------|----------------------------------------------------------------------|
| `embed_prompt.ort`           | Voice latents + text ids + duration delta -> prompt embeddings.      |
| `embed_latent.ort`           | One generated latent -> the next step's input embedding.             |
| `backbone.ort`               | 6-layer causal transformer with a KV cache; emits the flow condition and the end-of-speech logit. |
| `flow_head.ort`              | Flow-matching head: noise + condition -> velocity (one step).        |
| `mimi_transformer.ort`       | Mimi decoder, transformer half, with a sliding-window KV cache.      |
| `mimi_seanet.ort`            | Mimi decoder, convolutional half: hidden steps -> 24 kHz audio. Its transposed convolutions are exported as kernel-2 convolutions plus a channel-to-time interleave, which ORT runs several times faster. |
| `model.json`                 | Shapes, generation defaults (temperature, EOS threshold, frames after EOS) and token ids. |
| `tokenizer.bin`              | The SentencePiece Unigram vocabulary with scores, in the headed BinTokenizer form. |
| `voices.json`                | The voice catalog: id, name, locale, gender, description, file.      |
| `voices/<id>.safetensors`    | One voice: a `speaker_wavs` tensor of `[1, 512, frames]` speaker latents (about 256 KB). |
| `reference.f32`              | PyTorch's render of one sentence at temperature 0, which `phonon_tts_test` compares against. |

All graphs are float32; int8 quantization of the language-model graphs changes which frames the
model stops on and was rejected. The bundle is about 165 MB.

These files are **not** committed to git; `scripts/fetch-voice-assets.sh tts` fetches them from the
CDN once published, and `scripts/upload-tts-assets.sh` publishes a local tree.

## Regenerating

In a virtual environment with `torch`, `transformers>=5.21`, `onnx`, `onnxruntime`, `safetensors`
and `tokenizers`:

```bash
python3 scripts/export-phonon-model.py --out-dir core/moonshine-tts/data/phonon
```

The script cuts the six graphs out of the Hugging Face `transformers` implementation, checks each
against the PyTorch module it came from, runs the whole pipeline end to end against `model.generate`
at temperature 0, converts to `.ort` with `scripts/convert-models-to-ort.py`, writes
`tokenizer.bin` through `scripts/convert_tokenizer.py`, and downloads the English voices of
[gradium/phonon-voices](https://huggingface.co/gradium/phonon-voices).

## How generation works

The prompt is `[<|audio_prompt|> x voice frames][text ids][<|audio_bos|>]`. Each step the backbone
yields a 512-dim condition and an EOS logit; the flow head turns Gaussian noise scaled by
`sqrt(temperature)` into a 32-dim latent in one step; the latent is embedded as the next input.
Once the EOS logit passes its threshold the loop runs `frames_after_eos` more frames (3 for prompts
of four words or fewer, else 1). Latents are decoded one 80 ms frame at a time with the Mimi
decoder's state carried across calls (the transformer's KV cache; one frame of context re-run
through the convolutions), which is what makes streaming sample-exact with a whole render. The
flow LM for the next frame runs on a second thread while Mimi decodes the current one.

Text goes through the `ptts` preparation first: typography folding, numbers of a thousand or more
read with their scale words, sentence grouping into prompts of about 50 tokens, then capitalization
and a trailing full stop per prompt.
