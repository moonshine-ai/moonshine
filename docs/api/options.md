# Options

Most Moonshine constructors accept a string-to-string options map (Python dict, `moonshine_option_t` array in C, name/value pairs elsewhere). Values are always strings at the ABI; language bindings may accept bools and numbers and stringify them for you.

Use this page as the catalog of keys. Class and C API pages link here instead of repeating full descriptions.

- [Shared options](#shared-options)
- [Speech to Text](#speech-to-text)
- [Text to Speech](#text-to-speech)
- [Grapheme to Phonemes](#grapheme-to-phonemes)
- [Embeddings](#embeddings)
- [Speech clip extract](#speech-clip-extract)
- [Download manifests](#download-manifests)

## Shared options

These keys are recognized across multiple constructors. `log_api_calls` is stripped by a common parser before API-specific parsing; `ort_providers` and `coreml_cache_dir` apply wherever ONNX Runtime sessions are created.

| Key | Accepted by | Description |
| --- | --- | --- |
| `log_api_calls` | Transcriber, TTS, G2P, speech-clip extract, and other C entry points that run common option parsing | When true, log C API entry points and their arguments to stderr/console. |
| `ort_providers` (alias `ort_provider`) | Transcriber, TTS, G2P | Comma-separated, ordered ONNX Runtime execution providers (for example `CoreML,CPU`). Names are case-insensitive; short forms (`CPU`, `CoreML`, `NNAPI`) or full names work. Unset means CPU-only (recommended). Mobile libraries ship CPU-only — requesting another provider there is an error. See [execution providers](https://github.com/moonshine-ai/moonshine/blob/main/docs/execution-providers.md). |
| `coreml_cache_dir` | Transcriber, TTS, G2P | Directory for the CoreML compiled-model cache on macOS. Only used when `CoreML` is listed in `ort_providers`. |
| `log_profiling` | TTS, G2P | When true, log profiling information to the console. |
| `g2p_root` | TTS, G2P (and TTS dependency/voice listing) | Asset root for G2P and TTS file layout. Empty means the process current working directory. |
| `path_root` / `model_root` | Same as `g2p_root` | Aliases for `g2p_root`. |
| `tts_root` | TTS create / dependencies / voices | Alias for the asset root used when resolving TTS layout. |

MicTranscriber `.options()` and AgentFlow `.speech_options()` forward into the native transcriber and TTS loaders respectively, so the keys below apply there too.

## Speech to Text

Passed to `Transcriber(..., options=…)`, MicTranscriber `.options()`, and `moonshine_load_transcriber_from_files()` / `_from_memory_files()` / deprecated `_from_memory()`.

| Key | Default | Description |
| --- | --- | --- |
| `skip_transcription` | false | When true, run VAD/segmentation only (no STT). Use each line's audio buffer for further processing. |
| `max_tokens_per_second` | `6.5` | Truncate decoder loops when token rate looks pathological. Use about `13.0` for many non-Latin languages. |
| `use_speculative_decoding` | true | Streaming: verify the previous hypothesis and continue from the first mismatch. False falls back to greedy redecode from BOS. |
| `keyterms` | (none) | Comma-separated bias terms (streaming architectures only). See [Domain Customization](../models/domain-customization.md). Can also be set at runtime with `set_keyterms` / `moonshine_transcriber_set_keyterms()`. |
| `keyterm_boost` | `2.0` | Strength of key-term biasing. Raise towards 4.0 to favor the list at the cost of the words around it, lower towards 1.0 for the reverse. Above 4.0 it stops working. |
| `context` | (none) | A passage of free-form text to pick key terms out of, for when you have context but not a list (streaming architectures only). Added to any `keyterms`. Can also be set at runtime with `set_context` / `moonshine_transcriber_set_context()`. |
| `context_max_terms` | `200` | Most terms to take from `context`. Worth keeping modest: length is charged against the words you did not ask for. |
| `transcription_interval` | `0.5` | Seconds between automatic transcription passes (related to Python `update_interval`). |
| `vad_threshold` | `0.5` | VAD sensitivity. Lower → longer segments; higher → shorter chunks. `0` disables VAD (audio still chunked by `vad_max_segment_duration`). |
| `vad_window_duration` | `0.5` | Seconds of VAD scores to average when detecting speech. |
| `vad_hop_size` | `512` | VAD hop size in samples. |
| `vad_look_behind_sample_count` | `8192` | Samples to prepend when speech starts (compensates for averaged VAD lag), at 16 kHz. |
| `vad_max_segment_duration` | `15` | Maximum line length in seconds before a forced complete; threshold ramps down late in the segment. |
| `save_input_wav_path` | (none) | Folder path: write received audio as 16 kHz mono WAVs for debugging. |
| `log_ort_run` | false | Log ONNX Runtime inference runs and timings. |
| `word_timestamps` | false | Fill each line's `words` array. Needs the attention decoder asset. Implied by `identify_speakers`. |
| `use_speculative_decoding` | true | Streaming re-decode verifies the previous hypothesis instead of restarting from BOS. |
| `decode_incomplete_lines` | true | Decode in-progress lines so text can update while someone is still talking. Set false to wait until the line is complete. |
| `identify_speakers` | false | Enable diarization and `speaker_spans`. Needs diarization models ([details](https://github.com/moonshine-ai/moonshine/blob/main/docs/diarization-models.md)). |
| `diarization_model_dir` | (none) | Directory with `segmentation.ort` and `embedding.ort` when constructing a transcriber directly. |
| `diarization_cluster_cadence` | `2.0` | Minimum seconds of new audio between re-clustering passes. |
| `diarization_analyze_cadence` | `0` (= model default `1.0`) | Sliding-window step for segmentation/embedding. Live `add_audio` / `transcribe()` runs at most one window per call; remaining windows wait until the next call or Stop. Silent speaker classes skip embedding inference. |
| `diarization_cluster_window_sec` | `120` | Max recent history (seconds) for streaming VBx; `0` = unlimited. Batch/one-shot always uses full history. |
| `return_audio_data` | true | Include per-line PCM in transcript results. |
| `log_output_text` | false | Log STT text to the console. |
| `spelling_model_path` | (none) | Path to a spelling-CNN `.ort` for `MOONSHINE_FLAG_SPELLING_MODE`. |

Also accepts the [shared](#shared-options) keys `log_api_calls`, `ort_providers`, and `coreml_cache_dir`.

## Text to Speech

Passed to `TextToSpeech.options()`, AgentFlow `.speech_options()`, and `moonshine_create_tts_synthesizer_from_files()` / `_from_memory()`. The same set is used (where relevant) by `moonshine_get_tts_dependencies()` and `moonshine_get_tts_voices()`.

| Key | Description |
| --- | --- |
| `voice` | Catalog voice id. Prefix with `kokoro_`, `piper_`, or `zipvoice_` to select the vocoder (for example `kokoro_af_heart`). |
| `speed` | Speaking-rate multiplier. Also the only per-call override honored by `say()` / `synthesize()` / `moonshine_text_to_speech()` / `moonshine_phonemes_to_speech()`. |
| `lang` / `language` | Language tag when supplied via options (usually set by the constructor/`language()` setter instead). |
| `kokoro_dir` | Override Kokoro directory (`prosody.model.ort` + `prosody.weights.ort` + `decoder.model.ort` + `decoder.weights.ort` + `config.json` under it). |
| `kokoro_model` / `kokoro_model_onnx` | Override Kokoro model path. Names the model rather than a file that has to exist: the split pair beside it is preferred, and a single `.ort` at this exact path is loaded otherwise. |
| `kokoro_config` / `kokoro_config_json` | Override Kokoro config JSON path. |
| `piper_onnx` / `piper_model_onnx` / `piper_model` | Override Piper model path (must be `.ort`). |
| `piper_onnx_json` / `piper_model_json` / `piper_onnx_config` | Override Piper JSON sidecar. |
| `piper_voices_dir` / `voices_dir` | Override Piper voices directory. |
| `piper_voices_json_dir` / `voices_json_dir` | Override Piper `*.onnx.json` directory. |
| `normalize_audio` / `piper_normalize_audio` | Peak-normalize then apply gain/clip (default true). |
| `output_volume` / `piper_output_volume` | Linear gain after normalize (default `1`). |
| `piper_noise_scale` / `piper_noise_scale_override` | Piper inference noise scale. |
| `piper_noise_w` / `piper_noise_w_override` | Piper inference noise_w. |
| `zipvoice_clone_sample_rate` / `clone_sample_rate` | Sample rate for caller-supplied `zipvoice/clone_audio` (default `24000`). |
| `zipvoice_clone_transcript` / `clone_transcript` | Transcript for that clone clip. |
| `zipvoice_model` / `zipvoice_model_name` | `zipvoice` vs distilled; sets sampling defaults. |
| `zipvoice_distill` | Use distilled ZipVoice sampling defaults (default true). |
| `zipvoice_num_step` / `num_step` | Diffusion steps; `<=0` → model default. |
| `zipvoice_guidance_scale` / `guidance_scale` | Guidance scale; `<0` → model default. |
| `zipvoice_t_shift` / `t_shift` | Time-shift (default `0.5`). |
| `output` / `o` | Default WAV path for CLI-style tooling (default `out.wav`). |
| `engine` / `vocoder_engine` | Accepted but ignored (engine comes from the `voice` prefix). |

Also accepts [shared](#shared-options) root aliases and ORT keys. Unknown TTS keys are forwarded to the [G2P](#grapheme-to-phonemes) parser.

In-memory create uses file map keys (for example `kokoro/prosody.model.ort`, `zipvoice/clone_audio`, `clone_asr/...`) rather than option names — see the [C API](c-api.md#moonshine_create_tts_synthesizer_from_memory).

## Grapheme to Phonemes

Passed to `GraphemeToPhonemizer(..., options=…)` and `moonshine_create_grapheme_to_phonemizer_*` / `moonshine_get_g2p_dependencies()`.

### Roots and runtime

| Key | Description |
| --- | --- |
| `g2p_root` / `path_root` / `model_root` | Asset root (see [shared](#shared-options)). |
| `use_cuda` | Enable CUDA for G2P ORT sessions. |
| `oov_onnx_override` | Override English OOV model path/bytes. |
| `oov_onnx_config` | Override English OOV `onnx-config.json` UTF-8 text. |
| `allow_builtin_g2p_data` | Deprecated; ignored. |

Also accepts `ort_providers`, `coreml_cache_dir`, `log_profiling`, and `log_api_calls`.

### Lexicon and model path overrides

| Key | Sets |
| --- | --- |
| `english_dict_path` | `en_us/dict_filtered_heteronyms.tsv` |
| `german_dict_path` | `de/dict.tsv` |
| `french_dict_path` | `fr/dict.tsv` |
| `french_csv_dir` | `fr` (POS CSV directory) |
| `dutch_dict_path` | `nl/dict.tsv` |
| `italian_dict_path` | `it/dict.tsv` |
| `russian_dict_path` | `ru/dict.tsv` |
| `chinese_dict_path` | `zh_hans/dict.tsv` |
| `chinese_onnx_model_dir` | Chinese RoBERTa UPOS bundle directory |
| `korean_dict_path` | `ko/dict.tsv` |
| `vietnamese_dict_path` | `vi/dict.tsv` |
| `japanese_dict_path` | `ja/dict.tsv` |
| `japanese_onnx_model_dir` | Japanese tok-POS bundle directory |
| `arabic_dict_path` | `ar_msa/dict.tsv` |
| `arabic_onnx_model_dir` | Arabic diacritizer bundle directory |
| `hindi_dict_path` | `hi/dict.tsv` |
| `portuguese_dict_path` | Portuguese lexicon override |

### Language feature flags

Bools controlling language-specific G2P behavior (defaults are generally `true` unless noted):

| Key | Notes |
| --- | --- |
| `spanish_with_stress`, `spanish_narrow_obstruents` | Spanish |
| `german_with_stress`, `german_vocoder_stress` | German |
| `french_with_stress`, `french_liaison`, `french_liaison_optional`, `french_oov_rules`, `french_expand_cardinal_digits` | French |
| `dutch_with_stress`, `dutch_vocoder_stress`, `dutch_expand_cardinal_digits` | Dutch |
| `italian_with_stress`, `italian_vocoder_stress`, `italian_expand_cardinal_digits` | Italian |
| `russian_with_stress`, `russian_vocoder_stress` | Russian |
| `korean_expand_cardinal_digits` | Korean |
| `portuguese_with_stress`, `portuguese_vocoder_stress`, `portuguese_expand_cardinal_digits`, `portuguese_apply_pt_pt_final_esh` | Portuguese; `portuguese_keep_syllable_dots` defaults false |
| `turkish_with_stress`, `turkish_expand_cardinal_digits` | Turkish |
| `ukrainian_with_stress`, `ukrainian_expand_cardinal_digits` | Ukrainian |
| `hindi_with_stress`, `hindi_expand_cardinal_digits` | Hindi |

TTS-only keys (`voice`, Piper/Kokoro paths, and so on) are ignored when listing G2P dependencies.

## Embeddings

`moonshine_create_embedding_model()` takes `model_variant` as a dedicated argument (`fp32`, `fp16`, `q8`, `q4` default, `q4f16`), not an options map. Create-from-memory accepts an options array but currently ignores it.

For `moonshine_get_embedding_dependencies()`:

| Key | Description |
| --- | --- |
| `variant` (alias `model_variant`) | Which quantized/float file to include in the download manifest. |

## Speech clip extract

Passed to `moonshine_extract_speech_clip()` (and the Python/VoiceClone capture path that wraps it):

| Key | Default | Description |
| --- | --- | --- |
| `clip_duration_seconds` | `4` | Length of the window to extract. |
| `minimum_speech_seconds` | `2` | Minimum speech in the window for `is_complete`. |
| `vad_threshold` | `0.5` | Speech probability threshold. |
| `tail_pad_seconds` | `0` | Extra audio after the VAD window. |

Also accepts `log_api_calls`.

## Download manifests

### Speech to Text (`moonshine_get_stt_dependencies()`)

You can pass the same option list you would use to load the model. Only these change which files are listed:

| Key | Description |
| --- | --- |
| `model_arch` | Decimal string of a `MOONSHINE_MODEL_ARCH_*` constant; omitted → language default. |
| `word_timestamps` | Include the attention decoder in the download. |
| `include_spelling` / `spelling` | Include the spelling-CNN group when published for the language. |
| `spelling_model_path` | Non-empty path → same as including the spelling group. |

### TTS / G2P dependency and voice listing

Use the [TTS](#text-to-speech) and [G2P](#grapheme-to-phonemes) keys above (`voice`, `g2p_root`, and related). See `moonshine_get_tts_dependencies()`, `moonshine_get_tts_voices()`, and `moonshine_get_g2p_dependencies()` in the [C API](c-api.md).
