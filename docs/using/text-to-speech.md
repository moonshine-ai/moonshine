# Text to Speech

Voice interfaces often need to talk back, and Moonshine's `TextToSpeech` is designed to make that easy, across multiple languages. It's also self-contained, so you can use it independently from the transcription and agent modules.

- [Getting Started](#getting-started)
- [Streaming Text In](#streaming-text-in)
- [Voice Cloning](#voice-cloning)
- [Voice Samples](#voice-samples)
- [Converting Graphemes to Phonemes](#converting-graphemes-to-phonemes)

## Getting Started

You configure a synthesizer with chainable setters, call `load()` to fetch and open the voice, and then pass text into `say()` to speak it on the default audio device:

```python
from moonshine_voice import TextToSpeech

tts = TextToSpeech().language("fr")
tts.load()
tts.say("Bonjour, mon ami")
tts.wait()  # block until playback finishes
```

`load()` blocks, since the first call may have to download a voice. Pass `on_progress()` a handler to drive a progress bar:

```python
tts = TextToSpeech().language("fr").on_progress(lambda fraction, file: print(f"{fraction:.0%}"))
tts.load()
```

`say()` returns immediately and queues the text for background synthesis and playback. Calling `say()` multiple times queues each utterance in order, and the next utterance is pre-synthesized while the current one plays. You can also pass a list of strings, cancel everything with `stop()`, or poll with `is_talking()`:

```python
tts.say(["One.", "Two.", "Three."])
tts.stop()  # cancel remaining utterances and halt playback
```

If you're on a machine without an audio output, or want to do further processing, you can retrieve the audio samples using the `synthesize()` method:

<!-- doc-test: run -->
```python
from moonshine_voice import TextToSpeech

tts = TextToSpeech().language("en-us")
tts.load()
audio_data, sample_rate = tts.synthesize("Howdy, partner")
```

As you can see, text to speech supports multiple languages. To see which are available, run the `list_tts_languages()` function:

<!-- doc-test: run -->
```python
from moonshine_voice import list_tts_languages
list_tts_languages()

['ar-msa', 'de-de', 'en-gb', 'en-us', 'es-ar', 'es-es', 'es-mx', 'fr-fr', 'hi-in', 'it-it', 'ja-jp', 'ko-kr', 'nl-nl', 'pt-br', 'pt-pt', 'ru-ru', 'tr-tr', 'uk-ua', 'vi-vn', 'zh-hans']
```

For each language, you can list which voices are available:

<!-- doc-test: run -->
```python
from moonshine_voice import list_tts_voices

list_tts_voices("ru")

{'present': [], 'downloadable': ['piper_ru_RU-denis-medium', 'piper_ru_RU-dmitri-medium', 'piper_ru_RU-irina-medium', 'piper_ru_RU-ruslan-medium']}
```

If a voice is marked as `downloadable` that means if you pass it to `voice()` then Moonshine will download it to a cache automatically, and it will be available on your machine with no internet access required for subsequent calls.

## Streaming Text In

When the text comes from a language model, waiting for the whole reply before speaking any of it wastes the time the model spent generating. Push text as it arrives and get audio back as soon as there is enough to say:

```python
tts = TextToSpeech().language("en-us")
tts.load()

for chunk in tts.stream(llm_tokens()):
    print(f"{len(chunk.samples) / chunk.sample_rate:.2f}s: {chunk.text}")
```

There is no stream object to open or close. A synthesizer has one model and speaks one thing at a time, so pushing text starts a reply and ending input finishes it. Passing the text to `stream()` pushes and ends it for you; to drive it yourself, call `push_text()` from another thread and iterate `stream()` with no argument.

Text is held until a complete sentence or phrase arrives, because prosody depends on knowing where a clause ends — a synthesizer given one word at a time produces a list, not a sentence. Call `flush()` to synthesize what is buffered anyway, `end_input()` when no more text is coming, and `cancel_stream()` to abandon a reply someone has interrupted. Because a reply occupies the model while it plays, `say()` and `synthesize()` refuse until it finishes or is cancelled.

To speak the chunks rather than handle them yourself, use `say_stream()`, which feeds them into the same playback queue as `say()`:

```python
with tts.say_stream() as speech:
    for token in llm_tokens():
        tts.push_text(token)
    tts.end_input()
    speech.wait()
```

`AgentFlow` has the same method, so a reply can be spoken as the model writes it:

```python
with agent.say_stream() as push:
    for token in llm.stream(prompt):
        push(token)
```

Each chunk carries the samples, their sample rate, the text it covers, an utterance id, and whether it is the last chunk of that utterance. Utterances are numbered from one, so ids stay comparable across a reply that is flushed and refilled. A chunk that covers acoustic frames rather than a knowable span of characters has empty text.

The same surface exists in every binding: `pushText`/`flush`/`endInput`/`cancelStream` with an `onChunk` consumer in TypeScript, Swift and Java, plus an async iterator in TypeScript and an `AsyncThrowingStream` in Swift.

Most of the latency win comes from not waiting on the model: speaking a four-sentence reply written at fifty tokens a second, first audio arrives in well under a second streamed against a little over two seconds if you wait for the whole reply, and almost all of that gap is the model writing rather than the synthesizer synthesizing. Both engines also cut the first chunk below a sentence, so a long opening sentence starts playing before all of it has been decoded. Chunks grow as the reply goes on: a short first chunk buys the fast start, and longer later ones amortise the per-chunk cost over more audio. Kokoro's sub-sentence chunks are crossfaded, since its decoder normalises over whatever span it is handed and so cannot join seamlessly; Piper's reproduce the whole render sample for sample.

Streamed audio is levelled from a measurement taken per voice, because the peak normalization `say()` applies needs a finished waveform and streaming never has one. Loudness therefore lands close to `say()` rather than matching it exactly — expect a decibel or two, and a little more on an unusually quiet phrase.

## Voice Cloning

The integrated [ZipVoice model](https://github.com/k2-fsa/ZipVoice) can imitate someone's voice, given a short audio clip. Pass the clip to `clone_from()`, either as a path to a `.wav` file or as a `(pcm, sample_rate)` pair of mono float samples. You can also pass `transcript`, the text spoken in the clip; when omitted, Moonshine auto-transcribes the clip with its ASR model before cloning (this takes a few extra seconds on first use):

```python
from moonshine_voice import TextToSpeech
import importlib.resources;

clone_path = importlib.resources.files("moonshine_voice.assets").joinpath("clone-test.wav")
clone_transcript = "Ever tried. Ever failed. No matter. Try Again. Fail again. Fail better."

tts = TextToSpeech().language("en-us").cloning()
tts.load()
tts.clone_from(clone_path, transcript=clone_transcript)
tts.say("Ask not what your country can do for you, but what you can do for your country")
tts.wait()
```

`cloning()` tells `load()` to fetch ZipVoice and its clone-ASR assets up front, so `clone_from()` only swaps the reference clip. Call `cloning()` before `load()` — without it, `clone_from()` / `start_cloning()` raise a clear error. Catalog voices and cloning are mutually exclusive: `voice()` clears cloning, and `cloning()` clears the catalog voice.

To clone from someone speaking into the microphone rather than from a file, `start_cloning()` hands back a `VoiceClone` that listens until it has heard enough usable speech:

```python
clone = tts.start_cloning()
clone.on_ready(lambda: print("Got it, you can stop talking."))
clone.from_microphone()
tts.clone_from(clone)
```

Picking the clip out of the recording runs Moonshine's built-in voice-activity detector, which is compiled into the library, so nothing is downloaded for this step. `from_microphone()` blocks until the clip is ready or 20 seconds have passed; `on_progress()` reports how long it has been recording and how much speech it has found so far.

You can also try cloning from the command line. Since you won't always have easy access to a clean transcript of the speech you want to clone from, you can leave it out and have Moonshine automatically generate one, in both the API and command line.

<!-- doc-test: parse-only -->
```bash
curl -O -L 'https://github.com/moonshine-ai/moonshine/raw/refs/heads/main/language-bindings/python/src/moonshine_voice/assets/clone-test.wav'

python3 -m moonshine_voice.tts \
  --clone clone-test.wav \
  --text "I am so excited about Moonshine Voice's text to speech"
```

## Voice Samples

To help you choose a voice, here are sample clips of each one saying "Welcome to Moonshine Voice text to speech". Each entry is the voice name you can pass to `voice()`; click the ▶ next to it to hear it.

### ZipVoice

These voices were created using the zero-shot voice cloning capabilities of [ZipVoice](https://github.com/k2-fsa/ZipVoice), a high-quality flow-matching TTS model from the k2-fsa team. It takes significantly longer to generate than Kokoro or PiperTTS, but offers [voice cloning](text-to-speech.md#voice-cloning) and more realistic speech.

| | | |
| --- | --- | --- |
| `zipvoice_american_female` [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine@main/docs/audio/zipvoice_american_female.wav) | `zipvoice_american_male` [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine@main/docs/audio/zipvoice_american_male.wav) | `zipvoice_australian_male` [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine@main/docs/audio/zipvoice_australian_male.wav) |
| `zipvoice_canadian_female` [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine@main/docs/audio/zipvoice_canadian_female.wav) | `zipvoice_canadian_male` [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine@main/docs/audio/zipvoice_canadian_male.wav) | `zipvoice_english_female` [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine@main/docs/audio/zipvoice_english_female.wav) |
| `zipvoice_english_male` [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine@main/docs/audio/zipvoice_english_male.wav) | `zipvoice_indian_female` [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine@main/docs/audio/zipvoice_indian_female.wav) | `zipvoice_indian_male` [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine@main/docs/audio/zipvoice_indian_male.wav) |
| `zipvoice_irish_female` [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine@main/docs/audio/zipvoice_irish_female.wav) | `zipvoice_irish_male` [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine@main/docs/audio/zipvoice_irish_male.wav) | `zipvoice_new_zealand_female` [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine@main/docs/audio/zipvoice_new_zealand_female.wav) |
| `zipvoice_northern_irish_female` [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine@main/docs/audio/zipvoice_northern_irish_female.wav) | `zipvoice_south_african_female` [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine@main/docs/audio/zipvoice_south_african_female.wav) | `zipvoice_south_african_male` [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine@main/docs/audio/zipvoice_south_african_male.wav) |

### Kokoro

These voices come from the excellent [Kokoro](https://github.com/hexgrad/kokoro) project, an 82-million-parameter open-weight TTS model that delivers quality comparable to much larger models.

| American Female | American Male | British Female | British Male |
| --- | --- | --- | --- |
| `kokoro_af_alloy` [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine@main/docs/audio/kokoro_af_alloy.wav) | `kokoro_am_adam` [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine@main/docs/audio/kokoro_am_adam.wav) | `kokoro_bf_alice` [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine@main/docs/audio/kokoro_bf_alice.wav) | `kokoro_bm_daniel` [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine@main/docs/audio/kokoro_bm_daniel.wav) |
| `kokoro_af_aoede` [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine@main/docs/audio/kokoro_af_aoede.wav) | `kokoro_am_echo` [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine@main/docs/audio/kokoro_am_echo.wav) | `kokoro_bf_emma` [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine@main/docs/audio/kokoro_bf_emma.wav) | `kokoro_bm_fable` [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine@main/docs/audio/kokoro_bm_fable.wav) |
| `kokoro_af_bella` [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine@main/docs/audio/kokoro_af_bella.wav) | `kokoro_am_eric` [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine@main/docs/audio/kokoro_am_eric.wav) | `kokoro_bf_isabella` [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine@main/docs/audio/kokoro_bf_isabella.wav) | `kokoro_bm_george` [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine@main/docs/audio/kokoro_bm_george.wav) |
| `kokoro_af_heart` [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine@main/docs/audio/kokoro_af_heart.wav) | `kokoro_am_fenrir` [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine@main/docs/audio/kokoro_am_fenrir.wav) | `kokoro_bf_lily` [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine@main/docs/audio/kokoro_bf_lily.wav) | `kokoro_bm_lewis` [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine@main/docs/audio/kokoro_bm_lewis.wav) |
| `kokoro_af_jessica` [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine@main/docs/audio/kokoro_af_jessica.wav) | `kokoro_am_liam` [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine@main/docs/audio/kokoro_am_liam.wav) | | |
| `kokoro_af_kore` [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine@main/docs/audio/kokoro_af_kore.wav) | `kokoro_am_michael` [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine@main/docs/audio/kokoro_am_michael.wav) | | |
| `kokoro_af_nicole` [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine@main/docs/audio/kokoro_af_nicole.wav) | `kokoro_am_onyx` [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine@main/docs/audio/kokoro_am_onyx.wav) | | |
| `kokoro_af_nova` [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine@main/docs/audio/kokoro_af_nova.wav) | `kokoro_am_puck` [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine@main/docs/audio/kokoro_am_puck.wav) | | |
| `kokoro_af_river` [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine@main/docs/audio/kokoro_af_river.wav) | `kokoro_am_santa` [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine@main/docs/audio/kokoro_am_santa.wav) | | |
| `kokoro_af_sarah` [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine@main/docs/audio/kokoro_af_sarah.wav) | | | |
| `kokoro_af_sky` [▶](https://cdn.jsdelivr.net/gh/moonshine-ai/moonshine@main/docs/audio/kokoro_af_sky.wav) | | | |

### Piper TTS

The [Piper](https://github.com/OHF-Voice/piper1-gpl) project provides over a hundred lightweight voices across all of the languages Moonshine supports, from many contributors — too many to sample here. You can listen to every Piper voice on the [Piper voice samples page](https://rhasspy.github.io/piper-samples/), and use any of them with Moonshine through the `piper_` voice names returned by `list_tts_voices()`.

## Converting Graphemes to Phonemes

As you may notice from the voice names, Moonshine Voice uses models from the fantastic [Kokoro](https://github.com/hexgrad/kokoro) and [PiperTTS](https://huggingface.co/rhasspy/piper-voices) projects. You can find full details on all the model and data sources we use for text to speech at [core/moonshine-tts/data/README.md](https://github.com/moonshine-ai/moonshine/blob/main/core/moonshine-tts/data/README.md). 

Given that there are other great TTS projects out there, why does the world need yet another implementation? Moonshine tries to run on as many platforms as possible and supports commercial applications, and both Kokoro and Piper use [espeak-ng](https://github.com/espeak-ng/espeak-ng/) to convert text strings into phonemes, representations of the noises associated with the sentence, in the International Pronunciation Alphabet. Espeak-ng is licensed under the GPL, and while I am a fan of free software, the terms do make it hard to incorporate into applications that don't also release their source code under a similar license.

In the cloud this isn't as much of an issue, as many uses of espeak-ng can be implemented by calling out to an external executable, so the dependency isn't as problematic. This isn't an option on many edge operating systems unfortunately, as the only way to include code on iOS or Android is to link it into the application, which requires open sourcing the calling code.

To allow wider usage, we developed our own "grapheme to phoneme" module that performs a similar role, but has been written from scratch. You'll find the implementation in [core/moonshine-tts](https://github.com/moonshine-ai/moonshine/blob/main/core/moonshine-tts) and it's released under the same MIT License as the rest of this code base.

Every language requires a different process to convert its written form into speech, and often it varies by dialect too. This is why espeak-ng is so widely used, it has had years of work put into it to encode linguistic knowledge into a complex set of rules, many of which are heuristics that require a lot of testing to get right. The Moonshine Voice G2P engine is still new, and will need similar tuning to handle all of the variations across languages, but I'm hoping the initial implementation is a good start and will benefit from community feedback and contributions over time. Here are the current results for intelligibility across languages, using [scripts/tts_g2p_intelligibility.py](https://github.com/moonshine-ai/moonshine/blob/main/scripts/tts_g2p_intelligibility.py):

| Language | Moonshine CER | Reference CER |
| --- | --- | --- |
| ar_msa | 20.8% | 15.3% |
| de_de | 18.3% | 9.2% |
| en_us | 12.6% | 9.8% |
| es_ar | 7.9% | 10.6% |
| es_es | 4.2% | 4.5% |
| es_mx | 3.2% | 2.6% |
| fr_fr | 14.8% | 9.4% |
| hi_in | 26.5% | 15.9% |
| it_it | 24.2% | 11.4% |
| ja_jp | 38.1% | 16.8% |
| ko_kr | 25.0% | 18.6% |
| nl_nl | 15.9% | 3.3% |
| pt_br | 19.7% | 4.9% |
| pt_pt | 43.8% | 24.6% |
| ru_ru | 16.9% | 5.0% |
| tr_tr | 8.9% | 7.9% |
| uk_ua | 27.7% | 15.6% |
| vi_vn | 79.0% | 36.5% |
| zh_hans | 37.8% | 32.6% |

If you want access to just the grapheme to phoneme capability, without the speech synthesis, you can call it directly:

<!-- doc-test: run -->
```python
from moonshine_voice import GraphemeToPhonemizer

g2p = GraphemeToPhonemizer("en-us")
g2p.to_ipa("Hello world")

'həlˈoʊ wˈɝld'
```
