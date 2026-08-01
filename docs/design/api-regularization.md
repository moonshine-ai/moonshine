# Regularizing the client APIs

This document plans the work needed to make the JavaScript, Python, Swift and
Java APIs read the same way, and to back the code samples on the web demo pages
with example programs that are actually compiled and run. C++ is covered
separately in [the C++ section](#the-c-binding) at the end, because a different
set of rules applies to it.

It follows [api-principles.md](api-principles.md), which sets the goals, and
[api-comparison.md](api-comparison.md), which specifies the target API shape.
Neither of those changes. This document is about the distance still to travel
and the order to travel it in.

## Where things stand

The redesign described in `api-comparison.md` has largely landed. Construct,
configure with chainable setters, `load()`, then `start()` is the real shape of
the JavaScript, Swift and Java bindings today. `onText`, `onLine`, `onProgress`
and `onError` exist in all three. `VoiceClone` exists in Swift and Java with
`onReady`, `fromMicrophone`, `addAudio` and `isReady`. `DialogFlow` was ported
to both. `IntentRecognizer` is gone and phrase matching is internal.

Python was the exception, and only half of it. `DialogFlow` there was already
fully chainable and matched the target. `MicTranscriber` and `TextToSpeech` did
not: both loaded inside `__init__`, `MicTranscriber` made the caller resolve a
model path first, and cloning was a constructor argument with no
`clone_from()`. Phase 1 below brought both up to the pattern, so all four
languages now agree.

So the remaining work is three separable pieces: close two boilerplate gaps in
Swift and Java, fill four holes in the example matrix, and build the snippet
pipeline that lets the web pages quote those examples instead of duplicating
them.

`Transcriber` stays as it is in every language. The redesign deliberately left
the low-level type alone, and none of the three demo components need it.

## The example matrix

Three components, four languages. Twelve cells, seven of them currently filled.

| | Web | Python | Swift (iOS) | Android |
| --- | --- | --- | --- | --- |
| Microphone speech to text | done | done | done | done, Java |
| Text to speech with cloning | done | done | no cloning | no cloning, Kotlin |
| DialogFlow agent | done | done | missing | missing |

The macOS command line examples are a useful head start on the Swift gaps:
`examples/macos/DialogFlow` and `examples/macos/TextToSpeech --clone` are
working implementations of both.

## Phase 1: Python catches up (done)

The largest single piece, and the one everything else waited on, because the
Python examples could not be rewritten until the API they demonstrate existed.

### Download progress (done)

`get_model_for_language()` blocked and printed a `tqdm` bar to stderr with no
callback, so there was nothing to feed an `on_progress` handler. It now takes
one, reported as the same `(fraction, filename)` pair the other three bindings
use, and so do `download_tts_assets()`, `download_g2p_assets()` and
`ensure_tts_voice_downloaded()`. Supplying a handler silences the tqdm bars, on
the assumption that the caller is drawing its own.

Underneath, `download_file()` reports byte deltas and a `_ProgressTracker`
turns them into one monotonic fraction. Where the manifest declares a size per
file the fraction is byte-weighted; the TTS and G2P dependency lists are bare
keys with no sizes, so those fall back to counting files. One tracker spans
both the transcriber and the spelling model, so the bar fills once rather than
twice.

### MicTranscriber (done)

Today:

```python
model_path, model_arch = get_model_for_language("en")
mic = MicTranscriber(model_path=model_path, model_arch=model_arch)

class Printer(TranscriptEventListener):
    def on_line_text_changed(self, event):
        show_in_progress(event.line.text)
    def on_line_completed(self, event):
        append_line(event.line.text)

mic.add_listener(Printer())
mic.start()
```

Target:

```python
mic = (MicTranscriber()
       .on_text(show_in_progress)
       .on_line(lambda line: append_line(line.text)))

mic.load()
mic.start()
```

What landed: a no-argument constructor that cannot fail; `language()`,
`model_arch()`, `models_from()`, `use_transcriber()`, `update_interval()`,
`options()`, `spelling_model()`, `transcribe_flags()`, `device()`,
`samplerate()`, `channels()`, `blocksize()`, `on_text()`, `on_line()`,
`on_error()` and `on_progress()` as chainable setters; a real `load()` doing
the downloading and native setup that used to happen in `__init__`; `mute()`,
to match Swift; and `__enter__`/`__exit__`, which `Transcriber` and
`TextToSpeech` already had but `MicTranscriber` did not.

`load()` also finds the spelling model for the language by itself, which used
to be the caller's job, so `transcribe_flags(MOONSHINE_FLAG_SPELLING_MODE)`
works without any extra setup. `spelling_model(None)` opts out.

`add_listener()` and `TranscriptEventListener` stay, undeprecated. They are the
documented escape hatch for line ids, speaker spans and word timings in the
other three bindings, and Python should match. What goes away is the requirement
to use them for the common case. Listeners registered before `load()` are held
and attached once the stream exists, the same trick Swift uses, so a builder
chain can register everything up front.

The old constructor signature is a breaking change. It raises a `TypeError`
naming the replacement rather than a bare argument-count error.

### TextToSpeech (done)

Before, the constructor took a required `language`, did the downloading, and
accepted `clone=` and `clone_transcript=` for cloning. There was no
`clone_from()` and no `VoiceClone`.

Now:

```python
tts = TextToSpeech().language("en_us").voice("kokoro_af_heart")
tts.load()
tts.say("Hello from Moonshine.")

tts.clone_from("some-speech.wav")
tts.say("Now I sound like you.")
```

and the streaming form, which Python had no equivalent of at all:

```python
clone = tts.start_cloning()
clone.on_ready(lambda: print("Got it, you can stop talking."))
clone.from_microphone()
tts.clone_from(clone)
```

What landed: a no-argument constructor that cannot fail; `language()`,
`voice()`, `models_from()`, `cloning()`, `options()`, `output_device()`,
`volume()`, `debug()` and `on_progress()` as chainable setters; a real `load()`
doing the validation, downloading and native setup that used to happen in
`__init__`; and `clone_from()` / `start_cloning()` / `is_cloned` to match
Swift. The read-only `language` property became the `language()` setter, so the
tag is now `language_tag`, as it is in Swift.

`clone_from()` accepts a `.wav` path, a `(pcm, sample_rate)` pair, or a
`VoiceClone`, and rebuilds the synthesizer in place, freeing the previous one.
That means a synthesizer can start out on a catalog voice and be switched to a
cloned one later, which the old constructor-only path could not express.
`cloning()` fetches the ZipVoice engine during `load()` so the first clone is
quick; without it the engine comes down on the first `clone_from()` instead.
Since there is no voice until a clip arrives, `say()` on a `cloning()`
synthesizer that has not been cloned yet raises an error naming `clone_from()`
rather than a generic "not loaded".

`VoiceClone` is backed by the same `moonshine_extract_speech_clip` core call
that Swift and Java use, so this was binding work rather than new algorithm
work: `moonshine_api.py` grew a `SpeechClipC` struct and a
`moonshine_extract_speech_clip()` wrapper returning a `SpeechClip`, and
`voice_clone.py` layers incremental capture on top. `add_audio()` accumulates
and re-runs the search four times a second rather than on every buffer, and
`from_microphone()` blocks until the clip is ready or 20 seconds have passed,
taking the best window it has at that point. Because the detector is compiled
into the library, none of this downloads anything.

The old constructor signature is a breaking change, and like `MicTranscriber`
it raises a `TypeError` naming the replacement.

The clone voice needed one special case. A bare `zipvoice` names the engine
rather than a catalog voice, so it is excluded from the catalog validation and
the per-voice download step that a real voice id goes through; its model files
come down with the rest of the language's assets. Getting that wrong shows up
as a 404 for a voice file that was never published.

### DialogFlow rewiring

Python's `DialogFlow.load()` built `MicTranscriber(model_path=...)` and
`TextToSpeech(language=...)` directly and bridged transcript events through an
internal `_TranscriptBridge(TranscriptEventListener)`. Both halves are now on
the new builders, and both report real per-file download progress through
`on_progress()` instead of the placeholder 0.0 and 1.0 they used to emit either
side of a blocking call. The public `DialogFlow` API has not changed.

### Model architecture defaults, and a bug in Swift and Java

Python defaulted to `ModelArch.TINY` for `MicTranscriber`, against
`MEDIUM_STREAMING` everywhere else. Checking the catalog before copying that
default turned up a problem: **only English publishes a medium streaming
model.** Every other language has exactly one model, `BASE` or `TINY`:

| Language | Models published |
| --- | --- |
| en | medium streaming (default), small streaming, base, tiny streaming, tiny |
| ja | base (default), tiny |
| ar, es, vi, uk, zh | base |
| ko | tiny |

Asking the native catalog for a language and arch it does not have is a hard
error, not a downgrade: `moonshine_get_stt_dependencies` logs `unknown language
"es" or model_arch` and returns invalid-argument.

So Python now defaults to "the catalog's recommended model for this language",
which is medium streaming for English and the only published model everywhere
else. Naming an arch explicitly still fails loudly if it isn't published,
which is the right behaviour for an explicit request.

Swift and Java hardcode `mediumStreaming` as the field default in both
`MicTranscriber` and `DialogFlow`, with no fallback, which means
`MicTranscriber().language("es").load()` fails on both. Worth fixing in phase
2, either by resolving the default per language in each binding or by teaching
the native catalog to fall back.

### models_from means two different things

Swift and Java both document `modelsFrom(directory)` as "loads the model from a
directory you supply rather than downloading it". Python's `DialogFlow` uses
the same name for a cache root: "reads and caches model files under directory
instead of the default cache". Those are different operations.

The new `MicTranscriber.models_from()` follows Swift and Java. `DialogFlow` was
left alone for now (it resolves the model itself and hands the resulting
directory to `MicTranscriber.models_from()`, so behaviour is unchanged), but
one name meaning two things across bindings is exactly what this project is
supposed to remove. Settle it in phase 2.

### Tests

There was no direct coverage of `MicTranscriber`'s or `TextToSpeech`'s public
surface, only a threading test that mocked the transcriber out and a CLI smoke
test, so the new APIs got tests written alongside them:
`python/tests/test_mic_transcriber_api.py` and `python/tests/test_tts_api.py`
for the builders, `python/tests/test_download_progress.py` for the progress
plumbing, and `python/tests/test_voice_clone.py` for clip capture. The last of
those runs against the real detector rather than a mock, since it needs no
model files.

Two gaps in the harness turned up while doing it, both now fixed.
`scripts/test-python.sh` named two test files explicitly, so
`test_mic_transcriber_threading.py` had never run in CI; it now runs the
directory. And the microphone smoke test only asserted that the exit code was
not an argparse error, so a module that crashed on startup still passed; it now
requires the banner the module prints just before opening the capture device,
which works whether or not the machine has a microphone.

## Phase 2: remaining boilerplate in Swift and Java

Two things still make these read worse than the JavaScript, plus some tidying
that is now cheap because backwards compatibility is not a constraint.

### Model architecture defaults

`MicTranscriber().language("es").load()` fails in both bindings, because both
hardcode `mediumStreaming` and only English publishes one. See the phase 1
section above for the catalog. Either resolve the language's default arch in
each binding, or teach the native catalog to fall back when the requested arch
isn't published. The native fix is one change instead of three, and would also
cover any future binding.

### Swift callback threading

`onText` and `onLine` fire on the audio worker thread, so every user interface
application wraps each one in `Task { @MainActor in ... }`. The iOS Transcriber
example does this four times in a single builder chain. Java already solved
this: its callbacks default to the main thread with `callbacksOnMainThread(false)`
to opt out. Swift should match, with the same opt-out.

This is the clearest remaining win. It removes real lines from real code rather
than saving a token.

### One name, two meanings

Decide whether `models_from()` means "the model files are in this directory" or
"download into this directory", and make all four bindings agree. Swift, Java
and the new Python `MicTranscriber` mean the first; Python's `DialogFlow` means
the second. Both operations are useful, so this may need two names rather than
one winner.

### Legacy entry points

Now removable, since compatibility is not a concern:

- Swift: `MicTranscriber(modelPath:modelArch:...)`,
  `TextToSpeech(language:g2pRoot:...)`,
  `TextToSpeech(language:g2pRoot:clonePCM:...)`, and `Transcriber.load(...)`,
  which still reports the old `DownloadProgress(loaded, total, file)` shape
  rather than `(fraction, file)`.
- Java: `Transcriber.loadFromFiles`, `loadFromAssets`, and `CatalogLoader`.
- Both: `ModelSpec` is still public although `api-comparison.md` says it should
  be internal.

### Java Context: decided, no change

The conclusion is to leave this alone. The analysis behind that is recorded
below, because the option is cheap enough that it will come up again.

Every Java entry point takes a `Context`:

```java
MicTranscriber mic = new MicTranscriber(this);
```

where JavaScript, Swift and Python take nothing. The library could capture the
application context itself through an `androidx.startup` `Initializer` and offer
no-argument constructors.

**What it would cost.** Less than expected. `androidx.startup:startup-runtime`
is already on the classpath of every consuming application: the library depends
on `androidx.work:work-runtime:2.9.1`, whose POM depends on
`startup-runtime:1.1.1`. WorkManager therefore already merges
`androidx.startup.InitializationProvider` into every consuming manifest, so this
adds a `<meta-data>` entry to an existing provider rather than a new provider,
and no new artifact reaches consumers. The library manifest already contributes
an activity and three permissions, so contributing manifest entries is
established practice here.

Every constructor already calls `getApplicationContext()` immediately and none
of them need an `Activity`. Context is used for the model cache directory, the
`RECORD_AUDIO` permission check and dialog, and `AudioTrack` construction. So
holding the application context statically is semantically identical to what
happens today, and carries no leak risk.

**What it would cost that is real.** One new failure mode. Applications that
disable App Startup, either with `tools:node="remove"` on the provider or by
driving `AppInitializer` manually, would get a null context. Plain JVM unit
tests have no content providers either, though Robolectric and instrumentation
tests do. Both are handled by keeping the `Context` overloads and throwing
something that names the fix, but it is a support burden that does not exist
today.

**What it would not fix.** `load()`, `start()`, `say()` and `cloneFrom()` block
and still need an executor. That is inherent to Java without coroutines and
`api-comparison.md` accepts it. So the Java snippet keeps its
`executor.execute(...)` wrapper either way, and dropping `this` does not make it
match JavaScript, it just makes it one token shorter.

**Decision: do not do this, at least not now.** Principle two in
`api-principles.md` puts platform idiom above cross-language uniformity, and
passing a `Context` is the Android idiom. Room, Glide and WorkManager's own
public API all take one. The benefit here is one token in a snippet that will
not line up with JavaScript regardless, and the cost is a failure mode in
applications that manage their own startup. If the asymmetry turns out to grate
once the language tabs are live, it is an easy additive change to make later.

## Phase 3: fill the example matrix

Blocked on phase 1 for Python and phase 2 for Swift.

- **Python microphone speech to text.** Done: `examples/python/mic_transcription.py`
  is on the new API, and `examples/python/ollama-voice/ollama_voice.py` with it.
- **Python text to speech.** Done: `examples/python/text_to_speech.py` covers
  speaking, cloning from a file, and cloning from the microphone.
- **iOS text to speech.** Add cloning to the existing app.
  `examples/macos/TextToSpeech --clone` is the reference.
- **iOS DialogFlow.** New Xcode project, porting `examples/macos/DialogFlow`.
- **Android text to speech.** Rewrite from Kotlin to Java and add cloning. This
  leaves the repository with no Kotlin example, which is accepted: the Java API
  is callable from Kotlin unchanged, and one language per platform keeps the
  snippet tabs honest.
- **Android DialogFlow.** New Gradle project in Java.

Each Android example is a standalone Gradle project with its own wrapper,
settings file and version catalog. There is no shared parent build, so a new
example means copying a project skeleton and updating three identifiers.
`scripts/test-examples.sh` discovers examples by looking for `gradlew` and
`.xcodeproj`, so new ones are picked up without changing the script.

## Phase 4: snippets that come from the examples

Nothing extracts or verifies code samples today. The web pages hold hardcoded
template strings, and `api-comparison.md` quotes example files by hand with no
check that the quotes still match.

### Mechanism

Region markers in the real example sources, using each language's line comment:

```python
# snippet: stt-mic
mic = MicTranscriber().on_text(show_in_progress).on_line(append_line)
mic.load()
mic.start()
# end snippet
```

A script walks `examples/`, collects every marked region keyed by snippet name
and language, and writes a generated module the web pages import in place of
their inline strings. A test regenerates and diffs, so a snippet that drifts
from its example fails the build.

Application-specific identifiers inside the regions are acceptable. A reader
seeing `transcriptText.setText(...)` in the Android tab learns more than they
would from an invented placeholder, and the alternative is a snippet nobody
compiles. Where the surrounding code is genuinely too tangled to mark a clean
region, the example gets refactored so the marked part is a self-contained
function, which improves the example as well.

### Widening what gets tested

For "the snippets are tested" to be true, more examples have to be built than
are built today. Currently `scripts/test-examples.sh` compiles the Android
examples with `assembleDebug` and the iOS examples with `xcodebuild`, and
`scripts/build-all-platforms.sh` runs it only in the `publish-examples` stage of
a release. The web, Python and macOS examples are packaged without being built
or run at all, and no GitHub Actions workflow touches examples.

So this phase also needs: the Python examples exercised the way
`python/tests/test_modules.py` exercises the library modules, the existing
Puppeteer suite at `wasm/tests/web-examples.integration.test.mjs` brought into
the pipeline rather than left opt-in, and the snippet freshness check run
somewhere that a pull request will notice.

Two smaller pieces of this landed early, in phase 1, because the Python work
kept tripping over them: `scripts/test-python.sh` now runs the whole tests
directory rather than two named files, and the microphone smoke test now checks
the module reached the capture device rather than merely avoiding an argparse
error.

## Phase 5: language tabs on the web pages (built, on all three)

The original motivation, and now standing on all three demo pages ahead of phase
4, to see how it reads before committing to the extraction machinery. The tabs
are real; only their source is provisional.

The layout risk turned out to be smaller than the estimate. Reserving the
tallest snippet costs about 46px of dead space on the speech to text page, not
the 90px predicted, because the four microphone samples land within two lines of
each other rather than four. The reservation needs no measuring: the panes share
one CSS grid cell and the inactive ones are `visibility: hidden`, so the panel is
always as tall as the longest and switching language moves nothing below it. A
test drives every tab on every page and asserts the height does not change.

The three things that phase called out are done.

`highlight()` takes a language and looks up a keyword list, a comment pattern and
a method-call pattern per language. Swift needs its own call pattern, because
`.onText { … }` is a call with a trailing closure and the JavaScript pattern only
recognises one followed by `(`. Fixed while in there: strings are now stashed
before the comment scan rather than after, so a `//` inside a string is no longer
mistaken for the start of a comment.

The install line swaps with the tab, and carries an optional hint for the two
where pasting the line somewhere is not the obvious move: `Xcode ▸ Add Package
Dependencies…` beside the Swift Package Manager URL, and `build.gradle.kts
dependencies` beside the Gradle coordinate.

The headings that read "The code behind this page" now read "Live transcription,
in your language" and "Speech synthesis, in your language", each with a hint
saying the JavaScript tab is the code running on this page — which keeps the
claim the old heading was making, and confines it to the tab where it is true.

One collision worth recording: the site navigation already owns `.ms-tab`, so
the language tabs are `.ms-lang-tab`. On a narrow screen they scroll sideways
rather than wrapping, so the Copy button is never pushed onto a line of its own,
and the filename caption is hidden below `40rem` because the active tab already
says which language it is.

### The caption links to the source

Each snippet carries a repository-relative `path`, and the filename caption is an
anchor to it on the main branch — a tag would rot as the examples move between
releases. A snippet without a path stays a plain caption rather than becoming a
link to nowhere. A test asserts every path resolves to a file that exists, so a
moved example fails the suite rather than shipping a 404 to a reader.

### The voice agent page, which follows along as it runs

That page highlights the line the conversation is parked on, so its tabs need
more than a listing swap: the same flow lands on different lines in each
language, because Swift's `guard` needs a closing brace where JavaScript's `if`
does not. Each snippet therefore carries its own `steps` map, `codePanel` gained
an `onTab` callback, and the page remembers which step it is on so switching
language mid-conversation moves the highlight to the equivalent line rather than
dropping it.

This is also the one page whose snippets are not lifted verbatim: each is the
page's own trimmed flow written against that language's real API. A test checks
every step lands on a line mentioning the call it claims to be, which is the
part a careless edit would silently break.

Building it caught a real bug in `markCodeStep`, which cleared only the open
pane. The language a reader switched away from kept its highlight, waiting to
reappear on the way back. It now clears every pane and marks only the open one.

### What is still provisional

`examples/web/assets/snippets.js` is hand-copied from the real examples and
nothing checks that the code itself stays in step with them, only that the files
it names still exist. Phase 4 replaces that file with a generated one; it exports
the shape the generator should emit, so the pages should not need touching again.
Until then, an example that changes leaves the web page quietly stale, which is
the whole problem phase 4 exists to solve.

Two tabs are honest about gaps in the matrix rather than papering over them. The
Android text to speech tab is Kotlin and stops after `say`, because that is what
`examples/android/TextToSpeech` is and does; the Swift cloning lines on that page
come from the macOS example, since the iOS one does not clone. And the voice
agent page has three tabs, not four, because there is no Android DialogFlow
example to take a fourth from — the Java binding has the class, the matrix has
the hole. All three close in phase 3.

## Sequencing

Phase 1 and phase 2 are independent of each other and can run in parallel. Phase
3 depends on both, because the Python examples need the new Python API and the
Swift examples should be written against main-thread callbacks rather than
converted afterwards. Phase 4 depends on phase 3, and phase 5 on phase 4.

The one ordering trap: if the Java `Context` question is ever reopened and
answered yes, it has to land before the two new Java examples are written, not
after, so the snippets do not have to be revised twice.

## The C++ binding

`core/moonshine-cpp.h` is header-only and sits directly on the C API, with no
runtime of its own. That rules out a chunk of the target shape rather than
merely postponing it, so the goal here is narrower than for the other four: not
"read the same way", but "wherever C++ *can* do something the others do, do it
under the same name and in the same shape".

### What C++ cannot have, and why that is not a gap

The redesign's conventions assume a runtime that can open devices and fetch
things. C++ has neither, and giving it either would mean picking an HTTP client
and an audio backend for every application that links the library.

- **`MicTranscriber`** and **`DialogFlow`** need a capture device. Absent.
- **`say()`** needs an output device. `synthesize()` returns the samples, which
  is the half that does not need one.
- **`VoiceClone::fromMicrophone()`**, for the same reason. `addAudio()` is the
  half that does not.
- **Downloading**, so `load()`, `.modelsFrom()` and `onProgress` have nothing to
  do. Construction takes a path or a buffer, and cannot be deferred.

Convention 1 ("construct, configure, load") therefore does not apply: with no
download step there is no slow call to separate out, and a constructor that
cannot fail would just move the failure somewhere less obvious. Convention 4
(named `onText` / `onLine` callbacks) does not apply either, but for a different
reason: those live on `MicTranscriber`, and the redesign deliberately left the
low-level `Transcriber` alone in every language. C++'s `Transcriber` already
matches Swift's and Python's method for method, so adding named callbacks there
would have made it the odd one out rather than the consistent one.

### What was wrong, and is now fixed

Four things, all of which C++ could have had all along.

**Manifests were half-exposed.** `TextToSpeech::getDependencies` and
`GraphemeToPhonemizer::getDependencies` existed, but the speech-to-text,
embedding and diarization equivalents did not, nor either catalog. Computing a
manifest is not downloading — it returns JSON naming the files, URLs, sizes and
checksums — and it is exactly what a binding with no HTTP client needs most,
since it is the only way to learn what to fetch. This got worse when the
diarization models became a download, which left `identify_speakers` reachable
from C++ with no in-API way to find the two models it now requires. Added:
`Transcriber::getDependencies`, `Transcriber::getDiarizationDependencies`,
`Transcriber::getCatalog`, `EmbeddingModel::getDependencies` and
`EmbeddingModel::getCatalog`.

**In-memory loading was half-exposed.** `Transcriber` and `EmbeddingModel` had
`loadFromMemory`; `TextToSpeech` and `GraphemeToPhonemizer` did not, although
the C API has `_from_memory` for both. Added, with the same keyed-map signature
the other two use.

**Options had two types, one of them a trap.** `Transcriber` took owning
`(name, value)` string pairs; `TextToSpeech` and `GraphemeToPhonemizer` took
`std::vector<moonshine_option_t>`, the raw C struct of borrowed `const char *`.
That made `options.push_back({"voice", someString.c_str()})` compile and then
read freed memory if the string was a temporary — and the shipped C++ example
was one refactor away from it. Everything now takes `moonshine::Options`, and
brace-initialised call sites are unchanged because `{"voice", "x"}` builds a
pair just as happily as it built the struct.

**There was no voice cloning.** `cloneFrom` and `startCloning` are in all four
other bindings and were in none of C++, even though the machinery is entirely
platform-neutral: `moonshine_extract_speech_clip` runs the voice-activity
detector that is compiled into the library, so finding a reference clip needs no
models and no network. Added `extractSpeechClip()`, a `VoiceClone` that
accumulates audio and reports `isReady()`, and `TextToSpeech::cloneFrom` /
`startCloning` / `isCloned`.

Two details of that worth recording. `VoiceClone` holds a `shared_ptr` to its
state, so copies observe one capture; it is a reference type in Swift and Java
and behaving differently in C++ would surprise anyone porting between them. And
`TextToSpeech` now remembers the language, options and asset buffers it was
built from, because `cloneFrom` rebuilds the synthesizer in place and has to
replay them — the C API borrows the reference-clip bytes rather than copying, so
the object owns them for the new handle's lifetime.

Where the other bindings download a small speech-to-text model to transcribe the
reference clip, C++ takes either the transcript or a `Transcriber` the caller
already has. That is the "offline is possible, just longer" trade from
convention 5, applied to the one binding that is always offline.

### Naming, which was mostly already right

Worth stating because it was checked rather than assumed. `transcribeWithoutStreaming`,
`toIpa`, `synthesize`, `synthesizeFromPhonemes` and `EmbeddingModel` all already
matched the majority spelling; where C++ differs from one other binding, that
binding is the outlier (JavaScript alone says `transcribe` and
`textToPhonemes`). The only rename in this pass was a stale doc comment
referring to `GraphemeToPhonemizer::textToPhonemes`, which has never existed in
C++.

### Still open

- `Transcriber::loadFromMemory` takes `updateInterval` positionally between the
  architecture and the options, which no other overload does.
- `moonshine::Error` is a very general name for the transcript error event, in a
  namespace that also throws `MoonshineException`.
- The header still advertises itself as C++11 in its opening comment. Nothing
  disproves it, but nothing checks it either, and the library builds at C++20.

## Settled questions

- Python's `add_listener()` and `TranscriptEventListener` are kept and not
  deprecated. They are the escape hatch for line ids, speaker spans and word
  timings, and the other three bindings all offer one.
- Snippets are extracted from the real examples, keeping whatever
  application-specific identifiers appear there, rather than written as
  standalone minimal functions.
- Java keeps its `Context` parameters. See the analysis in phase 2.
- All Android examples are Java, and the repository keeps no Kotlin example.
- C++ does not get `MicTranscriber`, `DialogFlow`, `say()` or downloading. See
  the C++ section for what that rules out and what it does not.
