# Moonshine Voice for the Web (WebAssembly)

`@moonshine-ai/moonshine-wasm` runs Moonshine Voice — fast, accurate,
on-device speech-to-text, text-to-speech, and voice interfaces — directly in the
browser via WebAssembly. No audio ever leaves the device.

The API mirrors the Python, Swift, and Android bindings: a thin embind bridge
over the Moonshine C ABI, wrapped in an idiomatic TypeScript layer. The three
entry points are `AgentFlow` for voice interfaces, `MicTranscriber` for live
transcription, and `TextToSpeech` for synthesis and voice cloning. Each is
constructed with `new`, configured with chainable setters, and prepared with a
single `await load()`.

## Install

```bash
npm install @moonshine-ai/moonshine-wasm
```

## Quick start — streaming speech to text

```ts
import { MicTranscriber } from '@moonshine-ai/moonshine-wasm';

const mic = new MicTranscriber()
  .onText((text) => console.log('…', text))
  .onLine((line) => console.log('✓', line.text));

await mic.load();
await mic.start();
// … later …
await mic.stop();
mic.close();
```

### Line identifiers

`TranscriptLine.id` (and `SpeakerSpan.speakerId`) are **decimal strings**, not
numbers. They are 64-bit values allocated as a random base incremented once per
line, so they sit above `Number.MAX_SAFE_INTEGER`, where neighbouring doubles
are 2048 apart — representing them as JS numbers rounded consecutive lines onto
a single id. Treat them as opaque: compare with `===` and use them as keys, but
do not do arithmetic on them.

```ts
const lines = new Map<string, string>();
mic.onLine((line) => {
  lines.set(line.id, line.text);
});
```

### Transcribe a buffer (non-streaming)

```ts
import { Transcriber } from '@moonshine-ai/moonshine-wasm';

const transcriber = await Transcriber.load({ language: 'en' });
const transcript = transcriber.transcribe(float32Pcm, { sampleRate: 16000 });
console.log(transcript.lines.map((l) => l.text).join('\n'));
transcriber.close();
```

## Text to speech

```ts
import { TextToSpeech } from '@moonshine-ai/moonshine-wasm';

const tts = new TextToSpeech();
await tts.load();
await tts.say('Hello from Moonshine.');
tts.close();
```

Cloning is a create-time mode. Call `.cloning()` before `load()` so ZipVoice
and clone ASR download with the synthesizer; then pass a URL, `File`, `Blob`,
or raw PCM to `cloneFrom`:

```ts
const tts = new TextToSpeech().language('en_us').cloning();
await tts.load();
await tts.cloneFrom('some-speech.wav');
await tts.say('Now I sound like the recording.');
```

In the browser, `load()` / `say()` / `cloneFrom()` run synthesis on a Web Worker
so the page stays responsive. `synthesize()` stays on the main thread.

## Dialog flows

`AgentFlow` is the whole voice interface: it downloads the speech, embedding,
and voice models, opens the microphone, matches trigger phrases, and runs the
flow.
Flow bodies are ordinary `async` functions.

```ts
import { AgentFlow } from '@moonshine-ai/moonshine-wasm';

const agent = new AgentFlow();

agent.listenFor('set up wifi', async (d) => {
  const ssid = await d.ask("What's your wifi network?");
  if (await d.confirm(`Connect to ${ssid}?`)) {
    await d.say(`Connecting to ${ssid}.`);
  }
});

await agent.load();
await agent.startListening();
```

"cancel" and "start over" work anywhere inside a flow without being registered.
Outside one they are just words, so an interface that dictates whatever it hears
keeps them; `agent.always('cancel', ...)` makes a phrase live at every moment.
Attach `agent.onHeard(...)` and `agent.onSaid(...)` to log the conversation.

`speech(false)` runs the flow without a voice, which also skips the synthesizer
download. Prompts still reach `onSaid(...)`, so a screen can show what the agent
would have said.

## Models are downloaded at runtime

To keep the library small (well under 100 MB), only the VAD is embedded in the
`.wasm`. Every other model — STT, TTS, G2P, and the text embedding model — is
fetched from the Moonshine CDN (`https://download.moonshine.ai`) the first time
it's needed and cached in the browser via the Cache API. The exact file list and
URLs come from the C ABI manifest helpers, so the JS never hardcodes the layout.

Pass `onProgress` to any `load(...)` call to drive a download UI:

```ts
const mic = new MicTranscriber().onProgress((fraction, file, bytes) => {
  // fraction is 0..1 across the whole model, not the file in flight, so it
  // only ever moves forwards. bytes is { loaded, total } for a byte readout.
  bar.style.width = `${Math.round(100 * fraction)}%`;
});
```

A model is several files, and the manifest declares each one's size, so the
progress reported is the true percentage of the entire download rather than
each file restarting the bar at zero. `bytes.total` is undefined in the one
case where the sizes aren't known up front — files fetched by URL rather than
from a manifest — and `fraction` stays at 0 there, so show an indeterminate
bar when `total` is missing instead of a percentage.

### Self-hosting the model files

If you'd rather host the model files yourself, give `Transcriber.load` the raw
bytes (`{ encoder, decoder, tokenizer }` for a non-streaming model, or a keyed
`{ files }` map for any architecture — including streaming), or point it at your
own URLs and let the binding download and cache them for you:

```ts
const transcriber = await Transcriber.loadFromUrls(
  {
    'encoder_model.ort': '/models/tiny-en/encoder_model.ort',
    'decoder_model_merged.ort': '/models/tiny-en/decoder_model_merged.ort',
    'tokenizer.bin': '/models/tiny-en/tokenizer.bin',
  },
  { modelArch: ModelArch.Base },
);
```

The keys are the canonical manifest filenames (the same ones returned by the CDN
manifest), so streaming models just list their own files (`frontend.ort`,
`encoder.ort`, `adapter.ort`, `cross_kv.ort`, `decoder_kv.ort`,
`streaming_config.json`, `tokenizer.bin`). Everything is loaded purely in memory
— the browser has no natural filesystem — so the same code path serves every
architecture.

## Cross-origin isolation (required for the default build)

The default build enables **SIMD + multithreading** for best performance, which
needs `SharedArrayBuffer`. Browsers only expose that to
[cross-origin-isolated](https://developer.mozilla.org/docs/Web/API/crossOriginIsolated)
pages, so your server must send:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

The example server (`examples/web/serve.mjs`) sets these for you. If you can't
set these headers, build the SIMD-only fallback (see below) and load it with
`-DMOONSHINE_WASM_SINGLE_THREAD=ON`.

## Examples

See [`examples/web/`](../examples/web): `stt/`, `tts/`, `agent-flow/`, and
`dictation/`, with
an index at `/` linking them together. `stt/`, `tts/` and `agent-flow/` show the
Moonshine calls that drive them in a panel below the demo; `dictation/` is a
finished app rather than a walkthrough, so it shows nothing but the app. They
share a stylesheet and some chrome from `examples/web/assets/`, and have no build
step or external dependencies.

The demos import the published binding from the jsDelivr CDN by default. To
test a locally-built binding instead, append `?local=1` to the URL (this loads
`/wasm/dist/index.js`). After building the binding, run the dev server (which
sets the isolation headers) and open a demo:

```bash
scripts/build-wasm.sh
node examples/web/serve.mjs
# → http://localhost:8080/stt/?local=1
```

### When the microphone stays silent

Browsers keep their own notion of the default capture device, and it can
disagree with the operating system's. When it is wrong Chrome does not raise an
error — `getUserMedia` succeeds and the track delivers digital silence, so an
application looks like it is running perfectly and simply never hears anything.

`examples/web/mic-check/` diagnoses this. It reports the permission state, the
track's `muted` flag (set when the OS is withholding audio), raw per-channel peak
and RMS straight from the capture worklet, and whether the `AudioContext` started
suspended. Picking a device by name there saves it, and every demo opens that
device from then on; a saved device that later disappears is dropped so capture
falls back to the default rather than failing forever.

In your own code, name the device through the constraints each entry point takes:

```js
const devices = await navigator.mediaDevices.enumerateDevices();
const mic = devices.find((d) => d.kind === 'audioinput' && d.label.includes('USB'));
const audio = { deviceId: { exact: mic.deviceId } };

new MicTranscriber().audioConstraints(audio);
new AgentFlow().audioConstraints(audio);
voiceClone.fromMicrophone({ audioConstraints: audio });
```

### Making sure you're running your rebuild

When iterating on the binding it's easy to wonder whether the browser is running
the code you just built. Two independent caches are in play:

- **The HTTP cache**, holding the binding itself. The dev server sends
  `Cache-Control: no-store` on every response, so an ordinary reload always
  refetches — no hard reload needed. It also logs each request, so you can watch
  the files come across as the page loads. Note that `/wasm/dist/index.js` is
  only a re-export barrel; the file you edited usually arrives as a sibling such
  as `/wasm/dist/mic-transcriber.js`, so look for that line specifically.
- **The Cache Storage bucket `moonshine-models-v1`**, holding downloaded model
  files. This one is written by the binding's `AssetDownloader` and is
  deliberately independent of the HTTP cache, so it survives every kind of
  reload including Shift+Cmd+R. Append `?fresh=1` to a demo URL to empty it and
  redownload, e.g. `http://localhost:8080/stt/?local=1&fresh=1`.

If a source change still doesn't show up, rebuild before reloading: the pages
serve `wasm/dist`, which is produced by `npm run build` (TypeScript) or
`scripts/build-wasm.sh` (the C++ core).

## Building from source

You need [emsdk](https://emscripten.org/docs/getting_started/downloads.html)
**4.0.8** (the version ONNX Runtime 1.23 pins) activated on your `PATH`.

```bash
# One-time: build + vendor the ORT-wasm static library (SIMD + threads).
scripts/build-ort-wasm.sh            # add `single-thread` for the fallback too

# Build the module + TypeScript layer into wasm/dist.
scripts/build-wasm.sh                # `single-thread` for the SIMD-only build

# Run the tests.
scripts/test-wasm.sh
```

`scripts/build-wasm.sh` accepts `publish-npm` (npm publish) and `upload` (attach
a `dist` tarball to the GitHub release). It never uploads the library to
`download.moonshine.ai` — that CDN hosts model assets only.

### Why we build ONNX Runtime from source

Microsoft doesn't publish a prebuilt ORT-wasm **static** library, and the
`onnxruntime-web` npm package only ships a fully-linked `.wasm` module (no `.a`
to link into our C++ core). `scripts/build-ort-wasm.sh` builds
`libonnxruntime_webassembly.a` from ORT, pinned to the same version as the
native builds for ABI compatibility with the vendored headers.

### The minimal build, and what it costs you

Building from source also lets us cut ORT down to the operators our models
actually use. The archive is a *minimal* build restricted by
`core/third-party/onnxruntime/moonshine-required-operators.config`, which drops
about two thirds of ORT's code from the linked `.wasm`.

Two rules follow, and breaking either one produces a session-creation failure
in the browser rather than a build error:

1. **Models must be ORT-format.** A minimal build cannot parse `.onnx` at all.
   Everything shipped in `core/moonshine-tts/data` and everything the model
   catalog points at is `.ort`; use `scripts/convert-models-to-ort.py` for
   anything new.
2. **The operator config must cover every model.** Adding a model, or changing
   one so it uses a new operator, means regenerating the config and rebuilding
   the archive:

```bash
scripts/generate-ort-op-config.py     # enumerates local + catalog models
scripts/build-ort-wasm.sh force
```

Three kinds of model feed the config, and the third is the one that bites:
models under `core/moonshine-tts/data`, models the native catalog can download,
and models compiled into the library as a C array — now just the Silero VAD in
`core/silero-vad-model-data.h`. An embedded model is neither a file the tree
walk finds nor a URL the catalog lists, so it was silently omitted at first, and
the resulting build failed at runtime on a `Relu` the VAD needed. The generator
now finds them by scanning those generated sources for the `ORTM` file magic,
which means a new embedded model is picked up without editing anything, as long
as it lives in a source listed in `EMBEDDED_SOURCES`.

The two cpp-annote diarization models used to be in that third category and are
now in the second, having become a download
([docs/diarization-models.md](../docs/diarization-models.md)). Their operators
did not change with the move, only where the generator reads them from.

The `check-ort-op-config` ctest guards this. It runs offline against the
bundled and embedded models, so it catches a model added to the tree; the
catalog models are only checked by a full `generate-ort-op-config.py` run,
which downloads several GB the first time and caches them under
`~/.cache/moonshine-ort-op-config`. Run that before vendoring a new archive.

iOS and Android are built the same way now, by `scripts/build-ort-ios.sh` and
`scripts/build-ort-android.sh`, against this same config — so regenerating it
means rebuilding all three, not just the wasm archive. macOS, Linux and Windows
still use the prebuilt full ORT, but they are held to the ORT-only rule anyway
(`session.load_model_format=ORT` is set everywhere), so a model that would fail
in the browser fails on the desktop you develop on. See
[docs/ort-only-models.md](../docs/ort-only-models.md).

One difference worth knowing on mobile: an app is installed for months, while a
model can be pushed to the CDN today. A new model needing an operator that the
shipped app's library lacks fails on that app forever, not until the next page
load. Adding an operator to the config is therefore a change that needs a
client release before the model goes out.

## Versioning

The npm package version tracks the core Moonshine version (see `package.json`
and `python/pyproject.toml`). Keep the ORT pin in `scripts/build-ort-wasm.sh` in
lockstep with `core/third-party/onnxruntime/find-ort-library-path.cmake`.

## License

MIT.
