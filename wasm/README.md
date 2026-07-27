# Moonshine Voice for the Web (WebAssembly)

`@moonshine-ai/moonshine-wasm` runs Moonshine Voice — fast, accurate,
on-device speech-to-text, text-to-speech, and voice interfaces — directly in the
browser via WebAssembly. No audio ever leaves the device.

The API mirrors the Python, Swift, and Android bindings: a thin embind bridge
over the Moonshine C ABI, wrapped in an idiomatic TypeScript layer. The three
entry points are `DialogFlow` for voice interfaces, `MicTranscriber` for live
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

Cloning a voice takes one more call. Pass a URL, a `File`, a `Blob`, or raw PCM;
the binding trims it to a few seconds of speech and transcribes that clip for
the vocoder:

```ts
await tts.cloneFrom('some-speech.wav');
await tts.say('Now I sound like the recording.');
```

## Dialog flows

`DialogFlow` is the whole voice interface: it downloads the speech, intent, and
voice models, opens the microphone, matches trigger phrases, and runs the flow.
Flow bodies are ordinary `async` functions.

```ts
import { DialogFlow } from '@moonshine-ai/moonshine-wasm';

const dialog = new DialogFlow();

dialog.listenFor('set up wifi', async (d) => {
  const ssid = await d.ask("What's your wifi network?");
  if (await d.confirm(`Connect to ${ssid}?`)) {
    await d.say(`Connecting to ${ssid}.`);
  }
});

await dialog.load();
await dialog.startListening();
```

"cancel" and "start over" work at any point without being registered. Attach
`dialog.onHeard(...)` and `dialog.onSaid(...)` to log the conversation.

## Models are downloaded at runtime

To keep the library small (well under 100 MB), only the VAD is embedded in the
`.wasm`. Every other model — STT, TTS, G2P, and the intent embedding model — is
fetched from the Moonshine CDN (`https://download.moonshine.ai`) the first time
it's needed and cached in the browser via the Cache API. The exact file list and
URLs come from the C ABI manifest helpers, so the JS never hardcodes the layout.

Pass `onProgress` to any `load(...)` call to drive a download UI.

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

See [`examples/web/`](../examples/web): `stt/`, `tts/`, and `dialog-flow/`.
The demos import the published binding from the jsDelivr CDN by default. To
test a locally-built binding instead, append `?local=1` to the URL (this loads
`/wasm/dist/index.js`). After building the binding, run the dev server (which
sets the isolation headers) and open a demo:

```bash
scripts/build-wasm.sh
node examples/web/serve.mjs
# → http://localhost:8080/stt/?local=1
```

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

## Versioning

The npm package version tracks the core Moonshine version (see `package.json`
and `python/pyproject.toml`). Keep the ORT pin in `scripts/build-ort-wasm.sh` in
lockstep with `core/third-party/onnxruntime/find-ort-library-path.cmake`.

## License

MIT.
