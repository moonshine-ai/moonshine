# Moonshine Voice for the Web (WebAssembly)

`@moonshine-ai/moonshine-wasm` runs Moonshine Voice — fast, accurate,
on-device speech-to-text, text-to-speech, and voice intent recognition —
directly in the browser via WebAssembly. No audio ever leaves the device.

The API mirrors the Python, Swift, and Android bindings: a thin embind bridge
over the Moonshine C ABI, wrapped in an idiomatic TypeScript layer
(`Transcriber`, `Stream`, `MicrophoneTranscriber`, `TextToSpeech`,
`GraphemeToPhonemizer`, `IntentRecognizer`, `DialogFlow`).

## Install

```bash
npm install @moonshine-ai/moonshine-wasm
```

## Quick start — streaming speech to text

```ts
import { MicrophoneTranscriber, ModelArch } from '@moonshine-ai/moonshine-wasm';

const mic = await MicrophoneTranscriber.load({
  language: 'en',
  modelArch: ModelArch.BaseStreaming,
  listeners: [
    {
      onLineTextChanged: (e) => console.log('…', e.line.text),
      onLineCompleted: (e) => console.log('✓', e.line.text),
    },
  ],
});
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

const tts = await TextToSpeech.load({ language: 'en' });
await tts.speak('Hello from Moonshine.');
tts.close();
```

## Intent recognition + dialog flows

```ts
import { IntentRecognizer, DialogFlow } from '@moonshine-ai/moonshine-wasm';

const intent = await IntentRecognizer.load();

const runner = new DialogFlow({ intentRecognizer: intent, tts });
runner.registerFlow('set up wifi', function* (d) {
  const ssid = yield d.ask("What's your wifi network?");
  if (yield d.confirm(`Connect to ${ssid}?`)) {
    yield d.say(`Connecting to ${ssid}.`);
  }
});
// Feed completed transcript lines to `runner` (it's a TranscriptEventListener):
mic.addListener(runner);
```

## Models are downloaded at runtime

To keep the library small (well under 100 MB), only the VAD is embedded in the
`.wasm`. Every other model — STT, TTS, G2P, and the intent embedding model — is
fetched from the Moonshine CDN (`https://download.moonshine.ai`) the first time
it's needed and cached in the browser via the Cache API. The exact file list and
URLs come from the C ABI manifest helpers, so the JS never hardcodes the layout.

Pass `onProgress` to any `load(...)` call to drive a download UI.

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
After building the binding, run the dev server (which sets the isolation
headers) and open a demo:

```bash
scripts/build-wasm.sh
node examples/web/serve.mjs
# → http://localhost:8080/stt/
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
