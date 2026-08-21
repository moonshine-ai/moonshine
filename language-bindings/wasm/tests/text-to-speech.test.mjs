// Text-to-speech tests. The voice/dependency catalog checks mirror Swift's
// testGetVoices / testGetDependencies and Android's zipVoice listing tests and
// always run. Actual synthesis needs the (large, download-only) kokoro model,
// so it is skip-guarded and only runs where those assets are vendored/cached.

import test from 'node:test';
import assert from 'node:assert/strict';
import { importApi, loadRawModule, flattenTtsDependencyKeys, ttsAssetMapOrNull } from './helpers.mjs';

const mod = await loadRawModule();
const ttsSupported =
  typeof mod.TextToSpeech === 'function' && typeof mod.ttsDependencies === 'function';
const skipSurface = !ttsSupported && 'no TTS support in this build';

test('the en_us voice catalog lists kokoro voices', { skip: skipSurface }, () => {
  const catalog = JSON.parse(mod.ttsVoices('en_us'));
  assert.ok(Array.isArray(catalog.en_us));
  const ids = catalog.en_us.map((v) => v.id);
  assert.ok(ids.includes('kokoro_af_heart'), 'expected the default kokoro voice');
  for (const voice of catalog.en_us) {
    assert.equal(typeof voice.id, 'string');
    assert.equal(typeof voice.state, 'string');
  }
});

test('the en_us dependency manifest lists the kokoro model + voice', { skip: skipSurface }, () => {
  const deps = JSON.parse(mod.ttsDependencies('en_us', ''));
  assert.ok(deps && Array.isArray(deps.groups) && deps.groups.length > 0);
  const keys = flattenTtsDependencyKeys(deps);
  assert.ok(keys.includes('kokoro/prosody.model.ort'));
  assert.ok(keys.includes('kokoro/prosody.weights.ort'));
  assert.ok(keys.includes('kokoro/decoder.model.ort'));
  assert.ok(keys.includes('kokoro/decoder.weights.ort'));
  assert.ok(!keys.includes('kokoro/model.model.ort'));
  assert.ok(keys.includes('kokoro/config.json'));
  assert.ok(keys.some((k) => k.startsWith('kokoro/voices/')));
});

const depKeys = ttsSupported
  ? flattenTtsDependencyKeys(JSON.parse(mod.ttsDependencies('en_us', '')))
  : [];
const assets = ttsSupported ? ttsAssetMapOrNull(depKeys) : null;
const synthSkip = !ttsSupported
  ? 'no TTS support in this build'
  : assets === null
    ? 'kokoro TTS assets not vendored locally (download-only)'
    : false;

async function loadLocalTts() {
  const { TextToSpeech } = await importApi();
  return new TextToSpeech()
    .language('en_us')
    .assets(assets)
    .useModule(mod)
    .load();
}

test('synthesizes non-empty mono PCM for a short phrase', { skip: synthSkip }, async () => {
  const tts = await loadLocalTts();
  try {
    const { audio, sampleRate } = tts.synthesize('Hello world.');
    assert.ok(audio instanceof Float32Array);
    assert.ok(audio.length > 0);
    assert.ok(sampleRate > 0);
  } finally {
    tts.close();
  }
});

test('longer text produces at least as many samples', { skip: synthSkip }, async () => {
  const tts = await loadLocalTts();
  try {
    const short = tts.synthesize('Hi.');
    const long = tts.synthesize('Hello there, this is a much longer sentence to speak.');
    assert.ok(long.audio.length >= short.audio.length);
    assert.equal(long.sampleRate, short.sampleRate);
  } finally {
    tts.close();
  }
});

test('configuration setters chain and return the engine', async () => {
  const { TextToSpeech } = await importApi();
  const tts = new TextToSpeech();
  assert.equal(tts.language('en_us'), tts);
  assert.equal(tts.voice('kokoro_af_heart'), tts);
  assert.equal(tts.modelsFrom('https://example.test/tts'), tts);
  assert.equal(tts.cloning(), tts);
  assert.equal(tts.useModule(mod), tts);
  assert.equal(tts.isCloned, false);
});

test('splitSayUtterances splits sentences', { skip: skipSurface }, async () => {
  const { splitSayUtterances } = await importApi();
  const split = (text) => splitSayUtterances(text, { language: 'en_us', module: mod });
  assert.deepEqual(await split(''), []);
  assert.deepEqual(await split('Hello'), ['Hello']);
  assert.deepEqual(await split('Hello.'), ['Hello.']);
  assert.deepEqual(await split('Hello there. World is round'), [
    'Hello there.',
    'World is round',
  ]);
  assert.deepEqual(await split('Hello there! World is round? Yes it is.'), [
    'Hello there!',
    'World is round?',
    'Yes it is.',
  ]);
  assert.deepEqual(await split('3.14 is pi.'), ['3.14 is pi.']);
  assert.deepEqual(await split('Warning: the core is hot.'), [
    'Warning:',
    'the core is hot.',
  ]);
});

test('splitSayUtterances keeps abbreviations together', { skip: skipSurface }, async () => {
  const { splitSayUtterances } = await importApi();
  const split = (text) => splitSayUtterances(text, { language: 'en_us', module: mod });
  assert.deepEqual(await split('Dr. Smith is here now.'), ['Dr. Smith is here now.']);
  assert.deepEqual(await split('J. R. R. Tolkien wrote it.'), [
    'J. R. R. Tolkien wrote it.',
  ]);
  assert.deepEqual(await split('やめて。そこまでだ。'), ['やめて。', 'そこまでだ。']);
});

test('streaming yields chunks that cover the pushed text', { skip: synthSkip }, async () => {
  const tts = await loadLocalTts();
  try {
    // Nothing to do until a sentence completes.
    assert.equal(tts.nextChunk(), undefined);
    tts.pushText('Hello there. ');
    assert.equal(tts.isStreaming, true);
    tts.endInput();
    const chunks = [];
    for await (const chunk of tts.stream()) chunks.push(chunk);
    assert.ok(chunks.length > 0);
    assert.ok(chunks.at(-1).isFinal);
    for (const chunk of chunks) {
      assert.ok(chunk.audio instanceof Float32Array);
      assert.ok(chunk.audio.length > 0);
      assert.ok(chunk.sampleRate > 0);
      assert.equal(chunk.utteranceId, 1);
    }
    // The queue drained, so the synthesizer is idle again.
    assert.equal(tts.isStreaming, false);
  } finally {
    tts.close();
  }
});

test('streaming a whole string delivers the same audio as synthesize', { skip: synthSkip }, async () => {
  const tts = await loadLocalTts();
  try {
    const expected = tts.synthesize('Hello there.');
    let total = 0;
    for await (const chunk of tts.stream('Hello there.')) {
      total += chunk.audio.length;
    }
    assert.equal(total, expected.audio.length);
  } finally {
    tts.close();
  }
});

test('streaming forwards pieces from an async iterable', { skip: synthSkip }, async () => {
  const tts = await loadLocalTts();
  async function* tokens() {
    for (const piece of ['Hello ', 'there', '. ', 'Goodbye.']) {
      await new Promise((resolve) => setTimeout(resolve, 0));
      yield piece;
    }
  }
  try {
    const chunks = [];
    for await (const chunk of tts.stream(tokens())) chunks.push(chunk);
    assert.ok(chunks.length >= 2);
    // Both sentences were spoken, each numbered as its own utterance.
    assert.equal(chunks.at(-1).utteranceId, 2);
    assert.ok(chunks.at(-1).isFinal);
    assert.equal(tts.isStreaming, false);
  } finally {
    tts.close();
  }
});

test('leaving the loop early abandons the rest of the reply', { skip: synthSkip }, async () => {
  const tts = await loadLocalTts();
  try {
    for await (const chunk of tts.stream('Hello there. Goodbye.')) {
      assert.equal(chunk.utteranceId, 1);
      break;
    }
    assert.equal(tts.isStreaming, false);
    // Cancelling leaves the synthesizer usable for the next reply.
    assert.ok(tts.synthesize('Hello again.').audio.length > 0);
  } finally {
    tts.close();
  }
});

test('cancelling mid-reply ends the stream where it stopped', { skip: synthSkip }, async () => {
  const tts = await loadLocalTts();
  try {
    const chunks = [];
    for await (const chunk of tts.stream('Hello there. Goodbye. And again.')) {
      chunks.push(chunk);
      // Barge-in from outside the loop. The reply is not torn down here; the
      // loop finds out on its next pull, the way a worker thread would.
      if (chunks.length === 1) tts.cancelStream();
    }
    assert.equal(chunks.length, 1);
    assert.equal(tts.isStreaming, false);

    // The cancellation was reported to the reply it belonged to, so the next
    // one is unaffected by it.
    const next = [];
    for await (const chunk of tts.stream('Hello again.')) next.push(chunk);
    assert.ok(next.length > 0);
    assert.ok(next.at(-1).isFinal);
  } finally {
    tts.close();
  }
});

test('a reply left half-spoken does not cut the next one short', { skip: synthSkip }, async () => {
  const tts = await loadLocalTts();
  try {
    for await (const chunk of tts.stream('Hello there. Goodbye.')) {
      assert.ok(chunk.audio.length > 0);
      break;
    }
    const next = [];
    for await (const chunk of tts.stream('Hello again.')) next.push(chunk);
    assert.ok(next.length > 0);
    assert.ok(next.at(-1).isFinal);
  } finally {
    tts.close();
  }
});

test('a second reply streams from the same synthesizer', { skip: synthSkip }, async () => {
  const tts = await loadLocalTts();
  async function* tokens() {
    yield 'Goodbye ';
    yield 'now.';
  }
  try {
    const first = [];
    for await (const chunk of tts.stream('Hello there.')) first.push(chunk);
    assert.ok(first.length > 0);
    // Nothing is pushed until the source is awaited, so this also covers
    // iterating before there is any text to synthesize.
    const second = [];
    for await (const chunk of tts.stream(tokens())) second.push(chunk);
    assert.ok(second.length > 0);
    assert.ok(second.at(-1).isFinal);
  } finally {
    tts.close();
  }
});

test('a failing text source stops the reply and rethrows', { skip: synthSkip }, async () => {
  const tts = await loadLocalTts();
  async function* tokens() {
    yield 'Hello there. ';
    throw new Error('the model gave up');
  }
  try {
    await assert.rejects(async () => {
      for await (const chunk of tts.stream(tokens())) assert.ok(chunk.audio);
    }, /the model gave up/);
    assert.equal(tts.isStreaming, false);
  } finally {
    tts.close();
  }
});

test('synthesize during a streamed reply reports it is busy', { skip: synthSkip }, async () => {
  const { MoonshineBusyError } = await importApi();
  const tts = await loadLocalTts();
  try {
    tts.pushText('Hello there. ');
    assert.throws(() => tts.synthesize('Something else.'), MoonshineBusyError);
    tts.cancelStream();
    assert.equal(tts.isStreaming, false);
    assert.ok(tts.synthesize('Something else.').audio.length > 0);
  } finally {
    tts.close();
  }
});

test('synthesizing before load() is a clear error', async () => {
  const { TextToSpeech } = await importApi();
  const tts = new TextToSpeech();
  assert.throws(() => tts.synthesize('Hello'), /load\(\)/);
});

test('startCloning before load() is a clear error', async () => {
  const { TextToSpeech } = await importApi();
  const tts = new TextToSpeech().cloning().useModule(mod);
  assert.throws(() => tts.startCloning(), /load\(\)/);
});

test('cloneFrom without cloning() is a clear error', async () => {
  const { TextToSpeech } = await importApi();
  const tts = new TextToSpeech().useModule(mod);
  await assert.rejects(() => tts.cloneFrom(new Float32Array(16)), /cloning\(\)/);
});

test('voice() and cloning() are mutually exclusive', async () => {
  const { TextToSpeech } = await importApi();
  const tts = new TextToSpeech().voice('kokoro_af_heart').cloning();
  assert.equal(tts.cloning(), tts);
  // cloning cleared the catalog voice; voice() clears cloning.
  tts.voice('kokoro_af_heart');
  assert.throws(() => tts.startCloning(), /cloning\(\)/);
});

const zipvoiceDepKeys = ttsSupported
  ? flattenTtsDependencyKeys(
      JSON.parse(mod.ttsDependencies('en_us', 'zipvoice_american_female')),
    )
  : [];
const zipvoiceAssets = ttsSupported ? ttsAssetMapOrNull(zipvoiceDepKeys) : null;
const cloneLoadSkip = !ttsSupported
  ? 'no TTS support in this build'
  : zipvoiceAssets === null
    ? 'zipvoice TTS assets not vendored locally (download-only)'
    : false;

test('loading in clone mode builds ZipVoice for startCloning', { skip: cloneLoadSkip }, async () => {
  const { TextToSpeech } = await importApi();
  // startCloning needs a synthesizer handle (for extract + owned clone ASR),
  // so load() builds a built-in ZipVoice preset even before a clip exists.
  const tts = new TextToSpeech()
    .language('en_us')
    .cloning()
    .assets(zipvoiceAssets)
    .useModule(mod);
  try {
    await tts.load();
    assert.equal(tts.isCloned, false);
    const clone = tts.startCloning();
    assert.ok(clone);
  } finally {
    tts.close();
  }
});
