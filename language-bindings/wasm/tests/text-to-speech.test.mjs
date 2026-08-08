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
  assert.ok(keys.includes('kokoro/model.ort'));
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

test('splitSayUtterances breaks on punct plus space', async () => {
  const { splitSayUtterances } = await importApi();
  assert.deepEqual(splitSayUtterances(''), []);
  assert.deepEqual(splitSayUtterances('Hello'), ['Hello']);
  assert.deepEqual(splitSayUtterances('Hello.'), ['Hello.']);
  assert.deepEqual(splitSayUtterances('Hello. World'), ['Hello.', 'World']);
  assert.deepEqual(splitSayUtterances('Hello! World? Yes.'), [
    'Hello!',
    'World?',
    'Yes.',
  ]);
  assert.deepEqual(splitSayUtterances('3.14 is pi.'), ['3.14 is pi.']);
  assert.deepEqual(splitSayUtterances('Warning: the core is hot.'), [
    'Warning:',
    'the core is hot.',
  ]);
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
