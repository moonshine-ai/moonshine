// Text-to-speech tests. The voice/dependency catalog checks mirror Swift's
// testGetVoices / testGetDependencies and Android's zipVoice listing tests and
// always run. Actual synthesis needs the (large, download-only) kokoro model,
// so it is skip-guarded and only runs where those assets are vendored/cached.

import test from 'node:test';
import assert from 'node:assert/strict';
import { importApi, loadRawModule, ttsAssetMapOrNull } from './helpers.mjs';

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
  assert.ok(Array.isArray(deps) && deps.length > 0);
  assert.ok(deps.includes('kokoro/model.onnx'));
  assert.ok(deps.includes('kokoro/config.json'));
  assert.ok(deps.some((k) => k.startsWith('kokoro/voices/')));
});

const depKeys = ttsSupported
  ? JSON.parse(mod.ttsDependencies('en_us', ''))
  : [];
const assets = ttsSupported ? ttsAssetMapOrNull(depKeys) : null;
const synthSkip = !ttsSupported
  ? 'no TTS support in this build'
  : assets === null
    ? 'kokoro TTS assets not vendored locally (download-only)'
    : false;

test('synthesizes non-empty mono PCM for a short phrase', { skip: synthSkip }, async () => {
  const { TextToSpeech } = await importApi();
  const tts = await TextToSpeech.load({ language: 'en_us', assets, module: mod });
  try {
    const { audio, sampleRate } = tts.say('Hello world.');
    assert.ok(audio instanceof Float32Array);
    assert.ok(audio.length > 0);
    assert.ok(sampleRate > 0);
  } finally {
    tts.close();
  }
});

test('longer text produces at least as many samples', { skip: synthSkip }, async () => {
  const { TextToSpeech } = await importApi();
  const tts = await TextToSpeech.load({ language: 'en_us', assets, module: mod });
  try {
    const short = tts.say('Hi.');
    const long = tts.say('Hello there, this is a much longer sentence to speak.');
    assert.ok(long.audio.length >= short.audio.length);
    assert.equal(long.sampleRate, short.sampleRate);
  } finally {
    tts.close();
  }
});
