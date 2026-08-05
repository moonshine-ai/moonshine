// Speech-clip extraction: the search for a usable voice-cloning reference clip.
// Extract requires a loaded TTS synthesizer handle; these tests build one when
// kokoro assets are vendored locally (same gate as text-to-speech.test.mjs).

import test from 'node:test';
import assert from 'node:assert/strict';
import {
  fileExists,
  importApi,
  loadRawModule,
  flattenTtsDependencyKeys,
  readWav,
  ttsAssetMapOrNull,
  twoCities16kPath,
} from './helpers.mjs';

const mod = await loadRawModule();
const supported = typeof mod.extractSpeechClip === 'function';
const skip = !supported && 'this build has no speech clip support';

const depKeys =
  typeof mod.ttsDependencies === 'function'
    ? flattenTtsDependencyKeys(JSON.parse(mod.ttsDependencies('en_us', '')))
    : [];
const assets =
  typeof mod.TextToSpeech === 'function' ? ttsAssetMapOrNull(depKeys) : null;
const handleSkip = !supported
  ? skip
  : assets === null
    ? 'kokoro TTS assets not vendored locally (download-only)'
    : false;

function rawTtsHandle() {
  const keys = [...assets.keys()];
  const buffers = [...assets.values()];
  const tts = new mod.TextToSpeech('en_us', keys, buffers, [], []);
  return tts.handle();
}

const speechPath = twoCities16kPath();
const speechSkip = handleSkip !== false
  ? handleSkip
  : !fileExists(speechPath) && 'two_cities_16k.wav not vendored';

test('silence yields no clip', { skip: handleSkip }, () => {
  const handle = rawTtsHandle();
  const silence = new Float32Array(16000 * 6);
  const clip = mod.extractSpeechClip(silence, 16000, handle, 4, 2);
  assert.equal(clip.isComplete, false);
  assert.ok(clip.speechDuration < 2);
});

test('a recording shorter than the clip yields no clip', { skip: handleSkip }, () => {
  const handle = rawTtsHandle();
  const short = new Float32Array(16000);
  const clip = mod.extractSpeechClip(short, 16000, handle, 4, 2);
  assert.equal(clip.isComplete, false);
});

test('speech yields a four second clip', { skip: speechSkip }, () => {
  const handle = rawTtsHandle();
  const { audio, sampleRate } = readWav(speechPath);
  const clip = mod.extractSpeechClip(audio, sampleRate, handle, 4, 2);
  assert.equal(clip.isComplete, true);
  assert.ok(clip.audio instanceof Float32Array);
  assert.equal(clip.audio.length, 4 * 16000);
  assert.ok(clip.speechDuration >= 2);
  assert.ok(clip.startTime >= 0);
});

test('the requested clip length is honoured', { skip: speechSkip }, () => {
  const handle = rawTtsHandle();
  const { audio, sampleRate } = readWav(speechPath);
  const clip = mod.extractSpeechClip(audio, sampleRate, handle, 2, 1);
  assert.equal(clip.isComplete, true);
  assert.equal(clip.audio.length, 2 * 16000);
});

test('extractSpeechClip falls back rather than failing on silence', { skip: handleSkip }, async () => {
  const { extractSpeechClip } = await importApi();
  const handle = rawTtsHandle();
  const silence = new Float32Array(16000 * 6);
  const clip = await extractSpeechClip(silence, 16000, {
    module: mod,
    ttsHandle: handle,
  });
  // No speech to find, but the caller explicitly handed us this audio, so they
  // get the best window rather than an exception.
  assert.ok(clip instanceof Float32Array);
  assert.equal(clip.length, 4 * 16000);
});

test('VoiceClone reports readiness as audio is fed in', { skip: speechSkip }, async () => {
  const { VoiceClone } = await importApi();
  const handle = rawTtsHandle();
  const { audio, sampleRate } = readWav(speechPath);
  const clone = new VoiceClone(mod, handle);

  let readyFired = 0;
  clone.onReady(() => readyFired++);
  assert.equal(clone.isReady, false);

  // Feed it in second-long chunks, the way a microphone would.
  const chunk = sampleRate;
  for (let i = 0; i < audio.length && !clone.isReady; i += chunk) {
    clone.addAudio(audio.subarray(i, Math.min(i + chunk, audio.length)), sampleRate);
  }

  assert.equal(clone.isReady, true);
  assert.equal(readyFired, 1);
  assert.equal(clone.audio.length, 4 * 16000);
  assert.equal(clone.sampleRate, 16000);
  // A late onReady listener fires immediately rather than never.
  let late = 0;
  clone.onReady(() => late++);
  assert.equal(late, 1);
});

test('VoiceClone reset discards what it captured', { skip: speechSkip }, async () => {
  const { VoiceClone } = await importApi();
  const handle = rawTtsHandle();
  const { audio, sampleRate } = readWav(speechPath);
  const clone = new VoiceClone(mod, handle);
  clone.addAudio(audio, sampleRate);
  assert.equal(clone.isReady, true);
  clone.reset();
  assert.equal(clone.isReady, false);
  assert.equal(clone.recordedSeconds, 0);
});
