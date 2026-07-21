// End-to-end speech-to-text tests on the bundled tiny-en model + two_cities
// clip, mirroring the flagship batch/streaming tests in every other binding
// (Python test_modules, Swift TranscriberTests, Android TranscriberTest). These
// run real ORT inference through the multithreaded wasm module in Node.

import test from 'node:test';
import assert from 'node:assert/strict';
import {
  importApi,
  readWav,
  tinyEnAvailable,
  tinyEnBytes,
  twoCities16kPath,
} from './helpers.mjs';

const available = tinyEnAvailable();
const skip = available ? false : 'test-assets/tiny-en not found';

let api;
let transcriber;
let audio;
let sampleRate;
if (available) {
  api = await importApi();
  transcriber = await api.Transcriber.load({
    ...tinyEnBytes(),
    modelArch: api.ModelArch.Tiny,
  });
  ({ audio, sampleRate } = readWav(twoCities16kPath()));
}

const EXPECTED = ['best of times', 'worst of times'];

function joinedText(transcript) {
  return transcript.lines
    .map((l) => l.text)
    .join(' ')
    .toLowerCase();
}

test('batch transcription recovers the expected phrases', { skip }, () => {
  const transcript = transcriber.transcribe(audio, { sampleRate });
  assert.ok(transcript.lines.length > 0, 'expected at least one line');
  const text = joinedText(transcript);
  for (const phrase of EXPECTED) {
    assert.ok(text.includes(phrase), `transcript should contain "${phrase}"`);
  }
});

test('batch transcript lines carry well-formed timing/text', { skip }, () => {
  const transcript = transcriber.transcribe(audio, { sampleRate });
  for (const line of transcript.lines) {
    assert.equal(typeof line.text, 'string');
    assert.ok(line.text.length > 0);
    assert.ok(Number.isFinite(line.startTime) && line.startTime >= 0);
    assert.ok(Number.isFinite(line.duration) && line.duration >= 0);
    assert.ok(Array.isArray(line.words));
  }
});

test('empty audio yields no transcript lines', { skip }, () => {
  const transcript = transcriber.transcribe(new Float32Array(0), { sampleRate });
  assert.ok(Array.isArray(transcript.lines));
  assert.equal(transcript.lines.length, 0);
});

test('streaming emits line events and recovers the phrases', { skip }, () => {
  let started = 0;
  let completed = 0;
  let textChanged = 0;
  const stream = transcriber.createStream();
  stream.addListener({
    onLineStarted: () => started++,
    onLineCompleted: () => completed++,
    onLineTextChanged: () => textChanged++,
  });
  stream.start();

  const chunk = 1600; // ~0.1 s at 16 kHz
  let pending = 0;
  for (let i = 0; i < audio.length; i += chunk) {
    stream.addAudio(audio.subarray(i, Math.min(i + chunk, audio.length)), sampleRate);
    pending += chunk;
    if (pending >= 8000) {
      // Transcribe roughly twice a second to keep the test quick.
      stream.transcribe();
      pending = 0;
    }
  }
  stream.stop();
  const final = stream.transcribe();

  assert.ok(started > 0, 'expected at least one LineStarted');
  assert.ok(completed > 0, 'expected at least one LineCompleted');
  assert.ok(textChanged > 0, 'expected text to change while streaming');
  // Every completed line should also have started (Android's invariant).
  assert.ok(completed <= started);

  const text = joinedText(final);
  for (const phrase of EXPECTED) {
    assert.ok(text.includes(phrase), `streamed transcript should contain "${phrase}"`);
  }
  stream.close();
});

test('createStream + manual transcribe returns snapshots', { skip }, () => {
  const stream = transcriber.createStream();
  stream.start();
  stream.addAudio(audio.subarray(0, 16000), sampleRate);
  const snapshot = stream.transcribe();
  assert.ok(Array.isArray(snapshot.lines));
  stream.close();
});

test('archName maps enum values to their canonical strings', { skip }, () => {
  assert.equal(transcriber.archName(api.ModelArch.Tiny), 'tiny');
});

// Keep last: closes the shared transcriber and asserts close() is idempotent
// (mirrors Swift's testCloseIdempotent).
test('close is idempotent', { skip }, () => {
  assert.doesNotThrow(() => {
    transcriber.close();
    transcriber.close();
  });
});
