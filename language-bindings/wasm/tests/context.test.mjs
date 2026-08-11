// Picking key terms out of a passage through the wasm binding, mirroring the
// same test in every other binding (core/transcriber-test.cpp keyterm-biasing,
// Swift TranscriberTests.testSetContext, Android TranscriberTest.testSetContext).

import test from 'node:test';
import assert from 'node:assert/strict';
import {
  importApi,
  readWav,
  tinyStreamingEnAvailable,
  tinyStreamingEnFiles,
  twoCities16kPath,
} from './helpers.mjs';

const available = tinyStreamingEnAvailable();
const skip = available ? false : 'test-assets/tiny-streaming-en not found';

// Every word here has a token to itself except "Kubernetes", so the passage
// yields exactly that one term and the assertions can name it. The same fixture
// is used in every binding.
const CONTEXT = 'We will move the rest of the work to Kubernetes this year.';

let api;
let transcriber;
let audio;
let sampleRate;
if (available) {
  api = await importApi();
  // A boost this large overwhelms the acoustics, so a term drawn out of the
  // passage is forced into the output. That keeps the assertions below about
  // whether biasing reached the decoder at all, rather than about the exact
  // transcript, which is not stable enough to compare.
  transcriber = await api.Transcriber.load({
    files: tinyStreamingEnFiles(),
    modelArch: api.ModelArch.TinyStreaming,
    options: { keyterm_boost: '30' },
  });
  ({ audio, sampleRate } = readWav(twoCities16kPath()));
}

function transcribeCurrent() {
  return transcriber
    .transcribe(audio, { sampleRate })
    .lines.map((l) => l.text)
    .join(' ');
}

test('setContext finds the terms in a passage and biases towards them', { skip }, () => {
  // The term does not occur in the audio, so any appearance of it is
  // unambiguously the biasing, and any absence means it really is off.
  assert.ok(!transcribeCurrent().includes('Kubernetes'));

  transcriber.setContext(CONTEXT);
  assert.ok(transcribeCurrent().includes('Kubernetes'));

  transcriber.setContext('');
  assert.ok(!transcribeCurrent().includes('Kubernetes'));
});

test('a cap the passage cannot exceed changes nothing', { skip }, () => {
  transcriber.setContext(CONTEXT, 5);
  assert.ok(transcribeCurrent().includes('Kubernetes'));
});

// Keep last: the shared transcriber is no longer needed.
test('close after setting a context', { skip }, () => {
  assert.doesNotThrow(() => transcriber.close());
});
