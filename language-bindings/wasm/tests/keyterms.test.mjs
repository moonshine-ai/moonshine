// Runtime key-term biasing through the wasm binding, mirroring the same test in
// every other binding (core/transcriber-test.cpp keyterm-biasing, Swift
// TranscriberTests.testSetKeyterms, Android TranscriberTest.testSetKeyterms):
// terms can be replaced on a live transcriber, and clearing them puts the
// unbiased behaviour back.

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

let api;
let transcriber;
let audio;
let sampleRate;
if (available) {
  api = await importApi();
  // A boost this large overwhelms the acoustics, so a term in the list is forced
  // into the output. That keeps the assertions below about whether biasing
  // reached the decoder at all, rather than about the exact transcript, which is
  // not stable enough to compare.
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

test('setKeyterms biases the decoder, and clearing them turns it off', { skip }, () => {
  // A term that does not occur in the audio, so any appearance of it is
  // unambiguously the biasing, and any absence means it really is off.
  assert.ok(!transcribeCurrent().includes('Kubernetes'));

  transcriber.setKeyterms(['Kubernetes']);
  assert.ok(transcribeCurrent().includes('Kubernetes'));

  transcriber.setKeyterms([]);
  assert.ok(!transcribeCurrent().includes('Kubernetes'));
});

test('a term containing the delimiter is rejected', { skip }, () => {
  assert.throws(
    () => transcriber.setKeyterms(['Kubernetes,Ceph']),
    api.MoonshineInvalidArgumentError,
  );
});

// Keep last: the shared transcriber is no longer needed.
test('close after setting key terms', { skip }, () => {
  assert.doesNotThrow(() => transcriber.close());
});
