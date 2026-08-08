// Microphone capture helpers: the pure functions between the audio graph and
// the streaming transcriber. Both have failure modes that are silent at
// runtime — the transcriber just receives bad audio and reports nothing — so
// they are worth pinning down here.

import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { DIST } from './helpers.mjs';

const { downmixToMono, resampleTo16k } = await import(
  path.join(DIST, 'mic-transcriber.js')
);

/** Float32 arithmetic, so compare within a tolerance rather than exactly. */
function assertSamples(actual, expected) {
  assert.equal(actual.length, expected.length, 'sample count');
  for (let i = 0; i < expected.length; i++) {
    assert.ok(
      Math.abs(actual[i] - expected[i]) < 1e-6,
      `sample ${i}: expected ~${expected[i]}, got ${actual[i]}`,
    );
  }
}

test('downmixToMono copies a mono input rather than aliasing it', () => {
  const source = new Float32Array([0.1, -0.2, 0.3]);
  const mixed = downmixToMono([source]);
  assertSamples(mixed, [0.1, -0.2, 0.3]);
  source[0] = 1;
  assert.notEqual(mixed[0], 1, 'the worklet reuses its buffers, so this must be a copy');
});

test('downmixToMono averages the channels', () => {
  const mixed = downmixToMono([
    new Float32Array([1, 0, -1]),
    new Float32Array([0, 1, -1]),
  ]);
  assertSamples(mixed, [0.5, 0.5, -1]);
});

// The bug this guards against: some USB headsets and audio interfaces open as
// stereo with the microphone on the right channel and silence on the left.
// Forwarding only channel 0 fed the transcriber pure silence, which looks
// exactly like a working demo that never hears anything.
test('downmixToMono keeps audio that arrives only on the right channel', () => {
  const mixed = downmixToMono([
    new Float32Array([0, 0, 0, 0]),
    new Float32Array([0.8, -0.8, 0.4, -0.4]),
  ]);
  assert.ok(
    [...mixed].some((sample) => Math.abs(sample) > 0.01),
    'a right-channel-only microphone must not be flattened to silence',
  );
  assertSamples(mixed, [0.4, -0.4, 0.2, -0.2]);
});

test('downmixToMono handles more than two channels', () => {
  const mixed = downmixToMono([
    new Float32Array([0, 0]),
    new Float32Array([0, 0]),
    new Float32Array([0.6, 0.9]),
    new Float32Array([0, 0]),
  ]);
  assertSamples(mixed, [0.15, 0.225]);
});

test('resampleTo16k passes 16 kHz audio straight through', () => {
  const input = new Float32Array([0.1, 0.2, 0.3]);
  assert.equal(resampleTo16k(input, 16000), input);
});

test('resampleTo16k downsamples 48 kHz by three', () => {
  const input = new Float32Array(300).map((_, i) => Math.sin(i / 10));
  const output = resampleTo16k(input, 48000);
  assert.equal(output.length, 100);
  // Every third input sample, give or take the linear interpolation.
  assert.ok(Math.abs(output[10] - input[30]) < 1e-6);
});
