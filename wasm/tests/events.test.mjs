// Transcript-event sequencing tests. `diffTranscript` reproduces the C++ event
// model (LineStarted / LineTextChanged / LineUpdated / LineSpeakersChanged /
// LineCompleted) from successive snapshots, which is exactly what the
// Android TranscriberTest asserts against per-event line flags.

import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { DIST } from './helpers.mjs';

const { createDiffState, diffTranscript, dispatchError } = await import(
  path.join(DIST, 'events.js')
);

function line(id, text, flags = {}) {
  return {
    id,
    text,
    startTime: 0,
    duration: 0,
    isComplete: !!flags.isComplete,
    isUpdated: !!flags.isUpdated,
    isNew: !!flags.isNew,
    hasTextChanged: !!flags.hasTextChanged,
    haveSpeakersChanged: !!flags.haveSpeakersChanged,
    lastTranscriptionLatencyMs: 0,
    words: [],
    speakerSpans: [],
  };
}

function recorder() {
  const events = [];
  const listener = {
    onLineStarted: (e) => events.push(['started', e.line.id, e.line.text]),
    onLineUpdated: (e) => events.push(['updated', e.line.id, e.line.text]),
    onLineTextChanged: (e) => events.push(['textChanged', e.line.id, e.line.text]),
    onLineSpeakersChanged: (e) => events.push(['speakersChanged', e.line.id]),
    onLineCompleted: (e) => events.push(['completed', e.line.id, e.line.text]),
    onError: (e) => events.push(['error', e.error.message]),
  };
  return { events, listener };
}

test('a brand new line emits exactly one LineStarted', () => {
  const state = createDiffState();
  const { events, listener } = recorder();
  diffTranscript({ lines: [line(1, 'hello')] }, state, [listener]);
  assert.deepEqual(events, [['started', 1, 'hello']]);
});

test('changing text on an existing line emits LineTextChanged', () => {
  const state = createDiffState();
  const { events, listener } = recorder();
  diffTranscript({ lines: [line(1, 'hello')] }, state, [listener]);
  events.length = 0;
  diffTranscript({ lines: [line(1, 'hello there')] }, state, [listener]);
  assert.deepEqual(events, [['textChanged', 1, 'hello there']]);
});

test('unchanged text does not re-emit textChanged', () => {
  const state = createDiffState();
  const { events, listener } = recorder();
  diffTranscript({ lines: [line(1, 'steady')] }, state, [listener]);
  events.length = 0;
  diffTranscript({ lines: [line(1, 'steady')] }, state, [listener]);
  assert.deepEqual(events, []);
});

test('the isUpdated flag emits LineUpdated', () => {
  const state = createDiffState();
  const { events, listener } = recorder();
  diffTranscript({ lines: [line(1, 'x')] }, state, [listener]);
  events.length = 0;
  diffTranscript({ lines: [line(1, 'x', { isUpdated: true })] }, state, [listener]);
  assert.deepEqual(events, [['updated', 1, 'x']]);
});

test('the haveSpeakersChanged flag emits LineSpeakersChanged', () => {
  const state = createDiffState();
  const { events, listener } = recorder();
  diffTranscript(
    { lines: [line(1, 'x', { haveSpeakersChanged: true })] },
    state,
    [listener],
  );
  assert.ok(events.some((e) => e[0] === 'speakersChanged' && e[1] === 1));
});

test('completion emits LineCompleted exactly once', () => {
  const state = createDiffState();
  const { events, listener } = recorder();
  diffTranscript({ lines: [line(1, 'done', { isComplete: true })] }, state, [
    listener,
  ]);
  // Re-delivering the same completed line must not fire again.
  diffTranscript({ lines: [line(1, 'done', { isComplete: true })] }, state, [
    listener,
  ]);
  const completed = events.filter((e) => e[0] === 'completed');
  assert.equal(completed.length, 1);
});

test('multiple lines each start independently', () => {
  const state = createDiffState();
  const { events, listener } = recorder();
  diffTranscript({ lines: [line(1, 'a'), line(2, 'b')] }, state, [listener]);
  const started = events.filter((e) => e[0] === 'started').map((e) => e[1]);
  assert.deepEqual(started, [1, 2]);
});

test('a throwing listener does not break delivery to others', () => {
  const state = createDiffState();
  const bad = {
    onLineStarted() {
      throw new Error('listener blew up');
    },
  };
  const { events, listener } = recorder();
  assert.doesNotThrow(() =>
    diffTranscript({ lines: [line(1, 'hi')] }, state, [bad, listener]),
  );
  assert.deepEqual(events, [['started', 1, 'hi']]);
});

test('dispatchError routes to onError', () => {
  const { events, listener } = recorder();
  dispatchError([listener], new Error('boom'));
  assert.deepEqual(events, [['error', 'boom']]);
});
