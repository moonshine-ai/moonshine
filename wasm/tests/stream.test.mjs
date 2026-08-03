// How often a Stream is willing to enter the engine.
//
// Audio arrives from an AudioWorklet 128 frames at a time, so a page that asks
// for a transcription on every chunk asks some 375 times a second. A pass costs
// far more than that budget even when there is nothing new to say: with
// speakers enabled the engine re-clips every diarization turn onto every line
// before deciding to hold back, which is work that grows for as long as the
// session does. Measured on the tiny model, one such call cost 101us a minute
// into a session and 588us four minutes in, by which point the calls that did
// nothing were burning 19% of real time between them and still climbing. That
// is what puts a live transcript further and further behind the audio, so the
// stream collects audio and makes one pass per update interval instead.

import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import {
  DIST,
  importApi,
  readWav,
  tinyEnAvailable,
  tinyEnBytes,
  twoCities16kPath,
} from './helpers.mjs';

const { Stream, DEFAULT_UPDATE_INTERVAL } = await import(path.join(DIST, 'stream.js'));
const { TranscribeFlags } = await import(path.join(DIST, 'enums.js'));

const RATE = 16000;
/** What an AudioWorklet hands over per render quantum, resampled to 16 kHz. */
const QUANTUM = 43;

/**
 * Stands in for the embind Stream, counting the calls that reach it. Every
 * pass returns one more line so a skipped pass is visible in the result.
 */
function fakeRaw() {
  const raw = {
    passes: 0,
    audio: 0,
    started: 0,
    start() {
      raw.started++;
    },
    stop() {},
    close() {},
    addAudio(audio) {
      raw.audio += audio.length;
    },
    transcribe() {
      raw.passes++;
      return {
        lines: [
          {
            id: String(raw.passes),
            text: `pass ${raw.passes}`,
            startTime: 0,
            duration: 0,
            isComplete: false,
            isUpdated: false,
            isNew: true,
            hasTextChanged: true,
            haveSpeakersChanged: false,
            lastTranscriptionLatencyMs: 0,
            words: [],
            speakerSpans: [],
          },
        ],
      };
    },
  };
  return raw;
}

/** Feeds `seconds` of audio the way a capture callback would, chunk by chunk. */
function feedQuanta(stream, seconds, chunk = QUANTUM) {
  const calls = Math.round((seconds * RATE) / chunk);
  for (let i = 0; i < calls; i++) {
    stream.addAudio(new Float32Array(chunk), RATE);
    stream.transcribe();
  }
  return { calls, seconds: (calls * chunk) / RATE };
}

test('a stream makes one pass per update interval, not one per chunk', () => {
  const raw = fakeRaw();
  const stream = new Stream(raw);
  stream.start();

  const { calls, seconds } = feedQuanta(stream, 30);

  assert.ok(calls > 10000, `the test should ask often to be worth making: ${calls}`);
  assert.equal(raw.audio, seconds * RATE, 'every sample should still reach the engine');
  // One pass per interval. Not exactly: a pass covers the interval plus
  // however far the chunk that tipped it over went past, and that overshoot is
  // not carried, so the count lands a little under the ideal.
  const wanted = seconds / DEFAULT_UPDATE_INTERVAL;
  assert.ok(
    raw.passes > wanted * 0.95 && raw.passes <= wanted + 1,
    `${seconds}s of audio should need about ${wanted} passes, got ${raw.passes}`,
  );
});

test('the interval is measured in audio, not in calls or wall clock', () => {
  const raw = fakeRaw();
  const stream = new Stream(raw);
  stream.start();

  // One chunk holding a whole interval is enough on its own.
  stream.addAudio(new Float32Array(DEFAULT_UPDATE_INTERVAL * RATE), RATE);
  stream.transcribe();
  assert.equal(raw.passes, 1, 'a chunk worth a whole interval should be transcribed at once');

  // Asking again without feeding anything is asking about audio already seen.
  stream.transcribe();
  stream.transcribe();
  assert.equal(raw.passes, 1, 'repeat calls with no new audio should not reach the engine');

  stream.addAudio(new Float32Array(10 * RATE), RATE);
  stream.transcribe();
  assert.equal(raw.passes, 2, 'a long chunk is still one pass, over more audio');
});

test('a held-back call returns the last transcript rather than nothing', () => {
  const raw = fakeRaw();
  const stream = new Stream(raw);
  stream.start();

  assert.deepEqual(stream.transcribe().lines, [], 'nothing has been transcribed yet');

  stream.addAudio(new Float32Array(RATE), RATE);
  const fresh = stream.transcribe();
  assert.equal(fresh.lines[0].text, 'pass 1');

  stream.addAudio(new Float32Array(QUANTUM), RATE);
  assert.deepEqual(
    stream.transcribe(),
    fresh,
    'a caller who asks too soon should get the transcript as it last stood',
  );
});

test('holding back does not repeat events for lines already announced', () => {
  const raw = fakeRaw();
  const stream = new Stream(raw);
  const started = [];
  stream.addListener({ onLineStarted: (event) => started.push(event.line.id) });
  stream.start();

  const { calls } = feedQuanta(stream, 2);
  // One LineStarted per pass, and the thousands of skipped calls announce
  // nothing at all.
  assert.equal(started.length, raw.passes);
  assert.ok(started.length < calls / 100, `${started.length} events for ${calls} calls`);
});

test('ForceUpdate insists, so stopping still flushes the last line', () => {
  const raw = fakeRaw();
  const stream = new Stream(raw);
  stream.start();

  stream.addAudio(new Float32Array(QUANTUM), RATE);
  stream.transcribe();
  assert.equal(raw.passes, 0, 'too little audio to be worth a pass');

  stream.transcribe(TranscribeFlags.ForceUpdate);
  assert.equal(raw.passes, 1, 'a forced call should always reach the engine');

  stream.stop();
  assert.equal(raw.passes, 2, 'stopping should flush whatever was still being said');
});

// The interval is a floor rather than a cadence, because most of what a pass
// costs is not the audio in it. Measured on the tiny model with speakers, 102ms
// of a pass goes on getting started and 269ms on each second of audio it covers,
// so half-second passes spend 204ms of every second on overhead alone. A machine
// that cannot quite afford that does not settle at some fixed distance behind the
// audio: it loses ground on every pass. Replayed as a live session, a machine
// three times slower than the one those numbers came from ended a three-minute
// meeting 82 seconds behind, still delivering lines 80 seconds after the audio
// stopped. Passes that have to earn their keep grow instead, until each covers
// its own cost, which is the batch behaviour the situation calls for.
//
// The clock is the test's to say, so that none of this depends on how long
// anything really takes.
function atMyOwnPace(costs) {
  const real = performance.now.bind(performance);
  let clock = 0;
  let call = 0;
  performance.now = () => clock;
  const raw = fakeRaw();
  const engineTranscribe = raw.transcribe;
  raw.transcribe = (flags) => {
    // Read once before and once after: the second reading is the pass's cost.
    const cost = costs[Math.min(call++, costs.length - 1)];
    const answer = engineTranscribe(flags);
    clock += cost * 1000;
    return answer;
  };
  return {
    raw,
    /** Hands over `seconds` of audio, and lets that much time pass. */
    feed(stream, seconds) {
      const chunks = Math.round((seconds * RATE) / QUANTUM);
      for (let i = 0; i < chunks; i++) {
        stream.addAudio(new Float32Array(QUANTUM), RATE);
        clock += (QUANTUM / RATE) * 1000;
        stream.transcribe();
      }
    },
    restore() {
      performance.now = real;
    },
  };
}

test('a pass has to cover at least as much audio as the last one cost', () => {
  // Every pass takes two seconds, which is four intervals' worth of audio.
  const paced = atMyOwnPace([2]);
  try {
    const stream = new Stream(paced.raw);
    stream.start();

    paced.feed(stream, 0.6);
    assert.equal(paced.raw.passes, 1, 'the first pass has only the floor to clear');

    // The next one must wait for two seconds of audio, not half of one.
    paced.feed(stream, 1.3);
    assert.equal(paced.raw.passes, 1, 'a second pass should not be made on half a second');
    paced.feed(stream, 1);
    assert.equal(paced.raw.passes, 2, 'and should be made once it has two seconds to cover');

    // Which is four times less often than the floor alone would have asked.
    paced.feed(stream, 20);
    assert.ok(
      paced.raw.passes >= 9 && paced.raw.passes <= 13,
      `23s of audio at two seconds a pass should be about 11 passes, got ${paced.raw.passes}`,
    );
  } finally {
    paced.restore();
  }
});

test('a stream with time to spare keeps to the interval', () => {
  // A pass that costs a tenth of the interval is a machine with headroom, and
  // nothing about it should change.
  const paced = atMyOwnPace([DEFAULT_UPDATE_INTERVAL / 10]);
  try {
    const stream = new Stream(paced.raw);
    stream.start();
    paced.feed(stream, 10);
    // Not exactly one per interval: a pass covers the interval plus however far
    // the chunk that tipped it over went past, and that overshoot is not carried.
    const wanted = 10 / DEFAULT_UPDATE_INTERVAL;
    assert.ok(
      paced.raw.passes >= wanted - 2 && paced.raw.passes <= wanted + 1,
      `10s of audio should still be about ${wanted} passes, got ${paced.raw.passes}`,
    );
  } finally {
    paced.restore();
  }
});

test('one freak pass does not leave the transcript silent behind it', () => {
  // A pass that somehow took a minute -- a collection, a laptop lid -- must not
  // hold the next one back for a minute of audio.
  const paced = atMyOwnPace([60, 0.1]);
  try {
    const stream = new Stream(paced.raw);
    stream.start();
    paced.feed(stream, 0.6);
    assert.equal(paced.raw.passes, 1, 'the freak pass itself');

    paced.feed(stream, DEFAULT_UPDATE_INTERVAL * 10 + 0.1);
    assert.equal(paced.raw.passes, 2, 'the wait should be capped, not a minute long');

    // And with the freak behind it, the floor governs again.
    paced.feed(stream, 2);
    assert.ok(paced.raw.passes >= 5, `should be back to the interval, got ${paced.raw.passes}`);
  } finally {
    paced.restore();
  }
});

test('a forced pass is made however long the last one took', () => {
  const paced = atMyOwnPace([5]);
  try {
    const stream = new Stream(paced.raw);
    stream.start();
    paced.feed(stream, 0.6);
    assert.equal(paced.raw.passes, 1);

    stream.addAudio(new Float32Array(QUANTUM), RATE);
    stream.transcribe();
    assert.equal(paced.raw.passes, 1, 'a chunk is not five seconds of audio');
    stream.transcribe(TranscribeFlags.ForceUpdate);
    assert.equal(paced.raw.passes, 2, 'but stopping, or any other insistence, still gets a pass');
  } finally {
    paced.restore();
  }
});

test('an update interval of zero keeps the old pass-per-call behaviour', () => {
  const raw = fakeRaw();
  const stream = new Stream(raw, 0);
  stream.start();
  const { calls } = feedQuanta(stream, 0.2);
  assert.equal(raw.passes, calls);
});

test('restarting a stream does not serve the last session transcript', () => {
  const raw = fakeRaw();
  const stream = new Stream(raw);
  stream.start();
  stream.addAudio(new Float32Array(RATE), RATE);
  assert.equal(stream.transcribe().lines.length, 1);

  stream.start();
  assert.deepEqual(stream.transcribe().lines, [], 'a new session starts empty');
});

// The counting above says the engine is asked less often. This says the words
// are the same when it is: the clip goes in the way a capture callback would
// deliver it, one render quantum at a time.
//
// Four seconds at a time is what a machine with no headroom left ends up doing,
// and the whole point of letting it is that the transcript is no worse for it.
test(
  'a stream hears the whole clip in passes the size a loaded machine would need',
  { skip: tinyEnAvailable() ? false : 'test-assets/tiny-en not found' },
  async () => {
    const api = await importApi();
    const transcriber = await api.Transcriber.load({
      ...tinyEnBytes(),
      modelArch: api.ModelArch.Tiny,
    });
    const { audio, sampleRate } = readWav(twoCities16kPath());

    const heard = async (updateInterval) => {
      const stream = transcriber.createStream({ updateInterval });
      let passes = 0;
      const raw = stream.raw;
      const engine = raw.transcribe.bind(raw);
      raw.transcribe = (flags) => {
        passes++;
        return engine(flags);
      };
      stream.start();
      for (let i = 0; i < audio.length; i += QUANTUM) {
        stream.addAudio(audio.subarray(i, Math.min(i + QUANTUM, audio.length)), sampleRate);
        stream.transcribe();
      }
      stream.stop();
      const text = stream
        .transcribe(api.TranscribeFlags.ForceUpdate)
        .lines.map((l) => l.text)
        .join(' ')
        .toLowerCase();
      stream.close();
      return { text, passes };
    };

    const brisk = await heard(DEFAULT_UPDATE_INTERVAL);
    const batched = await heard(4);
    transcriber.close();

    for (const phrase of ['best of times', 'worst of times']) {
      assert.ok(batched.text.includes(phrase), `expected "${phrase}" in: ${batched.text}`);
    }
    assert.ok(
      batched.passes < brisk.passes / 4,
      `four-second passes should be far fewer: ${batched.passes} against ${brisk.passes}`,
    );
    // Not word for word: where a segment is cut depends on what the engine had
    // heard when it was asked. Close enough that a reader would not notice which
    // of the two they were given.
    const words = (text) => text.split(/\s+/).filter(Boolean).length;
    assert.ok(
      Math.abs(words(batched.text) - words(brisk.text)) <= words(brisk.text) * 0.1,
      `batching should not cost words: ${words(batched.text)} against ${words(brisk.text)}`,
    );
  },
);

test(
  'a stream fed one render quantum at a time still hears the whole clip',
  { skip: tinyEnAvailable() ? false : 'test-assets/tiny-en not found' },
  async () => {
    const api = await importApi();
    const transcriber = await api.Transcriber.load({
      ...tinyEnBytes(),
      modelArch: api.ModelArch.Tiny,
    });
    const { audio, sampleRate } = readWav(twoCities16kPath());
    const stream = transcriber.createStream();

    let passes = 0;
    const raw = stream.raw;
    const engineTranscribe = raw.transcribe.bind(raw);
    raw.transcribe = (flags) => {
      passes++;
      return engineTranscribe(flags);
    };

    stream.start();
    let calls = 0;
    for (let i = 0; i < audio.length; i += QUANTUM) {
      stream.addAudio(audio.subarray(i, Math.min(i + QUANTUM, audio.length)), sampleRate);
      stream.transcribe();
      calls++;
    }
    stream.stop();
    const text = stream.transcribe(api.TranscribeFlags.ForceUpdate)
      .lines.map((l) => l.text)
      .join(' ')
      .toLowerCase();
    stream.close();
    transcriber.close();

    for (const phrase of ['best of times', 'worst of times']) {
      assert.ok(text.includes(phrase), `expected "${phrase}" in: ${text}`);
    }
    const seconds = audio.length / sampleRate;
    assert.ok(
      passes < calls / 100,
      `asked ${calls} times over ${seconds.toFixed(0)}s, should not have made ${passes} passes`,
    );
  },
);
