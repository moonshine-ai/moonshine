// DialogFlow tests. The generator-based runner is unique to this binding (the
// Python/Swift/Android suites only smoke-test dialog plumbing), so this covers
// the conversational core: trigger matching, say/ask/confirm/choose, reprompt +
// max-retries, and the cancel/restart global handlers. No STT/TTS is needed —
// a recording `speakFn` stands in for playback and completed lines are fed in
// directly, exactly how a Stream would drive it.

import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { DIST } from './helpers.mjs';

const { DialogFlow, Dialog, InputMode, spellOut } = await import(
  path.join(DIST, 'dialog-flow.js')
);

function completedLine(text) {
  return { id: 1, text, isComplete: true, words: [], speakerSpans: [] };
}

function makeFlow(options = {}) {
  const spoken = [];
  const df = new DialogFlow({ speakFn: (t) => void spoken.push(t), ...options });
  return { df, spoken };
}

/** Feeds a completed utterance and waits for the serialized async processing. */
async function feed(df, text) {
  df.onLineCompleted({ line: completedLine(text) });
  await df.queue;
}

test('a flow runs ask -> confirm(yes) -> say to completion', async () => {
  const { df, spoken } = makeFlow();
  df.registerFlow('start setup', function* (d) {
    const name = yield d.ask('What is your name?');
    d.state.name = name;
    const ok = yield d.confirm(`Did you say ${name}?`);
    yield d.say(ok ? `Hello, ${name}.` : 'Okay, never mind.');
  });

  await feed(df, 'start setup');
  assert.equal(df.isActive, true);
  assert.equal(df.activeTrigger, 'start setup');
  assert.deepEqual(spoken, ['What is your name?']);

  await feed(df, 'Alice');
  assert.equal(spoken.at(-1), 'Did you say Alice?');

  await feed(df, 'yes');
  assert.ok(spoken.includes('Hello, Alice.'));
  assert.equal(df.isActive, false);
});

test('confirm(no) takes the negative branch', async () => {
  const { df, spoken } = makeFlow();
  df.registerFlow('start setup', function* (d) {
    const ok = yield d.confirm('Ready?');
    yield d.say(ok ? 'Starting.' : 'Cancelled.');
  });
  await feed(df, 'start setup');
  await feed(df, 'no');
  assert.ok(spoken.includes('Cancelled.'));
  assert.equal(df.isActive, false);
});

test('choose matches an option by its phrase', async () => {
  const { df, spoken } = makeFlow();
  df.registerFlow('pick color', function* (d) {
    const color = yield d.choose('Pick a color', {
      red: ['crimson'],
      blue: ['navy'],
    });
    yield d.say(`You chose ${color}.`);
  });
  await feed(df, 'pick color');
  await feed(df, 'I want crimson please');
  assert.ok(spoken.includes('You chose red.'));
});

test('an unrecognized answer reprompts, then aborts past max retries', async () => {
  const { df, spoken } = makeFlow();
  df.registerFlow('confirm me', function* (d) {
    const ok = yield d.confirm('Yes or no?');
    yield d.say(ok ? 'Y' : 'N');
  });
  await feed(df, 'confirm me');
  await feed(df, 'banana'); // 1st miss -> reprompt (default confirm maxRetries=1)
  const repromptCount = spoken.filter((s) => s.includes('yes or a no')).length;
  assert.equal(repromptCount, 1);
  await feed(df, 'banana'); // 2nd miss -> exceeds retries -> flow ends
  assert.equal(df.isActive, false);
  assert.ok(!spoken.includes('Y') && !spoken.includes('N'));
});

test('a global handler can cancel the active flow', async () => {
  const { df } = makeFlow();
  df.registerFlow('start setup', function* (d) {
    yield d.ask('Name?');
  });
  df.registerGlobal('cancel that', (d) => d.cancel());
  await feed(df, 'start setup');
  assert.equal(df.isActive, true);
  await feed(df, 'cancel that');
  assert.equal(df.isActive, false);
});

test('a global handler can restart the active flow', async () => {
  const { df, spoken } = makeFlow();
  let starts = 0;
  df.registerFlow('begin', function* (d) {
    starts++;
    yield d.ask('Name?');
  });
  df.registerGlobal('start over', (d) => d.restart());
  await feed(df, 'begin');
  assert.equal(starts, 1);
  await feed(df, 'start over');
  assert.equal(starts, 2);
  assert.equal(df.isActive, true);
  assert.equal(spoken.filter((s) => s === 'Name?').length, 2);
});

test('triggers match case-insensitively as a substring', async () => {
  const { df } = makeFlow();
  df.registerFlow('start setup', function* (d) {
    yield d.ask('Name?');
  });
  await feed(df, 'Could you please START SETUP now');
  assert.equal(df.isActive, true);
});

test('an utterance matching no trigger is ignored when idle', async () => {
  const { df, spoken } = makeFlow();
  df.registerFlow('start setup', function* (d) {
    yield d.ask('Name?');
  });
  await feed(df, 'the weather is nice today');
  assert.equal(df.isActive, false);
  assert.deepEqual(spoken, []);
});

test('say() speaks outside of any flow', async () => {
  const { df, spoken } = makeFlow();
  await df.say('Welcome!');
  assert.deepEqual(spoken, ['Welcome!']);
});

test('Dialog builds prompt objects with the documented defaults', () => {
  const d = new Dialog('trigger phrase');
  assert.equal(d.triggerPhrase, 'trigger phrase');

  const say = d.say('hi');
  assert.deepEqual(say, { kind: 'say', text: 'hi' });

  const ask = d.ask('Q?');
  assert.equal(ask.kind, 'ask');
  assert.equal(ask.mode, InputMode.Free);
  assert.equal(ask.maxRetries, 2);

  const confirm = d.confirm('Q2?');
  assert.equal(confirm.kind, 'confirm');
  assert.equal(confirm.maxRetries, 1);
  assert.ok(confirm.yesPhrases.includes('yes'));
  assert.ok(confirm.noPhrases.includes('no'));

  const choose = d.choose('Q3?', { a: ['x'] });
  assert.equal(choose.kind, 'choose');
  assert.equal(choose.maxRetries, 2);

  // replayLastPrompt reflects the most recently built prompt text.
  assert.deepEqual(d.replayLastPrompt(), { kind: 'say', text: 'Q3?' });
});

test('spellOut renders a string as space-separated characters', () => {
  assert.equal(spellOut('abc'), 'a b c');
  assert.equal(spellOut(''), '');
});
