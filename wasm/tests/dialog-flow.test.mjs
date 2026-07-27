// DialogFlow tests. This covers the conversational core: trigger matching,
// say/ask/confirm/choose, reprompt + max-retries, the built-in cancel/start
// over globals, and custom globals. No STT/TTS is needed — `speakWith` records
// what would have been spoken, and utterances are fed in with
// `handleUtterance`, which is exactly what the microphone does internally.

import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { DIST } from './helpers.mjs';

const { DialogFlow, Dialog, DialogNoMatch, spellOut } = await import(
  path.join(DIST, 'dialog-flow.js')
);

function makeFlow() {
  const spoken = [];
  const dialog = new DialogFlow().speakWith((text) => void spoken.push(text));
  return { dialog, spoken };
}

test('a flow runs ask -> confirm(yes) -> say to completion', async () => {
  const { dialog, spoken } = makeFlow();
  let finished = false;
  dialog.listenFor('start setup', async (d) => {
    const name = await d.ask('What is your name?');
    d.state.name = name;
    const ok = await d.confirm(`Did you say ${name}?`);
    await d.say(ok ? `Hello, ${name}.` : 'Okay, never mind.');
    finished = true;
  });

  await dialog.handleUtterance('start setup');
  assert.equal(dialog.isActive, true);
  assert.equal(dialog.activeTrigger, 'start setup');
  assert.deepEqual(spoken, ['What is your name?']);

  await dialog.handleUtterance('Alice');
  assert.equal(spoken.at(-1), 'Did you say Alice?');

  await dialog.handleUtterance('yes');
  assert.ok(spoken.includes('Hello, Alice.'));
  assert.equal(finished, true);
  assert.equal(dialog.isActive, false);
});

test('confirm(no) takes the negative branch', async () => {
  const { dialog, spoken } = makeFlow();
  dialog.listenFor('start setup', async (d) => {
    const ok = await d.confirm('Ready?');
    await d.say(ok ? 'Starting.' : 'Not starting.');
  });
  await dialog.handleUtterance('start setup');
  await dialog.handleUtterance('nope');
  assert.ok(spoken.includes('Not starting.'));
  assert.equal(dialog.isActive, false);
});

test('choose matches an option by its phrase', async () => {
  const { dialog, spoken } = makeFlow();
  dialog.listenFor('pick color', async (d) => {
    const color = await d.choose('Pick a color', {
      red: ['crimson'],
      blue: ['navy'],
    });
    await d.say(`You chose ${color}.`);
  });
  await dialog.handleUtterance('pick color');
  await dialog.handleUtterance('I want crimson please');
  assert.ok(spoken.includes('You chose red.'));
});

test('an unrecognized answer reprompts, then gives up past max retries', async () => {
  const { dialog, spoken } = makeFlow();
  let caught;
  dialog.listenFor('confirm me', async (d) => {
    try {
      const ok = await d.confirm('Yes or no?');
      await d.say(ok ? 'Y' : 'N');
    } catch (err) {
      caught = err;
      throw err;
    }
  });
  await dialog.handleUtterance('confirm me');
  // Default confirm maxRetries is 1, so the first miss reprompts.
  await dialog.handleUtterance('banana');
  assert.equal(spoken.filter((s) => s.includes('yes or a no')).length, 1);

  await dialog.handleUtterance('banana');
  assert.ok(caught instanceof DialogNoMatch);
  assert.equal(dialog.isActive, false);
  assert.ok(!spoken.includes('Y') && !spoken.includes('N'));
});

test('a timeout counts as a miss and reprompts', async () => {
  const { dialog, spoken } = makeFlow();
  dialog.listenFor('start', async (d) => {
    await d.ask('Name?', { timeoutMs: 10, maxRetries: 1 });
  });
  const running = dialog.handleUtterance('start');
  await new Promise((resolve) => setTimeout(resolve, 40));
  await dialog.handleUtterance('Alice');
  await running;
  assert.equal(spoken.filter((s) => s.includes('Name?')).length, 2);
});

test('the built-in cancel global stops the active flow', async () => {
  const { dialog } = makeFlow();
  let cancelled = false;
  dialog.listenFor('start setup', async (d) => {
    try {
      await d.ask('Name?');
    } finally {
      cancelled = true;
    }
  });
  await dialog.handleUtterance('start setup');
  assert.equal(dialog.isActive, true);
  await dialog.handleUtterance('cancel');
  assert.equal(cancelled, true);
  assert.equal(dialog.isActive, false);
});

test('the built-in start over global restarts the active flow', async () => {
  const { dialog, spoken } = makeFlow();
  let starts = 0;
  dialog.listenFor('begin', async (d) => {
    starts++;
    await d.ask('Name?');
  });
  await dialog.handleUtterance('begin');
  assert.equal(starts, 1);
  await dialog.handleUtterance('start over');
  assert.equal(starts, 2);
  assert.equal(dialog.isActive, true);
  assert.equal(spoken.filter((s) => s === 'Name?').length, 2);
});

test('a custom global runs without disturbing the flow', async () => {
  const { dialog, spoken } = makeFlow();
  dialog.listenFor('begin', async (d) => {
    const name = await d.ask('Name?');
    await d.say(`Hi ${name}.`);
  });
  dialog.always('what time is it', async (d) => {
    await d.say('Half past three.');
  });

  await dialog.handleUtterance('begin');
  await dialog.handleUtterance('what time is it');
  assert.ok(spoken.includes('Half past three.'));
  assert.equal(dialog.isActive, true);

  await dialog.handleUtterance('Alice');
  assert.ok(spoken.includes('Hi Alice.'));
});

test('triggers match case-insensitively as a substring', async () => {
  const { dialog } = makeFlow();
  dialog.listenFor('start setup', async (d) => {
    await d.ask('Name?');
  });
  await dialog.handleUtterance('Could you please START SETUP now');
  assert.equal(dialog.isActive, true);
});

test('an utterance matching no trigger is ignored when idle', async () => {
  const { dialog, spoken } = makeFlow();
  dialog.listenFor('start setup', async (d) => {
    await d.ask('Name?');
  });
  await dialog.handleUtterance('the weather is nice today');
  assert.equal(dialog.isActive, false);
  assert.deepEqual(spoken, []);
});

test('say() speaks outside of any flow', async () => {
  const { dialog, spoken } = makeFlow();
  await dialog.say('Welcome!');
  assert.deepEqual(spoken, ['Welcome!']);
});

test('onHeard and onSaid see both sides of the conversation', async () => {
  const heard = [];
  const said = [];
  const dialog = new DialogFlow()
    .speakWith(() => {})
    .onHeard((text) => heard.push(text))
    .onSaid((text) => said.push(text));
  dialog.listenFor('begin', async (d) => {
    await d.say('Hello.');
  });

  await dialog.handleUtterance('begin');
  assert.deepEqual(heard, ['begin']);
  assert.deepEqual(said, ['Hello.']);
});

test('configuration setters chain and return the dialog', () => {
  const dialog = new DialogFlow();
  assert.equal(dialog.language('es'), dialog);
  assert.equal(dialog.voice('kokoro_af_heart'), dialog);
  assert.equal(dialog.microphone(false), dialog);
  assert.equal(dialog.triggerThreshold(0.5), dialog);
  assert.equal(
    dialog.listenFor('x', async () => {}),
    dialog,
  );
  assert.equal(
    dialog.always('y', async () => {}),
    dialog,
  );
});

test('Dialog exposes the trigger phrase and scratch state', () => {
  const runner = new DialogFlow();
  const d = new Dialog(runner, 'trigger phrase');
  assert.equal(d.triggerPhrase, 'trigger phrase');
  assert.deepEqual(d.state, {});
  d.state.count = 1;
  assert.equal(d.state.count, 1);
});

test('spellOut renders a string as space-separated characters', () => {
  assert.equal(spellOut('abc'), 'a b c');
  assert.equal(spellOut(''), '');
});
