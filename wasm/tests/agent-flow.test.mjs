// AgentFlow tests. This covers the conversational core: trigger matching,
// say/ask/confirm/choose, reprompt + max-retries, the built-in cancel/start
// over globals, and custom globals. No STT/TTS is needed — `speakWith` records
// what would have been spoken, and utterances are fed in with
// `handleUtterance`, which is exactly what the microphone does internally.

import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { DIST } from './helpers.mjs';

const { AgentFlow, Dialog, DialogNoMatch, spellOut } = await import(
  path.join(DIST, 'agent-flow.js')
);

function makeFlow() {
  const spoken = [];
  const agent = new AgentFlow().speakWith((text) => void spoken.push(text));
  return { agent, spoken };
}

test('a flow runs ask -> confirm(yes) -> say to completion', async () => {
  const { agent, spoken } = makeFlow();
  let finished = false;
  agent.listenFor('start setup', async (d) => {
    const name = await d.ask('What is your name?');
    d.state.name = name;
    const ok = await d.confirm(`Did you say ${name}?`);
    await d.say(ok ? `Hello, ${name}.` : 'Okay, never mind.');
    finished = true;
  });

  await agent.handleUtterance('start setup');
  assert.equal(agent.isActive, true);
  assert.equal(agent.activeTrigger, 'start setup');
  assert.deepEqual(spoken, ['What is your name?']);

  await agent.handleUtterance('Alice');
  assert.equal(spoken.at(-1), 'Did you say Alice?');

  await agent.handleUtterance('yes');
  assert.ok(spoken.includes('Hello, Alice.'));
  assert.equal(finished, true);
  assert.equal(agent.isActive, false);
});

test('confirm(no) takes the negative branch', async () => {
  const { agent, spoken } = makeFlow();
  agent.listenFor('start setup', async (d) => {
    const ok = await d.confirm('Ready?');
    await d.say(ok ? 'Starting.' : 'Not starting.');
  });
  await agent.handleUtterance('start setup');
  await agent.handleUtterance('nope');
  assert.ok(spoken.includes('Not starting.'));
  assert.equal(agent.isActive, false);
});

test('choose matches an option by its phrase', async () => {
  const { agent, spoken } = makeFlow();
  agent.listenFor('pick color', async (d) => {
    const color = await d.choose('Pick a color', {
      red: ['crimson'],
      blue: ['navy'],
    });
    await d.say(`You chose ${color}.`);
  });
  await agent.handleUtterance('pick color');
  await agent.handleUtterance('I want crimson please');
  assert.ok(spoken.includes('You chose red.'));
});

test('an unrecognized answer reprompts, then gives up past max retries', async () => {
  const { agent, spoken } = makeFlow();
  let caught;
  agent.listenFor('confirm me', async (d) => {
    try {
      const ok = await d.confirm('Yes or no?');
      await d.say(ok ? 'Y' : 'N');
    } catch (err) {
      caught = err;
      throw err;
    }
  });
  await agent.handleUtterance('confirm me');
  // Default confirm maxRetries is 1, so the first miss reprompts.
  await agent.handleUtterance('banana');
  assert.equal(spoken.filter((s) => s.includes('yes or a no')).length, 1);

  await agent.handleUtterance('banana');
  assert.ok(caught instanceof DialogNoMatch);
  assert.equal(agent.isActive, false);
  assert.ok(!spoken.includes('Y') && !spoken.includes('N'));
});

test('a timeout counts as a miss and reprompts', async () => {
  const { agent, spoken } = makeFlow();
  agent.listenFor('start', async (d) => {
    await d.ask('Name?', { timeoutMs: 10, maxRetries: 1 });
  });
  const running = agent.handleUtterance('start');
  await new Promise((resolve) => setTimeout(resolve, 40));
  await agent.handleUtterance('Alice');
  await running;
  assert.equal(spoken.filter((s) => s.includes('Name?')).length, 2);
});

test('the built-in cancel global stops the active flow', async () => {
  const { agent } = makeFlow();
  let cancelled = false;
  agent.listenFor('start setup', async (d) => {
    try {
      await d.ask('Name?');
    } finally {
      cancelled = true;
    }
  });
  await agent.handleUtterance('start setup');
  assert.equal(agent.isActive, true);
  await agent.handleUtterance('cancel');
  assert.equal(cancelled, true);
  assert.equal(agent.isActive, false);
});

test('the built-in start over global restarts the active flow', async () => {
  const { agent, spoken } = makeFlow();
  let starts = 0;
  agent.listenFor('begin', async (d) => {
    starts++;
    await d.ask('Name?');
  });
  await agent.handleUtterance('begin');
  assert.equal(starts, 1);
  await agent.handleUtterance('start over');
  assert.equal(starts, 2);
  assert.equal(agent.isActive, true);
  assert.equal(spoken.filter((s) => s === 'Name?').length, 2);
});

test('a custom global runs without disturbing the flow', async () => {
  const { agent, spoken } = makeFlow();
  agent.listenFor('begin', async (d) => {
    const name = await d.ask('Name?');
    await d.say(`Hi ${name}.`);
  });
  agent.always('what time is it', async (d) => {
    await d.say('Half past three.');
  });

  await agent.handleUtterance('begin');
  await agent.handleUtterance('what time is it');
  assert.ok(spoken.includes('Half past three.'));
  assert.equal(agent.isActive, true);

  await agent.handleUtterance('Alice');
  assert.ok(spoken.includes('Hi Alice.'));
});

test('triggers match case-insensitively as a substring', async () => {
  const { agent } = makeFlow();
  agent.listenFor('start setup', async (d) => {
    await d.ask('Name?');
  });
  await agent.handleUtterance('Could you please START SETUP now');
  assert.equal(agent.isActive, true);
});

test('an utterance matching no trigger is ignored when idle', async () => {
  const { agent, spoken } = makeFlow();
  agent.listenFor('start setup', async (d) => {
    await d.ask('Name?');
  });
  await agent.handleUtterance('the weather is nice today');
  assert.equal(agent.isActive, false);
  assert.deepEqual(spoken, []);
});

test('otherwise sees only the lines nothing else claimed', async () => {
  const { agent, spoken } = makeFlow();
  const leftovers = [];
  agent.listenFor('start setup', async (d) => {
    await d.ask('Name?');
  });
  agent.otherwise((text) => void leftovers.push(text));

  await agent.handleUtterance('the weather is nice today');
  assert.deepEqual(leftovers, ['the weather is nice today']);

  // A trigger phrase belongs to the flow it starts, and the answer that
  // follows belongs to the prompt waiting for it.
  await agent.handleUtterance('start setup');
  await agent.handleUtterance('Alice');
  assert.deepEqual(leftovers, ['the weather is nice today']);
  assert.ok(spoken.includes('Name?'));
});

test('otherwise handlers see utterances in the order they were spoken', async () => {
  const { agent } = makeFlow();
  const leftovers = [];
  agent.otherwise(async (text) => {
    // A slow line must not let the one behind it overtake, or a dictation
    // buffer would end up scrambled.
    await new Promise((resolve) => setTimeout(resolve, text === 'first' ? 20 : 0));
    leftovers.push(text);
  });

  await Promise.all([
    agent.handleUtterance('first'),
    agent.handleUtterance('second'),
  ]);
  assert.deepEqual(leftovers, ['first', 'second']);
});

test('say() speaks outside of any flow', async () => {
  const { agent, spoken } = makeFlow();
  await agent.say('Welcome!');
  assert.deepEqual(spoken, ['Welcome!']);
});

test('onHeard and onSaid see both sides of the conversation', async () => {
  const heard = [];
  const said = [];
  const agent = new AgentFlow()
    .speakWith(() => {})
    .onHeard((text) => heard.push(text))
    .onSaid((text) => said.push(text));
  agent.listenFor('begin', async (d) => {
    await d.say('Hello.');
  });

  await agent.handleUtterance('begin');
  assert.deepEqual(heard, ['begin']);
  assert.deepEqual(said, ['Hello.']);
});

test('configuration setters chain and return the agent', () => {
  const agent = new AgentFlow();
  assert.equal(agent.language('es'), agent);
  assert.equal(agent.voice('kokoro_af_heart'), agent);
  assert.equal(agent.microphone(false), agent);
  assert.equal(agent.triggerThreshold(0.5), agent);
  assert.equal(
    agent.listenFor('x', async () => {}),
    agent,
  );
  assert.equal(
    agent.always('y', async () => {}),
    agent,
  );
});

test('Dialog exposes the trigger phrase and scratch state', () => {
  const runner = new AgentFlow();
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
