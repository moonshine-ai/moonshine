// Browser integration test for the web examples in examples/web/.
//
// This is the end-to-end counterpart to the (Node-only) unit tests: it starts
// the real static server (examples/web/serve.mjs), opens each example page in a
// headless Chrome via puppeteer, and drives it the way a user would — but using
// the file-based I/O the pages expose so no microphone/speaker is needed. Each
// page is loaded with `?local=1&assets=local`, so it exercises the freshly
// built /wasm/dist binding against the model assets vendored in the repo, with
// no network access at all.
//
// It is opt-in (heavy: launches a browser, runs real ORT + TTS inference) and
// only runs when MOONSHINE_BROWSER_TESTS=1. It also self-skips if puppeteer or
// a Chrome/Chromium binary can't be found. Run it via scripts/test-web-examples.sh.

import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { mkdtemp, readdir, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import {
  REPO_ROOT,
  tryLoadPuppeteer,
  findChrome,
  startExampleServer,
  launchBrowser,
} from './browser-helpers.mjs';

const TWO_CITIES_WAV = path.join(REPO_ROOT, 'test-assets', 'two_cities_16k.wav');

/** The clock format the meeting-notes deck uses, for asserting against it. */
const mmss = (t) => {
  const whole = Math.max(0, Math.floor(t));
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, '0')}`;
};

const enabled = process.env.MOONSHINE_BROWSER_TESTS === '1';
const puppeteer = enabled ? await tryLoadPuppeteer() : null;
const chromePath = enabled ? findChrome() : null;

const skip = !enabled
  ? 'set MOONSHINE_BROWSER_TESTS=1 to run the browser integration test'
  : !puppeteer
    ? 'puppeteer-core is not installed (npm i -D puppeteer-core)'
    : !chromePath
      ? 'no Chrome/Chromium found (set PUPPETEER_EXECUTABLE_PATH)'
      : false;

let server;
let browser;
if (!skip) {
  server = await startExampleServer();
  browser = await launchBrowser(puppeteer, chromePath);
}

test.after(async () => {
  await browser?.close();
  await server?.close();
});

/** Opens a fresh page, forwarding its console + errors to the test output. */
async function openPage(urlPath) {
  const page = await browser.newPage();
  page.on('console', (msg) => console.log(`[page:${urlPath}] ${msg.text()}`));
  page.on('pageerror', (err) => console.log(`[page:${urlPath}] ERROR ${err.message}`));
  await page.goto(`${server.origin}${urlPath}`, { waitUntil: 'load' });
  // The examples rely on SharedArrayBuffer, which needs cross-origin isolation.
  const isolated = await page.evaluate(() => self.crossOriginIsolated === true);
  assert.ok(isolated, 'page should be cross-origin isolated (COOP/COEP headers)');
  return page;
}

test('STT example transcribes an audio file with the local binding', { skip }, async () => {
  const page = await openPage('/stt/?local=1&assets=local');
  try {
    // The page leads with the microphone; file transcription lives in a
    // disclosure, which has to be open before the button can be clicked.
    await page.waitForSelector('#moreOptions');
    await page.evaluate(() => {
      document.getElementById('moreOptions').open = true;
    });

    const input = await page.waitForSelector('#audioFile');
    await input.uploadFile(TWO_CITIES_WAV);
    await page.click('#transcribeFile');

    // Wait until the page marks the transcript complete (model load + decode +
    // inference all happen here, so allow a generous budget).
    await page.waitForSelector('#transcript[data-done="1"]', { timeout: 120000 });

    const transcript = (
      await page.evaluate(() => window.__moonshineFileTranscript || '')
    ).toLowerCase();
    assert.ok(transcript.length > 0, 'expected a non-empty transcript');
    // two_cities_16k.wav says "...best of times ... worst of times...".
    assert.ok(transcript.includes('times'), `unexpected transcript: "${transcript}"`);
  } finally {
    await page.close();
  }
});

test('TTS example synthesizes audio and offers a WAV file', { skip }, async () => {
  // Force the lightweight Kokoro voice (the page defaults to the much larger
  // ZipVoice cloning model, which is impractical to load in CI).
  const page = await openPage('/tts/?local=1&assets=local&voice=kokoro_af_heart');
  try {
    // The page preloads the voice on startup, enabling #speak when ready.
    await page.waitForSelector('#speak:not([disabled])', { timeout: 120000 });
    await page.click('#speak');

    await page.waitForFunction(() => window.__ttsResult && window.__ttsResult.samples > 0, {
      timeout: 120000,
    });

    const result = await page.evaluate(() => window.__ttsResult);
    assert.ok(result.samples > 0, 'expected synthesized samples');
    assert.ok(result.sampleRate > 0, 'expected a positive sample rate');

    // The "download WAV" link should be revealed and point at a blob.
    const downloadShown = await page.evaluate(() => {
      const el = document.getElementById('download');
      return !el.hidden && (el.href || '').startsWith('blob:');
    });
    assert.ok(downloadShown, 'expected a downloadable WAV link after synthesis');
  } finally {
    await page.close();
  }
});

test('Agent-flow example runs a whole conversation from typed input', { skip }, async () => {
  const page = await openPage('/agent-flow/?local=1&assets=local&nomic=1');

  /** Reads the assistant's side of the on-page log. */
  const assistantLines = () =>
    page.evaluate(() =>
      [...document.querySelectorAll('#log .assistant')].map((el) => el.textContent),
    );

  /** Types an utterance and waits for the assistant's next line to match. */
  async function say(text, expected) {
    const before = (await assistantLines()).length;
    await page.type('#utterance', text);
    await page.click('#send');
    await page.waitForFunction(
      (count, pattern) => {
        const lines = [...document.querySelectorAll('#log .assistant')];
        return lines.length > count && new RegExp(pattern, 'i').test(lines[count].textContent);
      },
      { timeout: 120000 },
      before,
      expected,
    );
  }

  try {
    // nomic=1 auto-builds the runner (TTS + embedding model) without a mic.
    await page.waitForFunction(() => window.__dialogReady === true, { timeout: 120000 });

    // Each answer drives the flow one step further; the flow body is a plain
    // async function, so the whole conversation runs to completion.
    await say('set up wifi', 'wifi network');
    await say('home network', 'is that right');
    await say('yes', 'apply these changes');
    await say('yes', 'connecting to');

    const lines = await assistantLines();
    assert.ok(
      lines.some((t) => /home network/i.test(t)),
      `expected the answer to be read back, got: ${JSON.stringify(lines)}`,
    );
  } finally {
    await page.close();
  }
});

test('Dictation app types speech, obeys commands, and exports', { skip }, async () => {
  const page = await openPage('/dictation/?local=1&assets=local&nomic=1');
  const downloads = await captureDownloads(page);
  const document_ = () => page.$eval('#doc', (el) => el.innerText);

  /**
   * Sends a phrase the way the page's own text box does, and waits for the
   * document to settle. Every phrase below changes it: dictation inserts,
   * commands edit, and a command that was mistaken for dictation would show up
   * as its own words on the page.
   */
  async function say(text) {
    const before = await document_();
    await page.type('#utterance', text);
    await page.click('#send');
    await page.waitForFunction(
      (previous) => document.getElementById('doc').innerText !== previous,
      { timeout: 120000 },
      before,
    );
    return document_();
  }

  try {
    // nomic=1 builds the runner without a microphone, so the only model it
    // needs is the one that tells a command from dictation.
    await page.waitForFunction(() => document.body.dataset.state === 'ready', {
      timeout: 120000,
    });

    // An empty document says how to use the app, since the page says nothing
    // else anywhere.
    const placeholder = await page.$eval('#doc', (el) => el.dataset.placeholder);
    assert.match(placeholder, /new line/);
    assert.match(placeholder, /microphone/i);

    const dictated = await say('The quick brown fox jumped over the lazy dog.');
    assert.match(dictated, /The quick brown fox jumped over the lazy dog\./);

    const broken = await say('new line');
    assert.doesNotMatch(broken, /new line/i, 'the command should not be typed');
    // Trailing whitespace, because a contenteditable keeps a spare break at the
    // end of itself to hold the last line open.
    assert.match(broken, /dog\.\n\s*$/, 'expected a line break at the end');

    const second = await say('Then it did it again.');
    assert.match(second, /dog\.\nThen it did it again\./);

    // The three deletes, each taking exactly what it names off the end: the
    // full stop, then the word in front of it, then the whole sentence.
    assert.match(await say('delete character'), /^Then it did it again$/m);
    // A space at the end of a line comes back as a non-breaking one, which is
    // how a contenteditable keeps it from collapsing.
    assert.match(await say('delete word'), /^Then it did it\s*$/m);
    const emptied = await say('delete sentence');
    assert.match(emptied, /dog\.\n\s*$/, 'expected the second line back to nothing');

    // "scratch that" walks the dictation edits back one at a time.
    assert.match(await say('scratch that'), /Then it did it\s*$/m);

    // A sentence is still the last sentence after a line break, so a "new line"
    // in between takes the words with it rather than shielding them.
    await say('new line');
    const acrossBreak = await say('delete sentence');
    assert.doesNotMatch(acrossBreak, /Then it did it/, 'expected the sentence, not just the break');
    assert.match(acrossBreak, /dog\.\s*$/);

    // Phrases that mean what the commands mean but were dictated as prose. The
    // last three sit closest to the threshold the page picked, so they are what
    // fails if someone lowers it.
    for (const line of [
      'Please delete my account and all of my data.',
      'We should start with a new line of business.',
      'A sentence like that would never survive an editor.',
      'Delete it.',
      'Delete that.',
      'We deleted the character we did not need.',
    ]) {
      const typed = await say(line);
      assert.ok(typed.includes(line), `"${line}" should have been typed, got: ${typed}`);
    }

    // Bold applies to a selection, the way the button does it.
    await page.evaluate(() => {
      const doc = document.getElementById('doc');
      const range = document.createRange();
      range.setStart(doc.firstChild, 4);
      range.setEnd(doc.firstChild, 15);
      const selection = getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
    });
    await page.click('[data-mark="bold"]');
    const bolded = await page.$eval('#doc', (el) => el.innerHTML);
    assert.match(bolded, /<b>quick brown<\/b>/);

    // Markdown carries the marks and the line breaks; Word is a real docx, and
    // a stored zip keeps its text where a reader (or a test) can find it.
    await page.click('#markdown');
    const markdown = await downloads.text(/\.md$/);
    assert.match(markdown, /\*\*quick brown\*\*/);
    assert.match(markdown, /The \*\*quick brown\*\* fox jumped over the lazy dog\./);

    await page.click('#word');
    const docx = await downloads.bytes(/\.docx$/);
    assert.equal(docx.subarray(0, 4).toString('latin1'), 'PK\u0003\u0004', 'not a zip');
    const inside = docx.toString('utf8');
    assert.match(inside, /word\/document\.xml/);
    assert.match(inside, /<w:b\/>/, 'the bold run lost its formatting');
    assert.match(inside, /quick brown/);

    // Dictation lands at the caret, and bolding a phrase above left it in the
    // middle of the document, so put it back at the end the way clicking there
    // would.
    await page.evaluate(() => {
      const range = document.createRange();
      range.selectNodeContents(document.getElementById('doc'));
      range.collapse(false);
      const selection = getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
    });

    // A document that outgrows the sheet keeps up with itself, rather than
    // carrying on out of sight below the fold.
    for (let i = 0; i < 10; i++) {
      await say(`Line ${i} of a document long enough to fill the sheet, and then some more.`);
    }
    const scroll = await page.$eval('#doc', (el) => ({
      overflowing: el.scrollHeight > el.clientHeight,
      fromBottom: el.scrollHeight - el.scrollTop - el.clientHeight,
    }));
    assert.ok(scroll.overflowing, 'the document should have outgrown the paper by now');
    assert.ok(
      scroll.fromBottom <= 40,
      `expected the paper to follow the words down, ${scroll.fromBottom}px short of the end`,
    );

    // Nothing above needed a voice, and `speech(false)` means none was fetched:
    // the page runs with assets=local, where the only thing under the base URL
    // it hands the runner is the embedding model, so a voice download would
    // have 404ed and failed the load this test waited on.
  } finally {
    await page.close();
  }
});

test('Dictation app types what it hears out of a microphone', { skip }, async () => {
  // Its own browser, because this one has a wav file wired to the microphone in
  // place of a synthetic tone, and the other pages would start transcribing it.
  const speaking = await launchBrowser(puppeteer, chromePath, [
    `--use-file-for-fake-audio-capture=${TWO_CITIES_WAV}%noloop`,
  ]);
  try {
    const page = await speaking.newPage();
    page.on('pageerror', (err) => console.log(`[page:/dictation/] ERROR ${err.message}`));
    await page.goto(`${server.origin}/dictation/?local=1&assets=local`, { waitUntil: 'load' });
    await page.waitForFunction(() => document.body.dataset.state === 'ready', {
      timeout: 120000,
    });

    // Watch for the in-progress phrase, which only exists between the first
    // word of a phrase and the end of it.
    await page.evaluate(() => {
      window.__sawProvisional = false;
      const doc = document.getElementById('doc');
      new MutationObserver(() => {
        if (doc.querySelector('.prov')) window.__sawProvisional = true;
      }).observe(doc, { childList: true, subtree: true, characterData: true });
    });

    await page.click('#record');
    await page.waitForFunction(() => document.body.dataset.state === 'listening', {
      timeout: 120000,
    });
    await page.waitForFunction(
      () => /\w+\s+\w+/.test(document.getElementById('doc').innerText),
      { timeout: 120000 },
      // A phrase has to end before it is committed, so this waits out a pause
      // in the reading as well as the transcription itself.
    );

    assert.ok(
      await page.evaluate(() => window.__sawProvisional),
      'expected the words to show up as they were spoken, before the phrase ended',
    );
    // The provisional phrase is a preview, so it leaves nothing of its own
    // behind once the words are committed.
    await page.click('#record');
    await page.waitForFunction(() => document.body.dataset.state === 'ready', { timeout: 30000 });
    assert.equal(await page.$eval('#doc', (el) => el.querySelectorAll('.prov').length), 0);
    console.log(`[dictation] heard: ${await page.$eval('#doc', (el) => el.innerText)}`);
  } finally {
    await speaking.close();
  }
});

test('Meeting notes keeps edits attached to the audio they came from', { skip }, async () => {
  // No screen share here: a headless browser will not grant one. The typed-line
  // path puts a line on the page by the same route a transcribed one takes,
  // which is enough to check the document's bookkeeping.
  const page = await openPage('/meeting-notes/?local=1&assets=local&nocapture=1');
  try {
    await page.waitForFunction(() => document.body.dataset.state === 'ready', {
      timeout: 120000,
    });

    const send = async (text) => {
      await page.type('#utterance', text);
      await page.click('#send');
      await page.waitForFunction(
        (want) => document.getElementById('doc').innerText.includes(want),
        { timeout: 10000 },
        text.slice(0, 12),
      );
    };

    await send('we agreed to ship on friday');
    await send('and demo it on monday');

    // Two lines from the same speaker are one turn under one label, so the
    // transcript reads as prose rather than as a list.
    assert.equal(await page.$eval('#doc', (el) => el.querySelectorAll('.turn').length), 1);
    assert.deepEqual(
      await page.$$eval('#doc .who', (els) => els.map((e) => e.textContent)),
      ['You'],
    );

    // Typing inside a line leaves it attached to that line, which is what keeps
    // playback pointing at the right audio after a correction.
    await page.evaluate(() => {
      const span = document.querySelector('#doc .ln');
      const range = document.createRange();
      range.setStart(span.firstChild, 2);
      range.collapse(true);
      const selection = getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
    });
    await page.type('#doc', 'ZZ');
    const owned = await page.$eval('#doc', (el) => {
      const spans = [...el.querySelectorAll('.ln')];
      return {
        insideALine: spans.some((s) => s.textContent.includes('ZZ')),
        // Nothing may be left loose between the spans, or it would belong to no
        // audio at all.
        loose: [...el.querySelectorAll('.turn')].some((turn) =>
          [...turn.childNodes].some((n) => n.nodeType === Node.TEXT_NODE && n.data.trim()),
        ),
      };
    });
    assert.ok(owned.insideALine, 'an edit should stay inside the line it was made in');
    assert.ok(!owned.loose, 'no text should be left outside a line span');

    // The label is a block so that exports break a paragraph at it rather than
    // running it into the first word.
    const exported = await page.evaluate(() => window.__meetingNotes.text());
    assert.match(exported, /^You\n/, `label should head its own line: ${JSON.stringify(exported)}`);

    // Playback and the audio exports stay shut until there is a recording.
    assert.ok(await page.$eval('#play', (el) => el.disabled));
    assert.ok(await page.$eval('#wav', (el) => el.disabled));
  } finally {
    await page.close();
  }
});

test('Meeting notes renames a speaker everywhere at once', { skip }, async () => {
  // Driven through appendLine rather than the typed-line path, because the
  // point is a speaker who talks twice with somebody else in between, and typed
  // lines are all the same person.
  const page = await openPage('/meeting-notes/?local=1&assets=local&nocapture=1');
  try {
    await page.waitForFunction(
      () => window.__meetingNotes && document.body.dataset.state === 'ready',
      { timeout: 120000 },
    );

    await page.evaluate(() => {
      const say = (id, text, speaker, at) =>
        window.__meetingNotes.appendLine({
          id,
          text,
          source: 'meeting',
          speaker,
          startTime: at,
          endTime: at + 2,
        });
      say('a', 'shall we start', 0, 0);
      say('b', 'yes go ahead', 1, 2);
      say('c', 'right then', 0, 4);
    });

    const headings = () => page.$$eval('#doc .who', (els) => els.map((e) => e.textContent));
    assert.deepEqual(await headings(), ['Speaker 1', 'Speaker 2', 'Speaker 1']);

    // Type over a heading the way a reader would. Focus first, then place the
    // selection: focusing afterwards would move the caret and lose it.
    const typeOverFirstHeading = async (name) => {
      await page.focus('#doc');
      await page.evaluate(() => {
        const range = document.createRange();
        range.selectNodeContents(document.querySelector('#doc .who'));
        const selection = getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
      });
      await page.keyboard.type(name);
    };

    await typeOverFirstHeading('Jane Doe');
    assert.deepEqual(
      await headings(),
      ['Jane Doe', 'Speaker 2', 'Jane Doe'],
      'renaming one heading should rename that speaker, and only that speaker',
    );

    // A turn that speaker has not taken yet is named too, which is the half of
    // this that a per-element rename would miss.
    await page.evaluate(() =>
      window.__meetingNotes.appendLine({
        id: 'd',
        text: 'one more thing',
        source: 'meeting',
        speaker: 0,
        startTime: 6,
        endTime: 8,
      }),
    );
    assert.equal(
      (await headings()).at(-1),
      'Jane Doe',
      'a later turn from the same speaker should already know their name',
    );

    // The name is what leaves the page, since that is the point of setting one.
    const exported = await page.evaluate(() => window.__meetingNotes.text());
    assert.match(exported, /Jane Doe/);
    assert.doesNotMatch(exported, /Speaker 1/);

    // Correcting a name carries too, not just replacing one outright.
    await page.keyboard.press('Backspace');
    assert.equal(
      (await headings())[2],
      'Jane Do',
      'a correction should follow the speaker as much as a rename does',
    );

    // Emptying a heading is deleting that label, not renaming the speaker to
    // nothing, so their other turns keep the name. Asserted on the last of
    // them, which is never the one being typed in.
    await typeOverFirstHeading(' ');
    assert.equal(
      await page.$$eval('#doc .who[data-who="meeting:0"]', (els) => els.at(-1).textContent),
      'Jane Do',
      'emptying one heading should not blank the speaker’s other turns',
    );
  } finally {
    await page.close();
  }
});

test('Meeting notes follows diarization when it changes its mind', { skip }, async () => {
  // Speaker spans are revised as more of a voice is heard, including for lines
  // the recognizer has already called complete, so a line can change hands long
  // after it is on the page and the reader has edited it.
  const page = await openPage('/meeting-notes/?local=1&assets=local&nocapture=1');
  try {
    await page.waitForFunction(
      () => window.__meetingNotes && document.body.dataset.state === 'ready',
      { timeout: 120000 },
    );

    await page.evaluate(() => {
      const say = (id, text, speaker, at) =>
        window.__meetingNotes.appendLine({
          id,
          text,
          source: 'meeting',
          speaker,
          startTime: at,
          endTime: at + 2,
        });
      say('a', 'shall we start', 0, 0);
      say('b', 'yes go ahead', 1, 2);
      say('c', 'right then', 0, 4);
    });

    /** Each heading with the lines of the turn beneath it. */
    const shape = () =>
      page.evaluate(() =>
        [...document.querySelectorAll('#doc .who')].map((who) => ({
          name: who.textContent,
          lines: [...who.nextElementSibling.querySelectorAll('.ln')].map(
            (span) => span.dataset.line,
          ),
        })),
      );

    assert.deepEqual(await shape(), [
      { name: 'Speaker 1', lines: ['a'] },
      { name: 'Speaker 2', lines: ['b'] },
      { name: 'Speaker 1', lines: ['c'] },
    ]);

    // Give the reader something to lose: a speaker they have named, and a line
    // they have corrected.
    await page.focus('#doc');
    await page.evaluate(() => {
      const range = document.createRange();
      range.selectNodeContents(document.querySelector('#doc .who'));
      const selection = getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
    });
    await page.keyboard.type('Jane Doe');
    await page.evaluate(() => {
      const text = document.querySelector('#doc .ln[data-line="b"]').firstChild;
      const range = document.createRange();
      range.setStart(text, text.data.length - 1);
      range.collapse(true);
      const selection = getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
    });
    await page.keyboard.type(' indeed');

    // The middle line turns out to have been the first speaker all along, which
    // leaves one turn where there were three.
    await page.evaluate(() => window.__meetingNotes.reassignSpeaker('b', 0));
    assert.deepEqual(
      await shape(),
      [{ name: 'Jane Doe', lines: ['a', 'b', 'c'] }],
      'lines that now share a speaker should share a turn, under the name given to them',
    );
    assert.match(
      await page.$eval('#doc .ln[data-line="b"]', (el) => el.textContent),
      /indeed/,
      'regrouping should move the reader’s line, not rewrite it',
    );

    // The reassignment has to reach the record the captions and playback read,
    // not just the label on the page.
    assert.equal(await page.evaluate(() => window.__meetingNotes.lines.get('b').speaker), 0);

    // And a second change of mind splits the turn back apart.
    await page.evaluate(() => window.__meetingNotes.reassignSpeaker('b', 1));
    assert.deepEqual(await shape(), [
      { name: 'Jane Doe', lines: ['a'] },
      { name: 'Speaker 2', lines: ['b'] },
      { name: 'Jane Doe', lines: ['c'] },
    ]);

    // A line the page has never seen is not a reason to do anything.
    await page.evaluate(() => window.__meetingNotes.reassignSpeaker('meeting-999', 3));
    assert.equal((await shape()).length, 3);
  } finally {
    await page.close();
  }
});

test('Meeting notes keeps the reader’s place while diarization revises a window', { skip }, async () => {
  // Re-clustering runs every couple of seconds over the last two minutes and
  // can move many lines at once. That is often enough, over a long enough
  // meeting, that re-cutting the whole page each time would keep snatching the
  // caret away from whoever is reading it.
  const page = await openPage('/meeting-notes/?local=1&assets=local&nocapture=1');
  try {
    await page.waitForFunction(
      () => window.__meetingNotes && document.body.dataset.state === 'ready',
      { timeout: 120000 },
    );

    await page.evaluate(() => {
      const say = (id, text, speaker, at) =>
        window.__meetingNotes.appendLine({
          id,
          text,
          source: 'meeting',
          speaker,
          startTime: at,
          endTime: at + 2,
        });
      say('a', 'shall we start', 0, 0);
      say('b', 'yes go ahead', 1, 2);
      say('c', 'right then', 0, 4);
      say('d', 'after you', 1, 6);
      // Name the blocks, so it can be told later which of them survived
      // untouched and which were built again from scratch.
      document.querySelectorAll('#doc .turn').forEach((turn, i) => {
        turn.dataset.mark = String(i);
      });
    });

    const shape = () =>
      page.evaluate(() =>
        [...document.querySelectorAll('#doc .turn')].map((turn) => ({
          mark: turn.dataset.mark ?? null,
          lines: [...turn.querySelectorAll('.ln')].map((span) => span.dataset.line),
        })),
      );

    // Park the caret in the very first line, well away from what is about to
    // change.
    await page.focus('#doc');
    await page.evaluate(() => {
      const text = document.querySelector('#doc .ln[data-line="a"]').firstChild;
      const range = document.createRange();
      range.setStart(text, 5);
      range.collapse(true);
      const selection = getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
    });

    const regroups = () => page.evaluate(() => window.__meetingNotes.regroups());
    const before = await regroups();

    // One clustering pass hands over several lines in a row.
    await page.evaluate(() => {
      window.__meetingNotes.reassignSpeaker('c', 1);
      window.__meetingNotes.reassignSpeaker('d', 1);
    });

    assert.equal(
      (await regroups()) - before,
      1,
      'a window of revisions should re-cut the page once, not once per line',
    );

    assert.deepEqual(
      await shape(),
      [
        { mark: '0', lines: ['a'] },
        { mark: '1', lines: ['b', 'c', 'd'] },
      ],
      'the turns that did not change should be the same blocks they were',
    );

    // The caret is the point: it is still exactly where it was left.
    assert.deepEqual(
      await page.evaluate(() => {
        const selection = getSelection();
        return {
          line: selection.anchorNode?.parentElement?.dataset.line ?? null,
          offset: selection.anchorOffset,
        };
      }),
      { line: 'a', offset: 5 },
    );
  } finally {
    await page.close();
  }
});

test('Meeting notes stops following the words once the reader scrolls up', { skip }, async () => {
  // Lines land at the end of the document however far up the reader has gone,
  // so following them down has to give way to a reader who has scrolled away,
  // and pick up again when they come back to the end.
  const page = await openPage('/meeting-notes/?local=1&assets=local&nocapture=1');
  try {
    await page.waitForFunction(
      () => window.__meetingNotes && document.body.dataset.state === 'ready',
      { timeout: 120000 },
    );

    const say = (at) =>
      page.evaluate((i) => {
        window.__meetingNotes.appendLine({
          id: `l${i}`,
          text: `line ${i}, with enough words in it to take up room on the sheet`,
          source: 'meeting',
          speaker: i % 2,
          startTime: i * 2,
          endTime: i * 2 + 2,
        });
      }, at);

    // `gap` is how much of the document is left below the sheet: nothing means
    // the newest words are on screen.
    const sheet = () =>
      page.evaluate(() => {
        const doc = document.getElementById('doc');
        return {
          top: doc.scrollTop,
          gap: doc.scrollHeight - doc.clientHeight - doc.scrollTop,
          line: parseFloat(getComputedStyle(doc).lineHeight),
          following: window.__meetingNotes.following(),
        };
      });

    // A scroll event is delivered during a frame, not when scrollTop is set, so
    // two frames is long enough for the page to have heard about a scroll. The
    // checks below would otherwise be able to pass by reading the state before
    // it was reconsidered.
    const heardTheScroll = () =>
      page.evaluate(
        () => new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done))),
      );

    // Enough of a meeting to overflow the sheet several times over.
    for (let i = 0; i < 30; i++) await say(i);

    const followed = await sheet();
    assert.ok(followed.top > 0, 'a document past the end of the sheet should have scrolled');
    assert.ok(
      followed.gap <= followed.line,
      `following should come to rest at the end, not ${followed.gap}px short of it`,
    );

    // Nudging the page by less than the slack is not a reader leaving the end.
    await page.evaluate((by) => {
      document.getElementById('doc').scrollTop -= by;
    }, followed.line / 2);
    await heardTheScroll();
    assert.ok(
      await page.evaluate(() => window.__meetingNotes.following()),
      'a nudge within the slack should not count as leaving the end',
    );
    await say(30);
    assert.ok(
      (await sheet()).gap < 1,
      'so the next line should still bring the page back to the end',
    );

    // Now the reader goes back up to re-read something. Assigning scrollTop
    // rather than turning a wheel because the page cannot tell the difference:
    // a scrollbar drag, a Page Up and a trackpad flick all arrive as this same
    // event, and a real gesture would only add flake.
    await page.evaluate(() => {
      document.getElementById('doc').scrollTop = 0;
    });
    await page.waitForFunction(() => window.__meetingNotes.following() === false);

    const parked = await sheet();
    for (let i = 31; i < 37; i++) await say(i);
    const held = await sheet();
    assert.equal(
      held.top,
      parked.top,
      'lines arriving should not drag the page away from the reader',
    );
    assert.ok(held.gap > held.line, 'and the meeting should have gone on without them');

    // Coming back to the end is how the reader asks to be carried along again.
    await page.evaluate(() => {
      const doc = document.getElementById('doc');
      doc.scrollTop = doc.scrollHeight;
    });
    await page.waitForFunction(() => window.__meetingNotes.following() === true);

    const rejoined = await sheet();
    await say(37);
    const carried = await sheet();
    assert.ok(
      carried.top > rejoined.top,
      'the page should move on with the words again once the reader is back at the end',
    );
    assert.ok(carried.gap < 1, 'and settle on the newest line');
  } finally {
    await page.close();
  }
});

test('Meeting notes keeps up with the words when diarization re-cuts the page', { skip }, async () => {
  // Lines landing are not the only thing that moves the end of the document.
  // Diarization re-cuts the whole of its two-minute window at a time, and each
  // turn it makes or unmakes is a heading and a block: a window that decides two
  // dozen turns were really one, or the other way about, changes the height of
  // the page by a screenful or more. A page that only follows the words when
  // words arrive sits there a screenful behind until the next line lands, and is
  // then one bad scroll event away from concluding the reader has left and
  // giving up following altogether.
  const page = await openPage('/meeting-notes/?local=1&assets=local&nocapture=1');
  try {
    await page.waitForFunction(
      () => window.__meetingNotes && document.body.dataset.state === 'ready',
      { timeout: 120000 },
    );

    // Alternating speakers, so there are plenty of turns to lose and remake.
    await page.evaluate(() => {
      for (let i = 0; i < 24; i++) {
        window.__meetingNotes.appendLine({
          id: `l${i}`,
          text: `line ${i}, with enough words in it to take up room on the sheet`,
          source: 'meeting',
          speaker: i % 2,
          startTime: i * 2,
          endTime: i * 2 + 2,
        });
      }
    });

    const sheet = () =>
      page.evaluate(() => {
        const doc = document.getElementById('doc');
        return {
          top: doc.scrollTop,
          gap: doc.scrollHeight - doc.clientHeight - doc.scrollTop,
          turns: doc.querySelectorAll('.turn').length,
          following: window.__meetingNotes.following(),
        };
      });

    /** Hands diarization's latest answer to the page and lets it settle. */
    const recut = (speakerOf) =>
      page.evaluate((each) => {
        const who = new Function('i', `return (${each})(i)`);
        for (let i = 0; i < 24; i++) window.__meetingNotes.reassignSpeaker(`l${i}`, who(i));
        return new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done)));
      }, speakerOf.toString());

    assert.ok((await sheet()).gap < 1, 'the page should start out at the end of the meeting');

    // It was one person all along. Two dozen turns collapse into one, which is
    // now taller than the sheet several times over -- and a block that tall is
    // the case that used to throw the reader backwards, because the least
    // scrolling that brings a block that tall into view is what shows the top
    // of it.
    await recut((i) => 0);
    const merged = await sheet();
    assert.equal(merged.turns, 1, 'the whole meeting should have become one turn');
    assert.ok(
      merged.gap < 1,
      `a turn taller than the sheet should still be followed at its end, not ${merged.gap}px short`,
    );
    assert.ok(merged.following, 'and the page should still be following');

    // And then it was two people again, putting a thousand pixels of headings
    // and blocks back on the page at once.
    await recut((i) => i % 2);
    const split = await sheet();
    assert.equal(split.turns, 24, 'the turns should be back');
    assert.ok(split.top > merged.top, 'the page should have moved down with them');
    assert.ok(split.gap < 1, `and come to rest at the end, not ${split.gap}px short of it`);

    // None of which is licence to haul a reader who has gone back to re-read
    // something down to the end again.
    await page.evaluate(() => {
      document.getElementById('doc').scrollTop = 0;
    });
    await page.waitForFunction(() => window.__meetingNotes.following() === false);

    await recut((i) => 0);
    assert.equal((await sheet()).top, 0, 'a re-cut should leave a reader where they are');
  } finally {
    await page.close();
  }
});

test('Meeting notes keeps a double or triple click inside one block', { skip }, async () => {
  // Clicking twice or three times points at one thing: this name, or this
  // sentence. The browser reaches past it in either case -- a double-click takes
  // one word of "Speaker 10", and a triple-click on a name runs on to the words
  // below it while one on the words runs on to the next name -- and an edit
  // across that join is one the page has to refuse, or a heading and a turn
  // merge into each other and the ids tying words to their audio are lost with
  // the spans. So a reader who selects something is left unable to type over it.
  const page = await openPage('/meeting-notes/?local=1&assets=local&nocapture=1');
  try {
    await page.waitForFunction(
      () => window.__meetingNotes && document.body.dataset.state === 'ready',
      { timeout: 120000 },
    );

    await page.evaluate(() => {
      const say = (id, text, speaker, at) =>
        window.__meetingNotes.appendLine({
          id,
          text,
          source: 'meeting',
          speaker,
          startTime: at,
          endTime: at + 2,
        });
      say('a', 'shall we start', 0, 0);
      say('b', 'yes go ahead', 1, 2);
      say('c', 'right then', 0, 4);
    });
    await page.focus('#doc');

    // A real gesture: each press has to arrive as the second or third of a run
    // for the browser to count it as one, which is what selects a word or a
    // block in the first place.
    const clickOn = async (selector, times) => {
      const where = await page.evaluate((sel) => {
        const box = document.querySelector(sel).getBoundingClientRect();
        return { x: box.x + Math.min(20, box.width / 2), y: box.y + box.height / 2 };
      }, selector);
      await page.mouse.move(where.x, where.y);
      for (let press = 1; press <= times; press++) {
        await page.mouse.down({ clickCount: press });
        await page.mouse.up({ clickCount: press });
      }
    };

    // Read without case, because the sheet sets names in small capitals and a
    // selection reads as it is drawn.
    const selected = async () =>
      (await page.evaluate(() => getSelection().toString())).toLowerCase();

    const shape = () =>
      page.evaluate(() => ({
        blocks: [...document.getElementById('doc').children].map((el) => el.className),
        headings: [...document.querySelectorAll('#doc .who')].map((el) => el.textContent),
        ids: [...document.querySelectorAll('#doc .ln')].map((el) => el.dataset.line),
        text: [...document.querySelectorAll('#doc .ln')].map((el) => el.textContent.trim()),
      }));
    const original = await shape();

    await clickOn('#doc .who', 2);
    assert.equal(await selected(), 'speaker 1', 'a double-click should take the whole name');
    await page.keyboard.type('Jane Doe');
    assert.deepEqual(
      (await shape()).headings,
      ['Jane Doe', 'Speaker 2', 'Jane Doe'],
      'so typing replaces it outright, everywhere that speaker appears',
    );

    // Three clicks are the same request, more emphatically put.
    await clickOn('#doc .who', 3);
    assert.equal(await selected(), 'jane doe', 'a triple-click should take the name and no more');
    await page.keyboard.type('Jane');
    const renamed = await shape();
    assert.deepEqual(renamed.headings, ['Jane', 'Speaker 2', 'Jane'], 'and replace it too');
    assert.deepEqual(renamed.ids, original.ids, 'without disturbing the lines');
    assert.deepEqual(renamed.blocks, original.blocks, 'or the shape of the page');

    // The same gesture on the words means the sentence, and must stop short of
    // the name below it.
    await clickOn('#doc .turn', 3);
    assert.equal(await selected(), 'shall we start', 'a triple-click should take the sentence');
    await page.keyboard.type('shall we begin');
    const edited = await shape();
    assert.deepEqual(
      edited.text,
      ['shall we begin', 'yes go ahead', 'right then'],
      'so the sentence can be retyped',
    );
    assert.deepEqual(edited.headings, renamed.headings, 'leaving the names alone');
    assert.deepEqual(edited.blocks, original.blocks, 'and the shape of the page');
  } finally {
    await page.close();
  }
});

test('Meeting notes never heads a turn with the meeting itself', { skip }, async () => {
  // Diarization runs a beat behind the recognizer, so a line often finishes
  // before anybody has been attributed to it. Such a line used to open a turn of
  // its own headed "Meeting", which is not a name and belongs to nobody, and
  // which was replaced a second or two later when the attribution arrived --
  // appearing, to a reader, at random.
  const page = await openPage('/meeting-notes/?local=1&assets=local&nocapture=1');
  try {
    await page.waitForFunction(
      () => window.__meetingNotes && document.body.dataset.state === 'ready',
      { timeout: 120000 },
    );

    const say = (id, speaker, at) =>
      page.evaluate(
        ({ id, speaker, at }) =>
          window.__meetingNotes.appendLine({
            id,
            text: `something said at ${at} seconds`,
            source: 'meeting',
            speaker,
            startTime: at,
            endTime: at + 2,
          }),
        { id, speaker, at },
      );

    const shape = () =>
      page.evaluate(() =>
        [...document.getElementById('doc').children].map((el) =>
          el.className === 'who'
            ? el.textContent
            : [...el.querySelectorAll('.ln')].map((line) => line.dataset.line).join(),
        ),
      );

    // The opening words of the meeting, before diarization has said anything.
    // Whoever speaks first is the first speaker by construction, since the
    // numbers count from the first voice to appear.
    await say('a', null, 0);
    assert.deepEqual(await shape(), ['Speaker 1', 'a'], 'the first words should have a speaker');

    // Somebody else takes over, and is heard about in time.
    await say('b', 1, 2);
    // Then a line lands that diarization has not caught up with, which carries
    // on with whoever was talking rather than opening a turn of its own.
    await say('c', null, 4);
    assert.deepEqual(
      await shape(),
      ['Speaker 1', 'a', 'Speaker 2', 'b,c'],
      'a line nobody is attributed to yet should join the turn in progress',
    );

    // And when diarization does report, the line moves, exactly as any other
    // revision would move it.
    await page.evaluate(async () => {
      window.__meetingNotes.reassignSpeaker('c', 0);
      await Promise.resolve();
    });
    assert.deepEqual(
      await shape(),
      ['Speaker 1', 'a', 'Speaker 2', 'b', 'Speaker 1', 'c'],
      'and be moved to them when they are named',
    );

    // Silence is not an answer. A revision that names nobody must not be taken
    // as a line belonging to nobody, which would head a turn with the meeting
    // all over again.
    await page.evaluate(async () => {
      window.__meetingNotes.reassignSpeaker('c', null);
      await Promise.resolve();
    });
    assert.deepEqual(
      await shape(),
      ['Speaker 1', 'a', 'Speaker 2', 'b', 'Speaker 1', 'c'],
      'a revision naming nobody should be ignored',
    );
  } finally {
    await page.close();
  }
});

test('Meeting notes keeps every line of the transcript on the ruling', { skip }, async () => {
  // The sheet is ruled by a background that repeats every line, so anything
  // between the lines that is not a whole number of rules tall walks the text
  // off the ruling. A speaker heading arrives every few sentences, so the error
  // does not stay a quirk of the first turn: it accumulates down the page until
  // it laps and comes back.
  const page = await openPage('/meeting-notes/?local=1&assets=local&nocapture=1');
  try {
    await page.waitForFunction(
      () => window.__meetingNotes && document.body.dataset.state === 'ready',
      { timeout: 120000 },
    );

    const measured = await page.evaluate(() => {
      // Alternating speakers, so every turn brings a heading with it.
      for (let i = 0; i < 8; i++) {
        window.__meetingNotes.appendLine({
          id: `l${i}`,
          text: `turn ${i}, said with enough words to run on a little`,
          source: 'meeting',
          speaker: i % 2,
          startTime: i * 2,
          endTime: i * 2 + 2,
        });
      }

      const doc = document.getElementById('doc');
      const line = parseFloat(getComputedStyle(doc).lineHeight);
      // The ruling is `background-attachment: local`, so it is positioned from
      // the top of the scrollable content and travels with it. Phases measured
      // against that origin are what the reader sees, at any scroll position.
      const origin = doc.getBoundingClientRect().top - doc.scrollTop;
      const phase = (v) => +((((v - origin) % line) + line) % line).toFixed(2);

      const text = [];
      for (const span of doc.querySelectorAll('.ln')) {
        for (const rect of span.getClientRects()) text.push(phase(rect.bottom));
      }

      // What the headings themselves take up, which is where the drift came
      // from: an inline-block seated against the body text's strut stood 34px
      // tall in a 32px ruling, and a 0.5rem margin on top of that made 40px.
      const headings = [...doc.querySelectorAll('.who')].map((el) => {
        const style = getComputedStyle(el);
        return (
          el.getBoundingClientRect().height +
          parseFloat(style.marginTop) +
          parseFloat(style.marginBottom)
        );
      });

      return { line, text, headings };
    });

    assert.ok(measured.text.length >= 8, 'expected a line of text per turn');
    assert.deepEqual(
      [...new Set(measured.text)],
      [measured.text[0]],
      `every line should sit the same distance from its rule, got ${measured.text.join(', ')}`,
    );

    for (const height of measured.headings) {
      assert.ok(
        Math.abs(height % measured.line) < 0.5,
        `a heading must stand a whole number of rules tall, not ${height}px of ${measured.line}px`,
      );
    }
  } finally {
    await page.close();
  }
});

test('Meeting notes will not let a heading swallow the line below it', { skip }, async () => {
  // A heading is a block, so deleting across the join between it and the words
  // around it merges the two: the name absorbs the line, the line's span is
  // replaced by whatever the browser leaves behind, and the id tying those
  // words to their audio goes with it. The page is then a turn short, and since
  // it is re-cut by pairing each heading with the turn after it, everything
  // past the damage pairs up one out and the labels wander.
  const page = await openPage('/meeting-notes/?local=1&assets=local&nocapture=1');
  try {
    await page.waitForFunction(
      () => window.__meetingNotes && document.body.dataset.state === 'ready',
      { timeout: 120000 },
    );

    await page.evaluate(() => {
      const say = (id, text, speaker, at) =>
        window.__meetingNotes.appendLine({
          id,
          text,
          source: 'meeting',
          speaker,
          startTime: at,
          endTime: at + 2,
        });
      say('a', 'shall we start', 0, 0);
      say('b', 'yes go ahead', 1, 2);
      say('c', 'right then', 0, 4);
    });
    await page.focus('#doc');

    /** Puts the caret at one end of the first node matching a selector. */
    const caretTo = (selector, end) =>
      page.evaluate(
        ({ selector, end }) => {
          const text = document.querySelector(selector).firstChild;
          const range = document.createRange();
          range.setStart(text, end === 'end' ? text.data.length : 0);
          range.collapse(true);
          const selection = getSelection();
          selection.removeAllRanges();
          selection.addRange(range);
        },
        { selector, end },
      );

    const shape = () =>
      page.evaluate(() => {
        const doc = document.getElementById('doc');
        return {
          blocks: [...doc.children].map((el) => el.className),
          ids: [...doc.querySelectorAll('.ln')].map((el) => el.dataset.line),
          headings: [...doc.querySelectorAll('.who')].map((el) => el.textContent),
          text: [...doc.querySelectorAll('.ln')].map((el) => el.textContent),
        };
      });

    const original = await shape();
    assert.deepEqual(original.ids, ['a', 'b', 'c'], 'three turns to start with');

    // Forward-delete at the end of a name reaches for the line below it.
    await caretTo('#doc .who', 'end');
    await page.keyboard.press('Delete');
    assert.deepEqual(await shape(), original, 'Delete at the end of a name should do nothing');

    // Backspace at the start of those words reaches back for the name.
    await caretTo('#doc .ln', 'start');
    await page.keyboard.press('Backspace');
    assert.deepEqual(await shape(), original, 'Backspace at the start of a line should do nothing');

    // The same join, crossed by a selection and then typed over, which is how a
    // reader who drags a little too far renames a speaker.
    await page.evaluate(() => {
      const name = document.querySelector('#doc .who').firstChild;
      const words = document.querySelector('#doc .ln').firstChild;
      const range = document.createRange();
      range.setStart(name, 4);
      range.setEnd(words, 6);
      const selection = getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
    });
    await page.keyboard.type('Jane');
    assert.deepEqual(await shape(), original, 'an edit across the join should be refused whole');

    // None of which may come at the cost of editing the name itself,
    await page.evaluate(() => {
      const range = document.createRange();
      range.selectNodeContents(document.querySelector('#doc .who'));
      const selection = getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
    });
    await page.keyboard.type('Jane');
    const renamed = await shape();
    assert.deepEqual(
      renamed.headings,
      ['Jane', 'Speaker 2', 'Jane'],
      'renaming a speaker should still work, and still reach their other turns',
    );
    assert.deepEqual(renamed.ids, original.ids, 'and leave the lines alone');

    // or of editing the transcript.
    await caretTo('#doc .ln', 'end');
    await page.keyboard.type('indeed ');
    const edited = await shape();
    assert.match(edited.text[0], /indeed/, 'the words should still be editable');
    assert.deepEqual(edited.ids, original.ids, 'without disturbing the lines');
    assert.deepEqual(edited.blocks, original.blocks, 'or the shape of the page');
  } finally {
    await page.close();
  }
});

test('Meeting notes does not relabel a heading out from under a rename', { skip }, async () => {
  const page = await openPage('/meeting-notes/?local=1&assets=local&nocapture=1');
  try {
    await page.waitForFunction(
      () => window.__meetingNotes && document.body.dataset.state === 'ready',
      { timeout: 120000 },
    );

    await page.evaluate(() => {
      const say = (id, text, speaker, at) =>
        window.__meetingNotes.appendLine({
          id,
          text,
          source: 'meeting',
          speaker,
          startTime: at,
          endTime: at + 2,
        });
      say('a', 'shall we start', 0, 0);
      say('b', 'yes go ahead', 1, 2);
    });

    // Halfway through naming the first speaker, with the caret still in the
    // heading.
    await page.focus('#doc');
    await page.evaluate(() => {
      const range = document.createRange();
      range.selectNodeContents(document.querySelector('#doc .who'));
      const selection = getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
    });
    await page.keyboard.type('Jane');

    // Diarization now decides that first line belonged to the other speaker,
    // which would otherwise rewrite the heading being typed into.
    await page.evaluate(() => window.__meetingNotes.reassignSpeaker('a', 1));

    assert.equal(
      await page.$eval('#doc .who', (el) => el.textContent),
      'Jane',
      'a name being typed should not be overwritten mid-word',
    );
    assert.equal(
      await page.evaluate(() => getSelection().anchorNode?.parentElement?.className),
      'who',
      'and the caret should still be in it',
    );

    // Once the reader moves on, the label catches up with the recognizer: both
    // lines are the second speaker now, under the one heading.
    await page.evaluate(() => {
      const text = document.querySelector('#doc .ln[data-line="b"]').firstChild;
      const range = document.createRange();
      range.setStart(text, 0);
      range.collapse(true);
      const selection = getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
    });

    await page.waitForFunction(
      () => document.querySelector('#doc .who')?.textContent === 'Speaker 2',
      { timeout: 5000 },
    );
    assert.deepEqual(
      await page.evaluate(() =>
        [...document.querySelectorAll('#doc .who')].map((who) => ({
          name: who.textContent,
          lines: [...who.nextElementSibling.querySelectorAll('.ln')].map(
            (span) => span.dataset.line,
          ),
        })),
      ),
      [{ name: 'Speaker 2', lines: ['a', 'b'] }],
      'both lines belong to the one speaker the recognizer settled on',
    );
  } finally {
    await page.close();
  }
});

test('Meeting notes transcribes two streams, records them, and plays back', { skip }, async () => {
  // Its own browser, because the microphone is a wav file here. `fakescreen`
  // puts that same microphone where the shared screen's audio would be, which
  // is the only way to reach the two-stream path without a share to grant — and
  // it doubles as the echo case, since both streams then hear the same words.
  const speaking = await launchBrowser(puppeteer, chromePath, [
    `--use-file-for-fake-audio-capture=${TWO_CITIES_WAV}%noloop`,
  ]);
  try {
    const page = await speaking.newPage();
    page.on('pageerror', (err) => console.log(`[page:/meeting-notes/] ERROR ${err.message}`));
    await page.goto(`${server.origin}/meeting-notes/?local=1&assets=local&fakescreen=1`, {
      waitUntil: 'load',
    });
    await page.waitForFunction(() => document.body.dataset.state === 'ready', {
      timeout: 120000,
    });

    await page.click('#record');
    await page.waitForFunction(() => document.body.dataset.state === 'capturing', {
      timeout: 60000,
    });
    await page.waitForFunction(() => document.querySelectorAll('#doc .ln').length >= 2, {
      timeout: 120000,
    });
    // Let the reading run on, so there is enough recording to seek around in.
    await new Promise((resolve) => setTimeout(resolve, 6000));

    await page.click('#record');
    await page.waitForFunction(() => document.body.dataset.state === 'ready', { timeout: 30000 });

    const captured = await page.evaluate(() => ({
      recorded: window.__meetingNotes.recordedSeconds(),
      lines: [...window.__meetingNotes.lines.values()],
      text: document.getElementById('doc').innerText,
    }));
    console.log(`[meeting-notes] heard: ${captured.text.replace(/\n/g, ' | ')}`);

    assert.ok(captured.recorded > 3, `expected a recording, got ${captured.recorded}s`);
    assert.ok(captured.lines.length >= 2, 'expected several transcript lines');

    // Every line has to sit inside the recording it is meant to index, or the
    // highlight lands on the wrong words. This is the check that would catch
    // the resampler losing a fraction of a sample per chunk.
    for (const line of captured.lines) {
      assert.ok(
        line.startTime >= 0 && line.endTime <= captured.recorded + 1,
        `line at ${line.startTime}-${line.endTime}s falls outside a ${captured.recorded}s recording`,
      );
    }

    // Both streams are hearing the same words, so every microphone line is an
    // echo of a meeting line and should have been dropped.
    assert.deepEqual(
      [...new Set(captured.lines.map((l) => l.source))],
      ['meeting'],
      'the microphone duplicates should have been suppressed as echo',
    );

    // Stopping flushes one last line, and it belongs to the turn that was open
    // when it was spoken rather than to a new one under a repeated label.
    const labels = await page.$$eval('#doc .who', (els) => els.map((e) => e.textContent));
    for (const [i, label] of labels.entries()) {
      assert.notEqual(label, labels[i - 1], `"${label}" is labelled twice in a row: ${labels}`);
    }

    assert.ok(await page.$eval('#play', (el) => !el.disabled), 'playback should be available');
    await page.click('#play');
    await page.waitForFunction(
      () => document.querySelectorAll('#doc .ln.is-playing').length > 0,
      { timeout: 15000 },
    );
    assert.equal(await page.evaluate(() => document.body.dataset.playing), '1');
    await page.click('#play');
    assert.equal(await page.evaluate(() => document.body.dataset.playing), '0');
    assert.equal(
      await page.$eval('#doc', (el) => el.querySelectorAll('.ln.is-playing').length),
      0,
      'pausing should take the highlight off',
    );

    // Selecting a line and playing hears that line, not the meeting from the
    // top. The last one, so that starting there is unmistakably a seek.
    const select = (index) =>
      page.evaluate((i) => {
        const spans = [...document.querySelectorAll('#doc .ln')];
        const span = i < 0 ? spans.at(i) : spans[i];
        const range = document.createRange();
        range.selectNodeContents(span);
        const selection = getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
        return window.__meetingNotes.lines.get(span.dataset.line).startTime;
      }, index);

    const lastStart = await select(-1);
    await page.click('#play');
    await new Promise((resolve) => setTimeout(resolve, 400));
    const seeked = await page.evaluate(() => window.__meetingNotes.playhead());
    assert.ok(
      seeked >= lastStart - 0.1,
      `playing a selection should start at it: ${seeked}s for a line at ${lastStart}s`,
    );

    // Pausing and playing again with the same selection carries on from where
    // it stopped rather than starting the line over.
    await page.click('#play');
    const stoppedAt = await page.evaluate(() => window.__meetingNotes.playhead());
    await page.click('#play');
    await new Promise((resolve) => setTimeout(resolve, 200));
    const resumed = await page.evaluate(() => window.__meetingNotes.playhead());
    assert.ok(
      resumed >= stoppedAt - 0.05,
      `resume should carry on from ${stoppedAt}s, not restart at ${resumed}s`,
    );
    await page.click('#play');

    // Choosing something else is a new request, so the remembered position goes
    // with the selection it belonged to.
    const firstStart = await select(0);
    await page.click('#play');
    await new Promise((resolve) => setTimeout(resolve, 300));
    const restarted = await page.evaluate(() => window.__meetingNotes.playhead());
    assert.ok(
      restarted < stoppedAt,
      `a new selection should start at ${firstStart}s, not carry on from ${stoppedAt}s`,
    );
    await page.click('#play');

    // A caret with nothing selected is the common case: you are reading, you
    // reach a word you did not catch, and you want to hear it from there.
    const putCaretIn = (index) =>
      page.evaluate((i) => {
        const spans = [...document.querySelectorAll('#doc .ln')];
        const span = i < 0 ? spans.at(i) : spans[i];
        const range = document.createRange();
        range.setStart(span.firstChild, 0);
        range.collapse(true);
        const selection = getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
        return window.__meetingNotes.lines.get(span.dataset.line).startTime;
      }, index);

    /**
     * Waits for the page to take a caret move on board, which it hears about
     * through `selectionchange` and so does in the browser's own time. The
     * clock is where it says so, and until it has, the page is still answering
     * about the caret from before.
     */
    const clockToRead = (want) =>
      page
        .waitForFunction(
          (text) => document.getElementById('clock').textContent.split(' / ')[0] === text,
          { timeout: 5000 },
          want,
        )
        .catch(async () => {
          const shown = await page.$eval('#clock', (el) => el.textContent);
          assert.fail(
            `moving the caret should move the timecode to ${want}, but it reads ${shown}`,
          );
        });

    const spanCount = (await page.$$('#doc .ln')).length;
    const caretLine = await putCaretIn(Math.floor(spanCount / 2));
    await clockToRead(mmss(caretLine));
    const planned = await page.evaluate(() => window.__meetingNotes.plannedStart());
    assert.ok(
      Math.abs(planned - caretLine) < 0.05,
      `a caret should plan to start at its own line: ${planned}s for a line at ${caretLine}s`,
    );

    await page.click('#play');
    await new Promise((resolve) => setTimeout(resolve, 300));
    const fromCaret = await page.evaluate(() => window.__meetingNotes.playhead());
    assert.ok(
      fromCaret >= caretLine - 0.1,
      `a caret should play from itself: ${fromCaret}s for a line at ${caretLine}s`,
    );

    // Left where it stopped, the plan is to carry on rather than to start over.
    await page.click('#play');
    const heldAt = await page.evaluate(() => window.__meetingNotes.playhead());
    assert.ok(
      Math.abs((await page.evaluate(() => window.__meetingNotes.plannedStart())) - heldAt) < 0.05,
      `an untouched caret should plan to resume from ${heldAt}s`,
    );

    // At the very end there is nothing left to play, so it means the lot.
    await page.evaluate(() => {
      const range = document.createRange();
      range.selectNodeContents([...document.querySelectorAll('#doc .ln')].at(-1));
      range.collapse(false);
      const selection = getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
    });
    const earliest = await page.evaluate(() =>
      Math.min(...[...window.__meetingNotes.lines.values()].map((line) => line.startTime)),
    );
    await clockToRead(mmss(earliest));
    const fromEnd = await page.evaluate(() => window.__meetingNotes.plannedStart());
    assert.ok(
      Math.abs(fromEnd - earliest) < 0.05,
      `a caret at the end should start from the beginning: got ${fromEnd}s, wanted ${earliest}s`,
    );
  } finally {
    await speaking.close();
  }
});

test('Meeting notes keeps both speakers when echo suppression is off', { skip }, async () => {
  const speaking = await launchBrowser(puppeteer, chromePath, [
    `--use-file-for-fake-audio-capture=${TWO_CITIES_WAV}%noloop`,
  ]);
  try {
    const page = await speaking.newPage();
    page.on('pageerror', (err) => console.log(`[page:/meeting-notes/] ERROR ${err.message}`));
    await page.goto(`${server.origin}/meeting-notes/?local=1&assets=local&fakescreen=1`, {
      waitUntil: 'load',
    });
    await page.waitForFunction(() => document.body.dataset.state === 'ready', {
      timeout: 120000,
    });
    await page.evaluate(() => {
      document.getElementById('optEcho').checked = false;
    });

    await page.click('#record');
    await page.waitForFunction(() => document.body.dataset.state === 'capturing', {
      timeout: 60000,
    });
    await page.waitForFunction(
      () => {
        const sources = new Set(
          [...window.__meetingNotes.lines.values()].map((line) => line.source),
        );
        return sources.has('you') && sources.has('meeting');
      },
      { timeout: 120000 },
      // Without the filter the same speech arrives twice, once per stream,
      // which is exactly the duplication the filter exists to remove.
    );
    await page.click('#record');
  } finally {
    await speaking.close();
  }
});

test('Dictation app pauses and carries on by voice', { skip }, async () => {
  const page = await openPage('/dictation/?local=1&assets=local');
  const state = () => page.evaluate(() => document.body.dataset.state);
  const waitForState = (want) =>
    page.waitForFunction((s) => document.body.dataset.state === s, { timeout: 120000 }, want);

  /** Hands the runner a phrase without going through the microphone. */
  const say = (text) =>
    page.evaluate(async (t) => {
      document.getElementById('utterance').value = t;
      document.getElementById('send').click();
    }, text);

  try {
    await waitForState('ready');
    await page.click('#record');
    await waitForState('listening');

    // Pausing keeps the session open, because a runner that stopped listening
    // could not hear itself be started again.
    await say('stop dictation');
    await waitForState('paused');

    await say('start dictation');
    await waitForState('listening');

    // Whatever the synthetic microphone was making, nothing was typed while the
    // app was paused.
    await page.click('#record');
    await waitForState('ready');
    assert.equal(await state(), 'ready');
  } finally {
    await page.close();
  }
});

/**
 * Points the page's downloads at a temporary directory and hands back readers
 * for whatever lands there. Real downloads rather than a hook in the page, so
 * the test sees exactly the bytes a reader would end up with.
 */
async function captureDownloads(page) {
  const directory = await mkdtemp(path.join(tmpdir(), 'moonshine-downloads-'));
  const client = await page.createCDPSession();
  await client.send('Browser.setDownloadBehavior', {
    behavior: 'allow',
    downloadPath: directory,
    eventsEnabled: true,
  });

  /** Waits for a finished file matching `pattern`; Chrome writes .crdownload first. */
  async function file(pattern, timeoutMs = 15000) {
    const deadline = Date.now() + timeoutMs;
    for (;;) {
      const names = await readdir(directory);
      const match = names.find((name) => pattern.test(name) && !name.endsWith('.crdownload'));
      if (match) return path.join(directory, match);
      if (Date.now() > deadline) {
        throw new Error(`no download matching ${pattern} in ${timeoutMs}ms, saw: ${names}`);
      }
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
  }

  return {
    bytes: async (pattern) => readFile(await file(pattern)),
    text: async (pattern) => readFile(await file(pattern), 'utf8'),
  };
}

/** Clicks through every language tab, reporting what the panel shows on each. */
async function walkLanguageTabs(page) {
  await page.waitForSelector('.ms-code__tabs .ms-lang-tab');
  const count = await page.$$eval('.ms-code__tabs .ms-lang-tab', (els) => els.length);
  const seen = [];
  for (let i = 0; i < count; i++) {
    await page.click(`.ms-code__tabs .ms-lang-tab:nth-child(${i + 1})`);
    seen.push(
      await page.evaluate(() => {
        const visible = [...document.querySelectorAll('.ms-code__pane')].filter(
          (pane) => getComputedStyle(pane).visibility === 'visible',
        );
        const fileEl = document.querySelector('[data-file]');
        return {
          label: document.querySelector('.ms-lang-tab.is-active').textContent.trim(),
          visible: visible.length,
          code: visible[0]?.textContent ?? '',
          file: fileEl.textContent,
          href: fileEl.getAttribute('href'),
          install: document.querySelector('[data-install]')?.textContent ?? '',
          // Rounded because a fractional line-height leaves sub-pixel noise.
          height: Math.round(
            document.querySelector('.ms-code__panes').getBoundingClientRect().height,
          ),
        };
      }),
    );
  }
  return seen;
}

// One case per demo page. `expect` is matched against the snippet each tab
// shows, which is what catches a page wiring up the wrong snippet set.
const TAB_PAGES = [
  {
    name: 'STT',
    url: '/stt/?local=1&assets=local',
    labels: ['JavaScript', 'Python', 'Swift', 'Android'],
    files: [
      'live-transcription.js',
      'mic_transcription.py',
      'TranscriberApp.swift',
      'MainActivity.java',
    ],
    expect: [
      /import \{ MicTranscriber/,
      /from moonshine_voice import MicTranscriber/,
      /try await mic\.load\(\)/,
      /new MicTranscriber\(this\)/,
    ],
    installs: [
      'npm i @moonshine-ai/moonshine-wasm',
      'pip install moonshine-voice',
      'https://github.com/moonshine-ai/moonshine-swift/',
      'ai.moonshine:moonshine-voice:0.1.1',
    ],
  },
  {
    name: 'TTS',
    url: '/tts/?local=1&assets=local&voice=kokoro_af_heart',
    labels: ['JavaScript', 'Python', 'Swift', 'Android'],
    files: ['speak.js', 'text_to_speech.py', 'TextToSpeechApp.swift', 'MainActivity.java'],
    expect: [
      /import \{ TextToSpeech/,
      /from moonshine_voice import TextToSpeech/,
      /try await tts\.say\(/,
      /TextToSpeech tts = new TextToSpeech\(this\)/,
    ],
    installs: [
      'npm i @moonshine-ai/moonshine-wasm',
      'pip install moonshine-voice',
      'https://github.com/moonshine-ai/moonshine-swift/',
      'ai.moonshine:moonshine-voice:0.1.1',
    ],
  },
  {
    name: 'Voice agent',
    url: '/agent-flow/?local=1&assets=local&nomic=1',
    labels: ['JavaScript', 'Python', 'Swift', 'Android'],
    files: ['wifi-agent.js', 'agent_flow.py', 'AgentFlowApp.swift', 'MainActivity.java'],
    expect: [
      /agent\.listenFor\(/,
      /def setup_wifi\(d\):/,
      /func wifiSetup\(/,
      /private void wifiSetup\(AgentFlow\.Dialog d\)/,
    ],
    installs: [
      'npm i @moonshine-ai/moonshine-wasm',
      'pip install moonshine-voice',
      'https://github.com/moonshine-ai/moonshine-swift/',
      'ai.moonshine:moonshine-voice:0.1.1',
    ],
  },
];

for (const page_ of TAB_PAGES) {
  test(`${page_.name} example shows one snippet per language, without resizing`, { skip }, async () => {
    const page = await openPage(page_.url);
    try {
      const seen = await walkLanguageTabs(page);
      assert.deepEqual(
        seen.map((pane) => pane.label),
        page_.labels,
      );

      for (const [i, pane] of seen.entries()) {
        assert.equal(pane.visible, 1, `${pane.label} should show exactly one snippet`);
        // Each tab shows its own language, not copies of the JavaScript one.
        assert.match(pane.code, page_.expect[i], `${page_.name} ${pane.label} snippet`);
        // The caption links to the file it names, on the main branch, anchored
        // at the lines the snippet came from.
        assert.match(
          pane.href ?? '',
          /^https:\/\/github\.com\/moonshine-ai\/moonshine\/blob\/main\/examples\/\S+#L\d+-L\d+$/,
          `${pane.label} caption should link to its source lines`,
        );
      }

      assert.deepEqual(
        seen.map((pane) => pane.file),
        page_.files,
      );
      assert.deepEqual(
        seen.map((pane) => pane.install),
        page_.installs,
      );

      // The panel reserves room for the longest snippet, so choosing a language
      // never shifts the rest of the page under the reader.
      const heights = new Set(seen.map((pane) => pane.height));
      assert.equal(heights.size, 1, `panel height changed between tabs: ${[...heights]}`);
    } finally {
      await page.close();
    }
  });
}

test('Voice agent highlight follows the reader between languages', { skip }, async () => {
  const page = await openPage('/agent-flow/?local=1&assets=local&nomic=1');
  try {
    await page.waitForSelector('.ms-code__tabs .ms-lang-tab');

    // Park the flow on a step without running a conversation, which needs
    // models and a lot of time. The page's own step() is not reachable from
    // here, so drive markCodeStep the way it does.
    const highlighted = async (tabIndex, lineNumber) => {
      await page.click(`.ms-code__tabs .ms-lang-tab:nth-child(${tabIndex + 1})`);
      return page.evaluate(async (line) => {
        const ui = await import('/assets/moonshine-ui.js');
        ui.markCodeStep(document.getElementById('code'), line);
        const marked = document.querySelectorAll('.ms-line.is-running');
        return { count: marked.length, text: marked[0]?.textContent ?? '' };
      }, lineNumber);
    };

    // "confirmApply" is line 7 in JavaScript and Python but line 8 in Swift,
    // which is the reason each snippet carries its own step map.
    const js = await highlighted(0, 7);
    assert.equal(js.count, 1, 'exactly one line should be lit, in the open tab only');
    assert.match(js.text, /Apply these changes\?/);

    const swift = await highlighted(2, 8);
    assert.equal(swift.count, 1);
    assert.match(swift.text, /Apply these changes\?/);
    assert.match(swift.text, /try await/, 'expected the Swift line, not the JavaScript one');
  } finally {
    await page.close();
  }
});

// Chrome maintains its own "default" capture device, and when it disagrees with
// the operating system's it returns a live track of digital silence rather than
// an error. The only reliable workaround is naming a device explicitly, so the
// choice is saved and shared. These tests pin down that it is actually applied,
// and that a saved device which later disappears degrades to the default rather
// than wedging capture permanently.
const DEVICE_KEY = 'moonshine.audioInputDeviceId';

/** Records the constraints of every getUserMedia call the page makes. */
async function recordMicRequests(page) {
  await page.evaluateOnNewDocument(() => {
    window.__gum = [];
    const real = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    navigator.mediaDevices.getUserMedia = (constraints) => {
      window.__gum.push(JSON.parse(JSON.stringify(constraints)));
      return real(constraints);
    };
  });
}

test('STT example opens the saved capture device', { skip }, async () => {
  const page = await browser.newPage();
  try {
    // Seed the preference with a real, non-default fake device. Device ids are
    // origin-scoped, so this has to happen on the served origin.
    await page.goto(`${server.origin}/stt/?local=1&assets=local`, { waitUntil: 'load' });
    const chosen = await page.evaluate(async (key) => {
      const granted = await navigator.mediaDevices.getUserMedia({ audio: true });
      granted.getTracks().forEach((t) => t.stop());
      const devices = await navigator.mediaDevices.enumerateDevices();
      const device = devices.find(
        (d) => d.kind === 'audioinput' && d.deviceId && d.deviceId !== 'default',
      );
      localStorage.setItem(key, device.deviceId);
      return device.deviceId;
    }, DEVICE_KEY);

    await recordMicRequests(page);
    await page.goto(`${server.origin}/stt/?local=1&assets=local`, { waitUntil: 'load' });
    await page.waitForSelector('#toggle:not([disabled])', { timeout: 120000 });
    await page.click('#toggle');
    await page.waitForFunction(() => (window.__gum ?? []).length > 0, { timeout: 60000 });

    const requests = await page.evaluate(() => window.__gum);
    assert.ok(requests.length > 0, 'expected the page to open a microphone');
    for (const request of requests) {
      assert.equal(
        request.audio?.deviceId?.exact,
        chosen,
        `every capture should name the saved device, got ${JSON.stringify(request)}`,
      );
    }

    // The picker should reflect the saved choice rather than silently diverging.
    assert.equal(await page.$eval('#device', (el) => el.value), chosen);
  } finally {
    await page.close();
  }
});

test('STT example falls back when the saved device is gone', { skip }, async () => {
  const page = await browser.newPage();
  try {
    await page.goto(`${server.origin}/stt/?local=1&assets=local`, { waitUntil: 'load' });
    await page.evaluate((key) => localStorage.setItem(key, 'no-such-device'), DEVICE_KEY);

    await recordMicRequests(page);
    await page.goto(`${server.origin}/stt/?local=1&assets=local`, { waitUntil: 'load' });
    await page.waitForSelector('#toggle:not([disabled])', { timeout: 120000 });
    await page.click('#toggle');
    await page.waitForFunction(() => (window.__gum ?? []).length > 0, { timeout: 60000 });

    const requests = await page.evaluate(() => window.__gum);
    for (const request of requests) {
      assert.equal(
        request.audio,
        true,
        `a missing device should fall back to the default, got ${JSON.stringify(request)}`,
      );
    }
    assert.equal(
      await page.evaluate((key) => localStorage.getItem(key), DEVICE_KEY),
      null,
      'the stale device id should be forgotten, not retried forever',
    );
    // Capture should genuinely reach the running state, not merely not crash.
    // start() only labels the button once mic.start() has resolved, which is
    // after the getUserMedia call waited on above.
    await page.waitForFunction(
      () => document.getElementById('micLabel').textContent.trim() === 'Listening',
      { timeout: 60000 },
    );
  } finally {
    await page.close();
  }
});
