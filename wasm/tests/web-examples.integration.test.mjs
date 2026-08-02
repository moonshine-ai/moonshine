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
