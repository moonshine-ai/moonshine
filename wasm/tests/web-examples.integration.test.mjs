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

test('Dialog-flow example runs a whole conversation from typed input', { skip }, async () => {
  const page = await openPage('/dialog-flow/?local=1&assets=local&nomic=1');

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
