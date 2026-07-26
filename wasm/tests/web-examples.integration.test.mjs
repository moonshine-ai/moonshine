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

test('Dialog-flow example drives a flow from typed input', { skip }, async () => {
  const page = await openPage('/dialog-flow/?local=1&assets=local&nomic=1');
  try {
    // nomic=1 auto-builds the runner (TTS + intent recognizer) without a mic.
    await page.waitForFunction(() => window.__dialogReady === true, { timeout: 120000 });

    await page.type('#utterance', 'set up wifi');
    await page.click('#send');

    // The flow's first prompt asks for the wifi network name; it is spoken via
    // speakFn, which logs an `assistant:` line.
    await page.waitForFunction(
      () =>
        [...document.querySelectorAll('#log .assistant')].some((el) =>
          /wifi network/i.test(el.textContent),
        ),
      { timeout: 120000 },
    );

    const assistantLines = await page.evaluate(() =>
      [...document.querySelectorAll('#log .assistant')].map((el) => el.textContent),
    );
    assert.ok(
      assistantLines.some((t) => /wifi network/i.test(t)),
      `expected an assistant prompt about the wifi network, got: ${JSON.stringify(assistantLines)}`,
    );
  } finally {
    await page.close();
  }
});
