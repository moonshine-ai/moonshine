// Regression: TTS voice cloning must not freeze the main thread after capture.
//
//   MOONSHINE_BROWSER_TESTS=1 node --test tests/tts-clone-hang.investigation.mjs

import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import {
  REPO_ROOT,
  tryLoadPuppeteer,
  findChrome,
  startExampleServer,
} from './browser-helpers.mjs';

const TWO_CITIES_WAV = path.join(REPO_ROOT, 'test-assets', 'two_cities_16k.wav');

const enabled = process.env.MOONSHINE_BROWSER_TESTS === '1';
const puppeteer = enabled ? await tryLoadPuppeteer() : null;
const chromePath = enabled ? findChrome() : null;

const skip = !enabled
  ? 'set MOONSHINE_BROWSER_TESTS=1 to run'
  : !puppeteer
    ? 'puppeteer-core missing'
    : !chromePath
      ? 'Chrome not found'
      : false;

test('TTS cloneFrom stays responsive after capture', { skip, timeout: 600000 }, async () => {
  const server = await startExampleServer();
  const browser = await puppeteer.launch({
    executablePath: chromePath,
    headless: true,
    protocolTimeout: 300000,
    args: [
      `--use-file-for-fake-audio-capture=${TWO_CITIES_WAV}%noloop`,
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-dev-shm-usage',
      '--autoplay-policy=no-user-gesture-required',
      '--mute-audio',
      '--use-fake-ui-for-media-stream',
      '--use-fake-device-for-media-stream',
    ],
  });
  const page = await browser.newPage();
  page.on('pageerror', (err) => console.log(`[pageerror] ${err.message}`));

  try {
    await page.goto(
      `${server.origin}/tts/?local=1&fresh=1&voice=zipvoice_american_female`,
      { waitUntil: 'load' },
    );

    await page.waitForFunction(
      () => {
        const record = document.getElementById('record');
        const status = document.getElementById('status')?.textContent || '';
        return record && !record.disabled && /Ready/i.test(status);
      },
      { timeout: 600000 },
    );
    await page.click('#record');

    await page.waitForFunction(
      () =>
        /Trimming and transcribing/i.test(
          document.getElementById('cloneStatus')?.textContent || '',
        ),
      { timeout: 300000 },
    );

    // While trimming/transcribing runs, the main thread must keep answering.
    const deadline = Date.now() + 180000;
    let completed = false;
    while (Date.now() < deadline) {
      const snap = await Promise.race([
        page.evaluate(() => ({
          status: document.getElementById('cloneStatus')?.textContent || '',
          ok: true,
        })),
        new Promise((resolve) =>
          setTimeout(
            () => resolve({ ok: false, status: 'evaluate-timeout' }),
            8000,
          ),
        ),
      ]);
      assert.equal(
        snap.ok,
        true,
        `main thread blocked after trimming; status=${snap.status}`,
      );
      if (/Cloned from/i.test(snap.status)) {
        completed = true;
        break;
      }
      await new Promise((r) => setTimeout(r, 1000));
    }
    assert.equal(completed, true, 'clone did not finish');
  } finally {
    await browser.close();
    await server.close();
  }
});
