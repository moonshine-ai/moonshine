// Helpers for the browser integration test (web-examples.integration.test.mjs):
// spin up the examples static server (examples/web/serve.mjs, which serves the
// locally-built /wasm/dist binding plus the in-repo model assets), locate a
// Chrome/Chromium binary, and launch it via puppeteer-core.
//
// Everything here degrades gracefully: if puppeteer-core isn't installed or no
// browser can be found, the caller can skip the test rather than fail.

import { spawn } from 'node:child_process';
import net from 'node:net';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
export const REPO_ROOT = path.resolve(here, '..', '..');
export const EXAMPLES_DIR = path.join(REPO_ROOT, 'examples', 'web');
export const SERVE_SCRIPT = path.join(EXAMPLES_DIR, 'serve.mjs');

/** Dynamically imports puppeteer-core, returning null if it isn't installed. */
export async function tryLoadPuppeteer() {
  try {
    return (await import('puppeteer-core')).default;
  } catch {
    return null;
  }
}

/**
 * Resolves a Chrome/Chromium executable. Honors PUPPETEER_EXECUTABLE_PATH /
 * CHROME_PATH first, then falls back to the usual install locations. Returns
 * null when nothing is found so the test can skip.
 */
export function findChrome() {
  const fromEnv = process.env.PUPPETEER_EXECUTABLE_PATH || process.env.CHROME_PATH;
  if (fromEnv && fs.existsSync(fromEnv)) return fromEnv;

  const candidates = [
    // macOS
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary',
    '/Applications/Chromium.app/Contents/MacOS/Chromium',
    // Linux
    '/usr/bin/google-chrome',
    '/usr/bin/google-chrome-stable',
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser',
    '/snap/bin/chromium',
  ];
  return candidates.find((p) => fs.existsSync(p)) ?? null;
}

/** Grabs a free TCP port from the OS. */
export function getFreePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.on('error', reject);
    srv.listen(0, '127.0.0.1', () => {
      const { port } = srv.address();
      srv.close(() => resolve(port));
    });
  });
}

async function waitForServer(url, timeoutMs = 15000) {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    try {
      const res = await fetch(url);
      if (res.ok) return;
    } catch {
      /* not up yet */
    }
    if (Date.now() > deadline) {
      throw new Error(`Server at ${url} did not become ready in ${timeoutMs}ms`);
    }
    await new Promise((r) => setTimeout(r, 150));
  }
}

/**
 * Starts examples/web/serve.mjs on a free port and waits until it responds.
 * Returns `{ port, origin, close() }`.
 */
export async function startExampleServer() {
  const port = await getFreePort();
  const child = spawn(process.execPath, [SERVE_SCRIPT, String(port)], {
    cwd: EXAMPLES_DIR,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  child.stdout.on('data', () => {});
  child.stderr.on('data', (d) => process.stderr.write(`[serve] ${d}`));

  const origin = `http://localhost:${port}`;
  await waitForServer(`${origin}/stt/`);

  return {
    port,
    origin,
    close: () =>
      new Promise((resolve) => {
        child.once('exit', () => resolve());
        child.kill('SIGTERM');
      }),
  };
}

/** Launches a headless browser suited to the cross-origin-isolated examples. */
export async function launchBrowser(puppeteer, executablePath) {
  return puppeteer.launch({
    executablePath,
    headless: true,
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-dev-shm-usage',
      // Let TTS playback start without a gesture and stay silent in CI.
      '--autoplay-policy=no-user-gesture-required',
      '--mute-audio',
      // Grant microphone access automatically and back it with synthetic
      // devices, so capture paths can be exercised without real hardware.
      '--use-fake-ui-for-media-stream',
      '--use-fake-device-for-media-stream',
    ],
  });
}
