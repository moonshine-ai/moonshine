// Profiles the Meeting Notes web app: where wall time goes, and how far the
// transcript trails the recording clock.
//
// Drives headless Chrome the same way as the browser integration test
// (fake microphone + fakescreen, local WASM + test-assets). Runs twice —
// meeting diarization on, then off — and prints a JSON summary.
//
//   node scripts/profile-meeting-notes.mjs
//
// Environment:
//   MOONSHINE_PROFILE_SECONDS   capture length (default 60)
//   MOONSHINE_PROFILE_ARCH      tiny (default) or medium
//   MOONSHINE_PROFILE_DECODE     live = decode in-progress lines (app default is complete-only)
//   PUPPETEER_EXECUTABLE_PATH / CHROME_PATH

import path from 'node:path';
import {
  REPO_ROOT,
  tryLoadPuppeteer,
  findChrome,
  startExampleServer,
} from '../language-bindings/wasm/tests/browser-helpers.mjs';

const CAPTURE_SECONDS = Number(process.env.MOONSHINE_PROFILE_SECONDS ?? 60);
const PROFILE_WAV =
  process.env.MOONSHINE_PROFILE_WAV ||
  path.join(REPO_ROOT, 'test-assets', 'two_cities_16k.wav');
const PROFILE_ARCH = process.env.MOONSHINE_PROFILE_ARCH || 'tiny';
const PROFILE_DECODE = process.env.MOONSHINE_PROFILE_DECODE || '';

function percentile(sorted, p) {
  if (!sorted.length) return 0;
  const i = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return sorted[i];
}

function summarizeLag(samples, key) {
  const values = samples.map((s) => s[key]).filter((v) => Number.isFinite(v));
  if (!values.length) {
    return { n: 0, last: 0, max: 0, mean: 0, p50: 0, p95: 0 };
  }
  const sorted = [...values].sort((a, b) => a - b);
  const sum = values.reduce((a, b) => a + b, 0);
  return {
    n: values.length,
    last: values[values.length - 1],
    max: sorted[sorted.length - 1],
    mean: sum / values.length,
    p50: percentile(sorted, 50),
    p95: percentile(sorted, 95),
  };
}

/** Sum complete-event durations in a Chrome trace, grouped by thread name. */
function summarizeTrace(trace) {
  const events = trace?.traceEvents ?? [];
  const threadName = new Map();
  for (const ev of events) {
    if (ev.ph === 'M' && ev.name === 'thread_name' && ev.args?.name) {
      threadName.set(`${ev.pid}:${ev.tid}`, ev.args.name);
    }
  }
  const byThread = new Map();
  let scriptingUs = 0;
  let renderingUs = 0;
  let paintingUs = 0;
  let longTasks = 0;
  let longTaskMaxUs = 0;
  for (const ev of events) {
    if (ev.ph !== 'X' || !ev.dur) continue;
    const key = `${ev.pid}:${ev.tid}`;
    const name = threadName.get(key) || key;
    byThread.set(name, (byThread.get(name) ?? 0) + ev.dur);
    if (ev.name === 'RunTask' && ev.dur >= 50_000) {
      longTasks += 1;
      if (ev.dur > longTaskMaxUs) longTaskMaxUs = ev.dur;
    }
    if (ev.cat?.includes('devtools.timeline')) {
      if (ev.name === 'EvaluateScript' || ev.name === 'FunctionCall' || ev.name === 'v8.compile') {
        scriptingUs += ev.dur;
      } else if (ev.name === 'Layout' || ev.name === 'UpdateLayoutTree' || ev.name === 'RecalculateStyles') {
        renderingUs += ev.dur;
      } else if (ev.name === 'Paint' || ev.name === 'CompositeLayers') {
        paintingUs += ev.dur;
      }
    }
  }
  const threads = [...byThread.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8)
    .map(([name, us]) => ({ name, ms: us / 1000 }));
  return {
    threads,
    mainScriptingMs: scriptingUs / 1000,
    mainRenderingMs: renderingUs / 1000,
    mainPaintingMs: paintingUs / 1000,
    traceLongTasks: longTasks,
    traceLongTaskMaxMs: longTaskMaxUs / 1000,
  };
}

async function runCapture(browser, origin, { speakers, label }) {
  const page = await browser.newPage();
  page.on('pageerror', (err) => console.error(`[${label}] ${err.message}`));
  const speakersQ = speakers ? 'speakers=1' : 'speakers=0';
  const archQ = PROFILE_ARCH === 'tiny' ? '' : `&arch=${PROFILE_ARCH}`;
  const decodeQ = PROFILE_DECODE === 'live' ? '&decode=live' : '';
  await page.goto(
    `${origin}/meeting-notes/?local=1&assets=local&fakescreen=1&${speakersQ}${archQ}${decodeQ}`,
    { waitUntil: 'load' },
  );
  await page.waitForFunction(() => document.body.dataset.state === 'ready', {
    timeout: 600000,
  });

  const client = await page.createCDPSession();
  await client.send('Performance.enable');
  await page.tracing.start({
    categories: [
      'devtools.timeline',
      'v8.execute',
      'disabled-by-default-devtools.timeline',
    ],
  });

  const pageMetricsBefore = await page.metrics();
  await page.click('#record');
  await page.waitForFunction(() => document.body.dataset.state === 'capturing', {
    timeout: 60000,
  });
  // Capture wall time from record, not from first words: complete-only
  // decode has no provisional text until VAD closes a line.
  await new Promise((resolve) => setTimeout(resolve, CAPTURE_SECONDS * 1000));

  const app = await page.evaluate(() => {
    const m = window.__meetingNotes.metrics();
    return {
      ...m,
      lines: window.__meetingNotes.lines.size,
    };
  });
  const pageMetricsAfter = await page.metrics();
  let perfMetrics = [];
  try {
    const got = await client.send('Performance.getMetrics');
    perfMetrics = got.metrics ?? [];
  } catch {
    /* some Chrome builds omit this */
  }

  const traceBuf = await page.tracing.stop();
  let traceSummary = { threads: [] };
  try {
    const json = JSON.parse(Buffer.from(traceBuf).toString('utf8'));
    traceSummary = summarizeTrace(json);
  } catch (err) {
    console.error(`[${label}] could not parse trace: ${err.message}`);
  }

  await page.close();

  const lag = app.lagSamples ?? [];
  return {
    label,
    speakers,
    captureSeconds: CAPTURE_SECONDS,
    recordedSeconds: app.recordedSeconds,
    lines: app.lines,
    usingWorker: app.usingWorker,
    firstProvisionalMs: app.firstProvisionalMs,
    firstLineMs: app.firstLineMs,
    transcribeMs: app.transcribeMs,
    transcribeMaxMs: app.transcribeMaxMs,
    transcribePasses: app.transcribePasses,
    meetingPassMs: app.meetingPassMs,
    meetingPassMaxMs: app.meetingPassMaxMs,
    meetingPasses: app.meetingPasses,
    youPassMs: app.youPassMs,
    youPassMaxMs: app.youPassMaxMs,
    youPasses: app.youPasses,
    resampleMs: app.resampleMs,
    mixAtMs: app.mixAtMs,
    addAudioMs: app.addAudioMs,
    renderMs: app.renderMs,
    renderMaxMs: app.renderMaxMs,
    ingestLagMaxMs: app.ingestLagMaxMs,
    rafGapMaxMs: app.rafGapMaxMs,
    longTaskMaxMs: app.longTaskMaxMs,
    longTasks: app.longTasks,
    pendingAudioSeconds: app.pendingAudioSeconds,
    workerQueuedSeconds: app.workerQueuedSeconds,
    meetingLag: summarizeLag(lag, 'meetingLagS'),
    youLag: summarizeLag(lag, 'youLagS'),
    pending: summarizeLag(lag, 'pendingS'),
    workerQueued: summarizeLag(lag, 'workerQueuedS'),
    lagSamples: lag.map((s) => ({
      t: Number(s.tS.toFixed(2)),
      rec: Number(s.recordedS.toFixed(2)),
      meeting: Number(s.meetingLagS.toFixed(2)),
      you: Number(s.youLagS.toFixed(2)),
    })),
    chrome: {
      scriptDurationMs: ((pageMetricsAfter.ScriptDuration ?? 0) - (pageMetricsBefore.ScriptDuration ?? 0)) * 1000,
      taskDurationMs: ((pageMetricsAfter.TaskDuration ?? 0) - (pageMetricsBefore.TaskDuration ?? 0)) * 1000,
      layoutDurationMs: ((pageMetricsAfter.LayoutDuration ?? 0) - (pageMetricsBefore.LayoutDuration ?? 0)) * 1000,
      recalcStyleDurationMs:
        ((pageMetricsAfter.RecalcStyleDuration ?? 0) - (pageMetricsBefore.RecalcStyleDuration ?? 0)) * 1000,
      ...traceSummary,
      performanceMetrics: Object.fromEntries(perfMetrics.map((m) => [m.name, m.value])),
    },
  };
}

const puppeteer = await tryLoadPuppeteer();
const chromePath = findChrome();
if (!puppeteer) {
  console.error('puppeteer-core is not installed (npm i -D puppeteer-core in language-bindings/wasm)');
  process.exit(1);
}
if (!chromePath) {
  console.error('no Chrome/Chromium found (set PUPPETEER_EXECUTABLE_PATH)');
  process.exit(1);
}

console.error(
  `[profile-meeting-notes] ${CAPTURE_SECONDS}s capture, arch=${PROFILE_ARCH}, decode=${PROFILE_DECODE || 'live'}, wav=${PROFILE_WAV}, Chrome at ${chromePath}`,
);
const server = await startExampleServer();
const browser = await puppeteer.launch({
  executablePath: chromePath,
  headless: true,
  protocolTimeout: 1_800_000,
  args: [
    `--use-file-for-fake-audio-capture=${PROFILE_WAV}%noloop`,
    '--no-sandbox',
    '--disable-setuid-sandbox',
    '--disable-dev-shm-usage',
    '--autoplay-policy=no-user-gesture-required',
    '--mute-audio',
    '--use-fake-ui-for-media-stream',
    '--use-fake-device-for-media-stream',
  ],
});

try {
  const withDiarization = await runCapture(browser, server.origin, {
    speakers: true,
    label: 'diarization-on',
  });
  console.error(
    `[profile-meeting-notes] diarization on: meetingPassMax=${withDiarization.meetingPassMaxMs.toFixed(0)}ms ` +
      `lagMax=${withDiarization.meetingLag.max.toFixed(2)}s lagLast=${withDiarization.meetingLag.last.toFixed(2)}s ` +
      `rafGap=${withDiarization.rafGapMaxMs.toFixed(0)}ms`,
  );
  const withoutDiarization = await runCapture(browser, server.origin, {
    speakers: false,
    label: 'diarization-off',
  });
  console.error(
    `[profile-meeting-notes] diarization off: meetingPassMax=${withoutDiarization.meetingPassMaxMs.toFixed(0)}ms ` +
      `lagMax=${withoutDiarization.meetingLag.max.toFixed(2)}s lagLast=${withoutDiarization.meetingLag.last.toFixed(2)}s ` +
      `rafGap=${withoutDiarization.rafGapMaxMs.toFixed(0)}ms`,
  );
  const report = {
    capturedAt: new Date().toISOString(),
    captureSeconds: CAPTURE_SECONDS,
    wav: PROFILE_WAV,
    model: PROFILE_ARCH === 'medium' ? 'medium-streaming-en (local)' : 'tiny-streaming-en (local)',
    decode: PROFILE_DECODE || 'live',
    runs: [withDiarization, withoutDiarization],
  };
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
} finally {
  await browser.close();
  await server.close();
}
