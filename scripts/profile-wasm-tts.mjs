// CPU-profiles a single browser TTS synthesis and reports the hottest WASM
// functions by self time, so a slow ONNX kernel shows up by name.
//
// Also samples OS-level CPU usage of the Chrome processes while synthesis runs.
// Wall-clock-vs-CPU-time is the quickest way to tell whether ONNX Runtime is
// actually spreading work across its pthread pool or grinding on one core.
//
//   node scripts/profile-wasm-tts.mjs

import { execSync } from 'node:child_process';
import {
  tryLoadPuppeteer,
  findChrome,
  startExampleServer,
} from '../wasm/tests/browser-helpers.mjs';

const TEXT =
  'Hello from Moonshine. I am running right inside your browser, and nothing ' +
  'I say is sent to a server. Your table is ready, please make your way to ' +
  'the host stand.';

/** Total CPU-seconds consumed so far by every Chrome process in this tree. */
function chromeCpuSeconds() {
  try {
    const out = execSync(
      "ps -Ao time,comm | grep -i 'Google Chrome' | awk '{print $1}'",
      { encoding: 'utf8' },
    );
    let total = 0;
    for (const line of out.trim().split('\n')) {
      if (!line) continue;
      const parts = line.split(':').map(Number);
      // ps TIME is [dd-]hh:mm:ss or mm:ss.ss
      total +=
        parts.length === 3
          ? parts[0] * 3600 + parts[1] * 60 + parts[2]
          : parts[0] * 60 + parts[1];
    }
    return total;
  } catch {
    return NaN;
  }
}

async function main() {
  const puppeteer = await tryLoadPuppeteer();
  const chrome = findChrome();
  if (!puppeteer || !chrome) throw new Error('puppeteer-core / Chrome not available');

  const server = await startExampleServer();
  const browser = await puppeteer.launch({
    executablePath: chrome,
    headless: true,
    protocolTimeout: 1_800_000,
    args: ['--no-sandbox', '--disable-dev-shm-usage', '--mute-audio'],
  });

  try {
    const page = await browser.newPage();
    page.on('console', (m) => {
      if (/^\[bench\]/.test(m.text())) console.log(`  ${m.text()}`);
    });
    await page.goto(`${server.origin}/tts/?local=1&assets=local`, {
      waitUntil: 'domcontentloaded',
    });

    // Load + warm up outside the profiled region.
    await page.evaluate(async (origin) => {
      const { TextToSpeech } = await import(`${origin}/wasm/dist/index.js`);
      const tts = new TextToSpeech()
        .language('en_us')
        .voice('kokoro_af_heart')
        .modelsFrom('/tts-data');
      await tts.load();
      tts.synthesize('Warm up.');
      globalThis.__tts = tts;
      console.log('[bench] loaded and warmed up');
    }, server.origin);

    const client = await page.createCDPSession();
    await client.send('Profiler.enable');
    await client.send('Profiler.setSamplingInterval', { interval: 200 });

    const cpuBefore = chromeCpuSeconds();
    await client.send('Profiler.start');
    const wall = await page.evaluate((text) => {
      const t0 = performance.now();
      const r = globalThis.__tts.synthesize(text);
      const ms = performance.now() - t0;
      console.log(
        `[bench] profiled synthesis ${ms.toFixed(0)} ms for ` +
          `${(r.audio.length / r.sampleRate).toFixed(2)} s audio`,
      );
      return { ms, audioSeconds: r.audio.length / r.sampleRate };
    }, TEXT);
    const { profile } = await client.send('Profiler.stop');
    const cpuAfter = chromeCpuSeconds();

    console.log('\n=== Wall clock vs CPU ===');
    console.log(`Synthesis wall time:   ${(wall.ms / 1000).toFixed(2)} s`);
    console.log(`Audio produced:        ${wall.audioSeconds.toFixed(2)} s`);
    console.log(`Real-time factor:      ${(wall.ms / 1000 / wall.audioSeconds).toFixed(2)}`);
    const cpu = cpuAfter - cpuBefore;
    console.log(`Chrome CPU consumed:   ${cpu.toFixed(1)} s`);
    console.log(
      `Average cores busy:    ${(cpu / (wall.ms / 1000)).toFixed(2)} ` +
        `(1.0 means single-threaded)`,
    );

    reportProfile(profile);
    reportTrampolineCallers(profile);
  } finally {
    await browser.close();
    await server.close();
  }
}

/** Aggregates a CDP CPU profile into self-time per function. */
function reportProfile(profile) {
  const byId = new Map(profile.nodes.map((n) => [n.id, n]));
  const selfSamples = new Map();
  for (const id of profile.samples) {
    selfSamples.set(id, (selfSamples.get(id) ?? 0) + 1);
  }
  const totalSamples = profile.samples.length;
  const durationMs =
    (profile.endTime - profile.startTime) / 1000;

  const rows = [];
  for (const [id, count] of selfSamples) {
    const node = byId.get(id);
    if (!node) continue;
    const f = node.callFrame;
    const name = f.functionName || '(anonymous)';
    const where = f.url?.includes('.wasm') ? 'wasm' : f.url ? 'js' : '';
    rows.push({ name, where, count });
  }
  // Merge duplicates by name.
  const merged = new Map();
  for (const r of rows) {
    const key = `${r.name}|${r.where}`;
    merged.set(key, (merged.get(key) ?? 0) + r.count);
  }

  console.log(
    `\n=== Hottest functions by self time (${totalSamples} samples over ${durationMs.toFixed(0)} ms) ===`,
  );
  const sorted = [...merged.entries()].sort((a, b) => b[1] - a[1]).slice(0, 30);
  for (const [key, count] of sorted) {
    const [name, where] = key.split('|');
    const pct = (count / totalSamples) * 100;
    const ms = (count / totalSamples) * durationMs;
    console.log(
      `${pct.toFixed(1).padStart(6)}%  ${ms.toFixed(0).padStart(7)} ms  ${where.padEnd(4)}  ${name.slice(0, 110)}`,
    );
  }
}

/**
 * Emscripten's JS exception handling turns a C++ call that needs unwinding
 * into `wasm -> invoke_* (JS) -> getWasmTableEntry -> wasm`. Those frames carry
 * no ORT symbol, so charge each sample to its nearest named C++ ancestor to see
 * which kernel is paying for the round trips.
 */
function reportTrampolineCallers(profile) {
  const byId = new Map(profile.nodes.map((n) => [n.id, n]));
  const parent = new Map();
  for (const n of profile.nodes) {
    for (const c of n.children ?? []) parent.set(c, n.id);
  }

  const TRAMPOLINE =
    /^(invoke_|getWasmTableEntry|stackSave|stackRestore|emscripten_stack_|wasm-to-js|js-to-wasm|dynCall|\(anonymous\)$)/;
  const isTrampoline = (n) => {
    const name = n.callFrame.functionName || '(anonymous)';
    return TRAMPOLINE.test(name) || name === '';
  };

  const counts = new Map();
  let trampolineSamples = 0;
  const totalSamples = profile.samples.length;
  const durationMs = (profile.endTime - profile.startTime) / 1000;

  for (const id of profile.samples) {
    let node = byId.get(id);
    if (!node || !isTrampoline(node)) continue;
    trampolineSamples++;
    // Walk up to the first frame that carries a real C++ / JS symbol.
    let cur = node;
    let owner = '(unattributed)';
    while (cur) {
      if (!isTrampoline(cur)) {
        owner = cur.callFrame.functionName || '(anonymous)';
        break;
      }
      cur = byId.get(parent.get(cur.id));
    }
    counts.set(owner, (counts.get(owner) ?? 0) + 1);
  }

  console.log(
    `\n=== Exception-trampoline time charged to nearest real caller ` +
      `(${((trampolineSamples / totalSamples) * 100).toFixed(1)}% of all samples) ===`,
  );
  for (const [name, count] of [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 15)) {
    console.log(
      `${((count / totalSamples) * 100).toFixed(1).padStart(6)}%  ${(
        (count / totalSamples) *
        durationMs
      )
        .toFixed(0)
        .padStart(7)} ms  ${name.slice(0, 110)}`,
    );
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
