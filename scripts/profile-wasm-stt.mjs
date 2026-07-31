// Control for profile-wasm-tts.mjs: profiles a browser STT run so the two
// pipelines can be compared under identical conditions. STT reaches ONNX
// Runtime through the C status-returning API, TTS through the throwing C++
// API, so the share of time in exception trampolines is the thing to compare.
//
//   node scripts/profile-wasm-stt.mjs

import {
  tryLoadPuppeteer,
  findChrome,
  startExampleServer,
} from '../wasm/tests/browser-helpers.mjs';

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
    await page.goto(`${server.origin}/stt/?local=1&assets=local`, {
      waitUntil: 'domcontentloaded',
    });

    await page.evaluate(async (origin) => {
      const { Transcriber, ModelArch } = await import(`${origin}/wasm/dist/index.js`);
      const stt = await Transcriber.loadFromUrls(
        {
          'encoder_model.ort': '/test-assets/tiny-en/encoder_model.ort',
          'decoder_model_merged.ort': '/test-assets/tiny-en/decoder_model_merged.ort',
          'tokenizer.bin': '/test-assets/tiny-en/tokenizer.bin',
        },
        { modelArch: ModelArch.Tiny },
      );
      const res = await fetch('/test-assets/two_cities_16k.wav');
      const bytes = await res.arrayBuffer();
      const ctx = new AudioContext({ sampleRate: 16000 });
      const decoded = await ctx.decodeAudioData(bytes);
      globalThis.__pcm = decoded.getChannelData(0);
      globalThis.__stt = stt;
      stt.transcribe(globalThis.__pcm.slice(0, 16000 * 5), { sampleRate: 16000 });
      console.log(
        `[bench] STT loaded; clip is ${(globalThis.__pcm.length / 16000).toFixed(1)} s`,
      );
    }, server.origin);

    const client = await page.createCDPSession();
    await client.send('Profiler.enable');
    await client.send('Profiler.setSamplingInterval', { interval: 200 });
    await client.send('Profiler.start');
    const wall = await page.evaluate(() => {
      const t0 = performance.now();
      // Repeat so the profile has a comparable number of samples to the TTS run.
      let reps = 0;
      while (performance.now() - t0 < 10000) {
        globalThis.__stt.transcribe(globalThis.__pcm, { sampleRate: 16000 });
        reps++;
      }
      const ms = performance.now() - t0;
      const audioSeconds = (globalThis.__pcm.length / 16000) * reps;
      console.log(`[bench] ${reps} transcriptions in ${ms.toFixed(0)} ms`);
      return { ms, audioSeconds, reps };
    });
    const { profile } = await client.send('Profiler.stop');

    console.log('\n=== STT ===');
    console.log(`Wall time:        ${(wall.ms / 1000).toFixed(2)} s`);
    console.log(`Audio processed:  ${wall.audioSeconds.toFixed(2)} s`);
    console.log(`Real-time factor: ${(wall.ms / 1000 / wall.audioSeconds).toFixed(3)}`);
    reportProfile(profile);
  } finally {
    await browser.close();
    await server.close();
  }
}

function reportProfile(profile) {
  const byId = new Map(profile.nodes.map((n) => [n.id, n]));
  const selfSamples = new Map();
  for (const id of profile.samples) selfSamples.set(id, (selfSamples.get(id) ?? 0) + 1);
  const totalSamples = profile.samples.length;
  const durationMs = (profile.endTime - profile.startTime) / 1000;

  const merged = new Map();
  for (const [id, count] of selfSamples) {
    const node = byId.get(id);
    if (!node) continue;
    const f = node.callFrame;
    const key = `${f.functionName || '(anonymous)'}|${
      f.url?.includes('.wasm') ? 'wasm' : f.url ? 'js' : ''
    }`;
    merged.set(key, (merged.get(key) ?? 0) + count);
  }

  console.log(
    `\n=== Hottest functions by self time (${totalSamples} samples over ${durationMs.toFixed(0)} ms) ===`,
  );
  for (const [key, count] of [...merged.entries()].sort((a, b) => b[1] - a[1]).slice(0, 20)) {
    const [name, where] = key.split('|');
    console.log(
      `${((count / totalSamples) * 100).toFixed(1).padStart(6)}%  ${(
        (count / totalSamples) *
        durationMs
      )
        .toFixed(0)
        .padStart(7)} ms  ${where.padEnd(4)}  ${name.slice(0, 110)}`,
    );
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
