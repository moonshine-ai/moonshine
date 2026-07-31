// Measures browser TTS/STT throughput against the locally-built WASM binding.
//
// Starts examples/web/serve.mjs (which supplies the cross-origin isolation
// headers the threaded build needs), drives headless Chrome via puppeteer-core,
// and times synthesis of texts of increasing length so per-call overhead can be
// separated from per-second-of-audio cost. The same page then times an STT run
// over the synthesized audio, giving a like-for-like real-time factor.
//
//   node scripts/bench-wasm-tts.mjs
//
// Environment:
//   MOONSHINE_BENCH_REPS   synthesis repetitions per text (default 3)

import {
  tryLoadPuppeteer,
  findChrome,
  startExampleServer,
} from '../wasm/tests/browser-helpers.mjs';

/**
 * Same flags as the integration-test launcher, but with a protocol timeout
 * generous enough for a whole benchmark sweep to run inside one evaluate().
 */
function launchBrowser(puppeteer, executablePath) {
  return puppeteer.launch({
    executablePath,
    headless: true,
    protocolTimeout: 1_800_000,
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-dev-shm-usage',
      '--autoplay-policy=no-user-gesture-required',
      '--mute-audio',
    ],
  });
}

const REPS = Number(process.env.MOONSHINE_BENCH_REPS ?? 3);

const TEXTS = [
  { label: 'short (7 words)', text: 'Hello from Moonshine, running in your browser.' },
  {
    label: 'medium (24 words)',
    text:
      'Hello from Moonshine. I am running right inside your browser, and nothing I say is sent to a server. ' +
      'Your table is ready, please make your way to the host stand.',
  },
  {
    label: 'long (56 words)',
    text:
      'Once upon a time, in a valley nobody had bothered to name, there lived a very stubborn goat. ' +
      'The goat believed that the bridge belonged to him, and that anyone wishing to cross it should first ' +
      'explain themselves. Travellers found this tiresome, but the goat was patient, and the valley was wide, ' +
      'and there was no other way across the river for many miles.',
  },
];

async function main() {
  const puppeteer = await tryLoadPuppeteer();
  if (!puppeteer) throw new Error('puppeteer-core is not installed (npm i in wasm/)');
  const chrome = findChrome();
  if (!chrome) throw new Error('No Chrome/Chromium found');

  const server = await startExampleServer();
  const browser = await launchBrowser(puppeteer, chrome);
  try {
    const page = await browser.newPage();
    page.on('console', (m) => {
      const t = m.text();
      if (/^\[bench\]/.test(t)) console.log(`  ${t}`);
      else if (/error|abort/i.test(t)) console.log(`  [page] ${t}`);
    });
    page.on('pageerror', (e) => console.log(`  [pageerror] ${e.message}`));

    await page.goto(`${server.origin}/tts/?local=1&assets=local`, {
      waitUntil: 'domcontentloaded',
    });

    const env = await page.evaluate(() => ({
      crossOriginIsolated: globalThis.crossOriginIsolated,
      sharedArrayBuffer: typeof SharedArrayBuffer !== 'undefined',
      hardwareConcurrency: navigator.hardwareConcurrency,
      userAgent: navigator.userAgent,
    }));
    console.log('\n=== Environment ===');
    console.log(JSON.stringify(env, null, 2));

    const results = await page.evaluate(
      async (origin, texts, reps) => {
        const { TextToSpeech, Transcriber, ModelArch } = await import(
          `${origin}/wasm/dist/index.js`
        );

        const out = { tts: [], stt: [], load: {} };
        const note = (m) => console.log(`[bench] ${m}`);

        let t0 = performance.now();
        const tts = new TextToSpeech()
          .language('en_us')
          .voice('kokoro_af_heart')
          .modelsFrom('/tts-data');
        await tts.load();
        out.load.ttsMs = performance.now() - t0;
        note(`TTS loaded in ${out.load.ttsMs.toFixed(0)} ms`);

        // Warm-up: the first call pays for lazy G2P dictionary loading and
        // ORT's first-run allocations, which would otherwise skew rep 1.
        t0 = performance.now();
        tts.synthesize('Warm up the synthesizer.');
        out.load.warmupMs = performance.now() - t0;
        note(`warm-up synthesis ${out.load.warmupMs.toFixed(0)} ms`);

        let longest = null;
        for (const { label, text } of texts) {
          const times = [];
          let audioSeconds = 0;
          for (let i = 0; i < reps; i++) {
            const start = performance.now();
            const r = tts.synthesize(text);
            const ms = performance.now() - start;
            times.push(ms);
            audioSeconds = r.audio.length / r.sampleRate;
            note(
              `TTS ${label} rep ${i + 1}: ${ms.toFixed(0)} ms for ` +
                `${audioSeconds.toFixed(2)} s audio (RTF ${(ms / 1000 / audioSeconds).toFixed(2)})`,
            );
            if (!longest || r.audio.length > longest.audio.length) {
              longest = { audio: r.audio, sampleRate: r.sampleRate };
            }
          }
          out.tts.push({ label, chars: text.length, times, audioSeconds });
        }

        // STT over the longest synthesized clip, for a same-machine,
        // same-runtime comparison of real-time factor.
        try {
          t0 = performance.now();
          const stt = await Transcriber.loadFromUrls(
            {
              'encoder_model.ort': '/test-assets/tiny-en/encoder_model.ort',
              'decoder_model_merged.ort':
                '/test-assets/tiny-en/decoder_model_merged.ort',
              'tokenizer.bin': '/test-assets/tiny-en/tokenizer.bin',
            },
            { modelArch: ModelArch.Tiny },
          );
          out.load.sttMs = performance.now() - t0;
          note(`STT loaded in ${out.load.sttMs.toFixed(0)} ms`);

          // Resample 24k synth output to the 16k the STT model expects.
          const ratio = longest.sampleRate / 16000;
          const n = Math.floor(longest.audio.length / ratio);
          const pcm = new Float32Array(n);
          for (let i = 0; i < n; i++) pcm[i] = longest.audio[Math.floor(i * ratio)];

          stt.transcribe(pcm, { sampleRate: 16000 });
          const times = [];
          for (let i = 0; i < reps; i++) {
            const start = performance.now();
            stt.transcribe(pcm, { sampleRate: 16000 });
            const ms = performance.now() - start;
            times.push(ms);
            note(
              `STT rep ${i + 1}: ${ms.toFixed(0)} ms for ${(n / 16000).toFixed(2)} s audio ` +
                `(RTF ${(ms / 1000 / (n / 16000)).toFixed(3)})`,
            );
          }
          out.stt.push({ label: 'tiny-en batch', times, audioSeconds: n / 16000 });
        } catch (e) {
          out.sttError = String(e);
        }

        return out;
      },
      server.origin,
      TEXTS,
      REPS,
    );

    report(results, await browser.pages());

    // Emscripten spawns one dedicated worker per pthread, so counting them
    // shows whether ORT actually has a thread pool to run on.
    console.log(`\nDedicated workers in page: ${(await page.workers()).length}`);
  } finally {
    await browser.close();
    await server.close();
  }
}

const median = (xs) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];

function report(r, _pages) {
  console.log('\n=== Load times ===');
  console.log(`TTS (Kokoro 82M) load: ${r.load.ttsMs?.toFixed(0)} ms`);
  if (r.load.sttMs) console.log(`STT (tiny-en) load:    ${r.load.sttMs.toFixed(0)} ms`);

  console.log('\n=== TTS synthesis (Kokoro af_heart, 24 kHz) ===');
  console.log(
    'text'.padEnd(20),
    'audio s'.padStart(9),
    'median ms'.padStart(11),
    'RTF'.padStart(7),
    '  all reps (ms)',
  );
  for (const row of r.tts) {
    const m = median(row.times);
    console.log(
      row.label.padEnd(20),
      row.audioSeconds.toFixed(2).padStart(9),
      m.toFixed(0).padStart(11),
      (m / 1000 / row.audioSeconds).toFixed(2).padStart(7),
      '  ' + row.times.map((t) => t.toFixed(0)).join(', '),
    );
  }

  if (r.sttError) {
    console.log(`\nSTT comparison failed: ${r.sttError}`);
  } else {
    console.log('\n=== STT transcription ===');
    for (const row of r.stt) {
      const m = median(row.times);
      console.log(
        row.label.padEnd(20),
        row.audioSeconds.toFixed(2).padStart(9),
        m.toFixed(0).padStart(11),
        (m / 1000 / row.audioSeconds).toFixed(2).padStart(7),
        '  ' + row.times.map((t) => t.toFixed(0)).join(', '),
      );
    }
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
