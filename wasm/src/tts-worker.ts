/**
 * Web Worker that owns a Moonshine TTS synthesizer.
 *
 * Loaded as a module worker from {@link ./tts-worker-host.ts}. Keeps the heavy
 * `moonshine_text_to_speech` work off the page's main thread.
 */

import type { MoonshineModule, RawTextToSpeech } from './module.js';
import type { TtsWorkerRequest, TtsWorkerResponse } from './tts-worker-protocol.js';

type EmscriptenFactory = (opts?: Record<string, unknown>) => Promise<MoonshineModule>;

let modulePromise: Promise<MoonshineModule> | undefined;
let raw: RawTextToSpeech | undefined;

const SUPPRESSED_STDERR = /Unknown CPU vendor\. cpuinfo_vendor value:/;

function printErr(...args: unknown[]): void {
  if (typeof args[0] === 'string' && SUPPRESSED_STDERR.test(args[0])) return;
  console.error(...args);
}

async function loadModule(wasmBaseUrl: string): Promise<MoonshineModule> {
  if (!modulePromise) {
    modulePromise = (async () => {
      // @ts-expect-error generated at build time
      const mod = await import('./moonshine.mjs');
      const factory = (mod.default ?? mod) as EmscriptenFactory;
      return factory({
        printErr,
        locateFile: (path: string) => new URL(path, wasmBaseUrl).href,
      });
    })();
  }
  return modulePromise;
}

function reply(msg: TtsWorkerResponse, transfer?: Transferable[]): void {
  if (transfer?.length) {
    postMessage(msg, transfer);
  } else {
    postMessage(msg);
  }
}

function fail(id: number, err: unknown): void {
  const message = err instanceof Error ? err.message : String(err);
  reply({ type: 'error', id, message });
}

onmessage = (event: MessageEvent<TtsWorkerRequest>) => {
  const msg = event.data;
  void (async () => {
    try {
      switch (msg.type) {
        case 'setEngine': {
          const mod = await loadModule(msg.wasmBaseUrl);
          if (!mod.TextToSpeech) {
            throw new Error('This Moonshine WASM build was compiled without TTS support.');
          }
          const next = new mod.TextToSpeech(
            msg.language,
            msg.keys,
            msg.buffers,
            msg.optionNames,
            msg.optionValues,
          );
          raw?.close();
          raw = next;
          reply({ type: 'ok', id: msg.id });
          break;
        }
        case 'synthesize': {
          if (!raw) {
            throw new Error('Call load() before say().');
          }
          const result = raw.say(msg.text);
          // Copy into a detached buffer so the host can take ownership.
          const copy = result.audio.slice().buffer;
          reply(
            {
              type: 'synthesized',
              id: msg.id,
              sampleRate: result.sampleRate,
              audioBuffer: copy,
            },
            [copy],
          );
          break;
        }
        case 'close': {
          raw?.close();
          raw = undefined;
          reply({ type: 'ok', id: msg.id });
          break;
        }
        default: {
          const neverMsg: never = msg;
          throw new Error(`Unknown worker message: ${(neverMsg as { type: string }).type}`);
        }
      }
    } catch (err) {
      fail(msg.id, err);
    }
  })();
};
