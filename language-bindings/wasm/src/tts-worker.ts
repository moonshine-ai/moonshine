/**
 * Web Worker that owns a Moonshine TTS synthesizer.
 *
 * Loaded as a module worker from {@link ./tts-worker-host.ts}. Keeps the heavy
 * `moonshine_text_to_speech` work off the page's main thread.
 */

import { loadMoonshineModule } from './module.js';
import type { RawTextToSpeech } from './module.js';
import type { TtsWorkerRequest, TtsWorkerResponse } from './tts-worker-protocol.js';

let raw: RawTextToSpeech | undefined;

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
          // loadMoonshineModule applies the cross-origin pthread blob workaround
          // when this worker (or the binding) was loaded from a CDN.
          const mod = await loadMoonshineModule({
            locateFile: (path) => new URL(path, msg.wasmBaseUrl).href,
          });
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
