/**
 * Main-thread host for {@link ./tts-worker.ts}.
 *
 * Spawns a module worker when the environment supports it (browsers). Node
 * tests fall back to main-thread synthesis inside {@link TextToSpeech}.
 */

import type { TtsWorkerRequest, TtsWorkerResponse } from './tts-worker-protocol.js';
import type { TtsSynthesisResult } from './types.js';

export interface TtsWorkerEngineConfig {
  language: string;
  keys: string[];
  buffers: Uint8Array[];
  optionNames: string[];
  optionValues: string[];
}

type Pending = {
  resolve: (value: TtsWorkerResponse) => void;
  reject: (reason: Error) => void;
};

/** True when we can run TTS synthesis off the main thread. */
export function ttsWorkerSupported(): boolean {
  // Node 18+ may expose `Worker` (worker_threads) with a different runtime;
  // only enable the off-thread path in a document / window environment.
  return (
    typeof window !== 'undefined' &&
    typeof document !== 'undefined' &&
    typeof Worker !== 'undefined' &&
    typeof URL !== 'undefined'
  );
}

/**
 * Base URL for files next to this module (`moonshine.wasm`, the worker, …).
 * Trailing slash included so `new URL(path, base)` resolves correctly.
 */
export function moonshineWasmBaseUrl(): string {
  return new URL('./', import.meta.url).href;
}

export class TtsWorkerHost {
  private worker: Worker;
  private nextId = 1;
  private pending = new Map<number, Pending>();
  private readonly wasmBaseUrl: string;
  private closed = false;

  constructor(wasmBaseUrl: string = moonshineWasmBaseUrl()) {
    this.wasmBaseUrl = wasmBaseUrl;
    this.worker = new Worker(new URL('./tts-worker.js', import.meta.url), {
      type: 'module',
    });
    this.worker.onmessage = (event: MessageEvent<TtsWorkerResponse>) => {
      const msg = event.data;
      const slot = this.pending.get(msg.id);
      if (!slot) return;
      this.pending.delete(msg.id);
      if (msg.type === 'error') {
        slot.reject(new Error(msg.message));
      } else {
        slot.resolve(msg);
      }
    };
    this.worker.onerror = (event) => {
      const error = new Error(event.message || 'TTS worker failed');
      for (const slot of this.pending.values()) {
        slot.reject(error);
      }
      this.pending.clear();
    };
  }

  private request(
    payload: TtsWorkerRequest extends infer R
      ? R extends { id: number }
        ? Omit<R, 'id'>
        : never
      : never,
  ): Promise<TtsWorkerResponse> {
    if (this.closed) {
      return Promise.reject(new Error('TTS worker is closed.'));
    }
    const id = this.nextId++;
    const message = { ...payload, id } as TtsWorkerRequest;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage(message);
    });
  }

  /** (Re)creates the worker-side synthesizer from asset buffers. */
  async setEngine(config: TtsWorkerEngineConfig): Promise<void> {
    const response = await this.request({
      type: 'setEngine',
      language: config.language,
      keys: config.keys,
      buffers: config.buffers,
      optionNames: config.optionNames,
      optionValues: config.optionValues,
      wasmBaseUrl: this.wasmBaseUrl,
    });
    if (response.type !== 'ok') {
      throw new Error('Unexpected TTS worker response to setEngine.');
    }
  }

  /** Runs `moonshine_text_to_speech` on the worker. */
  async synthesize(text: string): Promise<TtsSynthesisResult> {
    const response = await this.request({ type: 'synthesize', text });
    if (response.type !== 'synthesized') {
      throw new Error('Unexpected TTS worker response to synthesize.');
    }
    return {
      audio: new Float32Array(response.audioBuffer),
      sampleRate: response.sampleRate,
    };
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    const id = this.nextId++;
    try {
      this.worker.postMessage({ type: 'close', id } satisfies TtsWorkerRequest);
    } catch {
      /* worker may already be gone */
    }
    for (const slot of this.pending.values()) {
      slot.reject(new Error('TTS worker closed.'));
    }
    this.pending.clear();
    this.worker.terminate();
  }
}
