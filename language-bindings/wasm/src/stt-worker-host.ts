/**
 * Main-thread host for {@link ./stt-worker.ts}.
 *
 * Spawns a module worker when the environment supports it (browsers). Node
 * tests and {@link Stream.transcribe} stay on the main thread.
 */

import type { ModelArch } from './enums.js';
import type { TranscriptEventListener } from './events.js';
import type {
  SttLoadSource,
  SttWorkerEvent,
  SttWorkerRequest,
  SttWorkerResponse,
  SttWorkerRpcResponse,
} from './stt-worker-protocol.js';

export interface SttLoadTranscriberConfig {
  transcriberId: string;
  modelArch: ModelArch;
  options?: Record<string, string>;
  source: SttLoadSource;
}

type Pending = {
  resolve: (value: SttWorkerRpcResponse) => void;
  reject: (reason: Error) => void;
};

function isRpcResponse(msg: SttWorkerResponse): msg is SttWorkerRpcResponse {
  return msg.type === 'ok' || msg.type === 'error';
}

/** True when we can run STT inference off the main thread. */
export function sttWorkerSupported(): boolean {
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

/**
 * Spawns a module Worker for `scriptUrl`, using a same-origin blob bridge when
 * the script is cross-origin. Browsers reject `new Worker(crossOriginUrl)` even
 * when the URL is CORS-enabled (same restriction that hits Emscripten pthreads).
 */
function spawnModuleWorker(scriptUrl: URL): Worker {
  const href = scriptUrl.href;
  if (
    typeof location !== 'undefined' &&
    new URL(href).origin !== location.origin
  ) {
    const bridge = `import ${JSON.stringify(href)};`;
    const blobUrl = URL.createObjectURL(
      new Blob([bridge], { type: 'text/javascript' }),
    );
    return new Worker(blobUrl, { type: 'module' });
  }
  return new Worker(scriptUrl, { type: 'module' });
}

export class SttWorkerHost {
  private worker: Worker;
  private nextId = 1;
  private pending = new Map<number, Pending>();
  private readonly wasmBaseUrl: string;
  private closed = false;
  private readonly listeners = new Map<string, TranscriptEventListener>();
  /** Audio posted to a stream that the worker has not yet ingested. */
  private readonly inflightSeconds = new Map<string, number>();
  onProgress?: (
    transcriberId: string,
    loaded: number,
    total: number | undefined,
    file: string,
  ) => void;
  onPass?: (streamId: string, ms: number) => void;

  constructor(wasmBaseUrl: string = moonshineWasmBaseUrl()) {
    this.wasmBaseUrl = wasmBaseUrl;
    this.worker = spawnModuleWorker(new URL('./stt-worker.js', import.meta.url));
    this.worker.onmessage = (event: MessageEvent<SttWorkerResponse>) => {
      const msg = event.data;
      if (isRpcResponse(msg)) {
        const slot = this.pending.get(msg.id);
        if (!slot) return;
        this.pending.delete(msg.id);
        if (msg.type === 'error') {
          slot.reject(new Error(msg.message));
        } else {
          slot.resolve(msg);
        }
        return;
      }
      this.dispatchEvent(msg);
    };
    this.worker.onerror = (event) => {
      const error = new Error(event.message || 'STT worker failed');
      for (const slot of this.pending.values()) {
        slot.reject(error);
      }
      this.pending.clear();
    };
  }

  private dispatchEvent(msg: SttWorkerEvent): void {
    switch (msg.type) {
      case 'progress':
        this.onProgress?.(msg.transcriberId, msg.loaded, msg.total, msg.file);
        return;
      case 'event':
        this.listeners.get(msg.streamId)?.[msg.name]?.({ line: msg.line });
        return;
      case 'errorEvent':
        this.listeners.get(msg.streamId)?.onError?.({
          error: new Error(msg.message),
        });
        return;
      case 'ingested': {
        const left = (this.inflightSeconds.get(msg.streamId) ?? 0) - msg.seconds;
        this.inflightSeconds.set(msg.streamId, Math.max(0, left));
        return;
      }
      case 'pass':
        this.onPass?.(msg.streamId, msg.ms);
        return;
    }
  }

  private request(
    payload: SttWorkerRequest extends infer R
      ? R extends { id: number }
        ? Omit<R, 'id'>
        : never
      : never,
  ): Promise<SttWorkerRpcResponse> {
    if (this.closed) {
      return Promise.reject(new Error('STT worker is closed.'));
    }
    const id = this.nextId++;
    const message = { ...payload, id } as SttWorkerRequest;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage(message);
    });
  }

  /** Downloads and constructs a transcriber inside the worker. */
  async loadTranscriber(config: SttLoadTranscriberConfig): Promise<void> {
    const optionNames = config.options ? Object.keys(config.options) : [];
    const optionValues = optionNames.map((name) => config.options![name]!);
    const response = await this.request({
      type: 'loadTranscriber',
      transcriberId: config.transcriberId,
      modelArch: config.modelArch,
      optionNames,
      optionValues,
      source: config.source,
      wasmBaseUrl: this.wasmBaseUrl,
    });
    if (response.type !== 'ok') {
      throw new Error('Unexpected STT worker response to loadTranscriber.');
    }
  }

  async createStream(
    transcriberId: string,
    streamId: string,
    options: { flags?: number; updateInterval?: number; priority?: number } = {},
  ): Promise<void> {
    this.inflightSeconds.set(streamId, 0);
    const response = await this.request({
      type: 'createStream',
      transcriberId,
      streamId,
      flags: options.flags,
      updateInterval: options.updateInterval,
      priority: options.priority,
    });
    if (response.type !== 'ok') {
      throw new Error('Unexpected STT worker response to createStream.');
    }
  }

  setListener(streamId: string, listener: TranscriptEventListener): void {
    this.listeners.set(streamId, listener);
  }

  async start(streamId: string): Promise<void> {
    const response = await this.request({ type: 'start', streamId });
    if (response.type !== 'ok') {
      throw new Error('Unexpected STT worker response to start.');
    }
  }

  /**
   * Posts PCM to the worker without waiting for a pass. The buffer is copied
   * and transferred; the caller can keep using `audio`.
   */
  addAudio(
    streamId: string,
    audio: Float32Array,
    sampleRate: number,
    options: { transcribe?: boolean } = {},
  ): void {
    if (this.closed) {
      throw new Error('STT worker is closed.');
    }
    const copy = audio.slice();
    const seconds = sampleRate > 0 ? copy.length / sampleRate : 0;
    this.inflightSeconds.set(
      streamId,
      (this.inflightSeconds.get(streamId) ?? 0) + seconds,
    );
    const message: SttWorkerRequest = {
      type: 'addAudio',
      streamId,
      sampleRate,
      audioBuffer: copy.buffer,
      enqueuePass: options.transcribe !== false,
    };
    this.worker.postMessage(message, [copy.buffer]);
  }

  /** Seconds of audio posted to the worker that it has not yet ingested. */
  queuedSeconds(streamId: string): number {
    return this.inflightSeconds.get(streamId) ?? 0;
  }

  async stop(streamId: string): Promise<void> {
    const response = await this.request({ type: 'stop', streamId });
    if (response.type !== 'ok') {
      throw new Error('Unexpected STT worker response to stop.');
    }
  }

  async closeStream(streamId: string): Promise<void> {
    this.listeners.delete(streamId);
    this.inflightSeconds.delete(streamId);
    const response = await this.request({ type: 'closeStream', streamId });
    if (response.type !== 'ok') {
      throw new Error('Unexpected STT worker response to closeStream.');
    }
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    const id = this.nextId++;
    try {
      this.worker.postMessage({ type: 'close', id } satisfies SttWorkerRequest);
    } catch {
      /* worker may already be gone */
    }
    for (const slot of this.pending.values()) {
      slot.reject(new Error('STT worker closed.'));
    }
    this.pending.clear();
    this.listeners.clear();
    this.inflightSeconds.clear();
    this.worker.terminate();
  }
}
