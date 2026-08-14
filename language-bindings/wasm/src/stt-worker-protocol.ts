/**
 * Message protocol between {@link SttWorkerHost} and the STT Web Worker.
 *
 * The worker owns {@link Transcriber} and {@link Stream} so live transcription
 * can run off the main thread. Sync {@link Stream.transcribe} on the page stays
 * on the main thread and is out of scope here.
 */

import type { TranscriptLine } from './types.js';

export type SttLoadSource =
  | { kind: 'urls'; files: Record<string, string> }
  | { kind: 'catalog'; language: string; includeSpelling?: boolean };

export type SttWorkerRequest =
  | {
      type: 'loadTranscriber';
      id: number;
      transcriberId: string;
      modelArch: number;
      optionNames: string[];
      optionValues: string[];
      source: SttLoadSource;
      /** Base URL for moonshine.wasm / pthread workers, ending with `/`. */
      wasmBaseUrl: string;
    }
  | {
      type: 'createStream';
      id: number;
      transcriberId: string;
      streamId: string;
      flags?: number;
      updateInterval?: number;
      /** Higher values transcribe first when several streams are due. */
      priority?: number;
    }
  | { type: 'start'; id: number; streamId: string }
  | {
      type: 'addAudio';
      streamId: string;
      sampleRate: number;
      /** Detached ArrayBuffer of Float32 PCM; transferred from the host. */
      audioBuffer: ArrayBuffer;
      /** When false, audio is buffered without scheduling a pass (stop will ForceUpdate). */
      enqueuePass?: boolean;
    }
  | { type: 'stop'; id: number; streamId: string }
  | { type: 'closeStream'; id: number; streamId: string }
  | { type: 'closeTranscriber'; id: number; transcriberId: string }
  | { type: 'close'; id: number };

export type SttLineEventName =
  | 'onLineStarted'
  | 'onLineUpdated'
  | 'onLineTextChanged'
  | 'onLineSpeakersChanged'
  | 'onLineCompleted';

export type SttWorkerRpcResponse =
  | { type: 'ok'; id: number }
  | { type: 'error'; id: number; message: string };

export type SttWorkerEvent =
  | {
      type: 'progress';
      transcriberId: string;
      loaded: number;
      total?: number;
      file: string;
    }
  | { type: 'event'; streamId: string; name: SttLineEventName; line: TranscriptLine }
  | { type: 'errorEvent'; streamId: string; message: string }
  | {
      type: 'ingested';
      streamId: string;
      seconds: number;
    }
  | { type: 'pass'; streamId: string; ms: number };

export type SttWorkerResponse = SttWorkerRpcResponse | SttWorkerEvent;
