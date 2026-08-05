/**
 * Message protocol between {@link TextToSpeech} and the TTS Web Worker.
 *
 * The worker owns the WASM synthesizer so {@link TextToSpeech.say} can run
 * synthesis off the main thread. Sync {@link TextToSpeech.synthesize} stays on
 * the main thread and is out of scope here.
 */

export type TtsWorkerRequest =
  | {
      type: 'setEngine';
      id: number;
      language: string;
      keys: string[];
      /** Parallel to keys; structured-cloned (not transferred) so the host keeps a copy. */
      buffers: Uint8Array[];
      optionNames: string[];
      optionValues: string[];
      /** Base URL for moonshine.wasm / pthread workers, ending with `/`. */
      wasmBaseUrl: string;
    }
  | {
      type: 'synthesize';
      id: number;
      text: string;
    }
  | {
      type: 'close';
      id: number;
    };

export type TtsWorkerResponse =
  | { type: 'ok'; id: number }
  | {
      type: 'synthesized';
      id: number;
      sampleRate: number;
      /** Detached ArrayBuffer of Float32 PCM; transferred from the worker. */
      audioBuffer: ArrayBuffer;
    }
  | { type: 'error'; id: number; message: string };
