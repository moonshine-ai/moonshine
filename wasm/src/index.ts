/**
 * @moonshine-ai/moonshine-wasm — WebAssembly binding for Moonshine Voice.
 * Mirrors the object model of the Python, Swift, and Android bindings.
 *
 * The three entry points are {@link AgentFlow} for voice interfaces,
 * {@link MicTranscriber} for live transcription, and {@link TextToSpeech} for
 * speech synthesis and voice cloning. Each is constructed with `new`,
 * configured with chainable setters, and prepared with a single `await load()`.
 *
 * {@link Transcriber}, {@link Stream}, {@link GraphemeToPhonemizer}, and
 * {@link AssetDownloader} are the lower-level pieces those are built from, for
 * applications that need them directly.
 */

export {
  loadMoonshineModule,
  resetMoonshineModule,
  type LoadModuleOptions,
  type MoonshineModule,
  type RawSpeechClip,
} from './module.js';

export {
  MoonshineError,
  MoonshineUnknownError,
  MoonshineInvalidHandleError,
  MoonshineInvalidArgumentError,
  MoonshineDownloadError,
  MoonshineErrorCode,
} from './errors.js';

export {
  ModelArch,
  TranscribeFlags,
  modelArchToString,
  stringToModelArch,
} from './enums.js';

export type {
  WordTiming,
  SpeakerSpan,
  TranscriptLine,
  Transcript,
  TtsSynthesisResult,
} from './types.js';

export type {
  TranscriptEvent,
  TranscriptEventListener,
  LineStarted,
  LineUpdated,
  LineTextChanged,
  LineSpeakersChanged,
  LineCompleted,
  TranscriptErrorEvent,
} from './events.js';

export {
  AssetDownloader,
  type AssetDownloaderOptions,
  type DownloadedAsset,
} from './asset-downloader.js';

export {
  Transcriber,
  type TranscriberLoadOptions,
  type TranscriberFromBytes,
  type TranscriberFromFiles,
  type TranscriberFromCatalog,
  type TranscriberFromUrlsOptions,
} from './transcriber.js';
export { Stream, DEFAULT_UPDATE_INTERVAL } from './stream.js';
export { MicTranscriber, type ProgressCallback } from './mic-transcriber.js';

// Only usable if the module was built with TTS support.
export {
  TextToSpeech,
  type CloneSource,
  type TtsVoiceEntry,
  type TtsVoicesOptions,
} from './text-to-speech.js';
export {
  VoiceClone,
  extractSpeechClip,
  type VoiceCloneOptions,
} from './voice-clone.js';
export {
  GraphemeToPhonemizer,
  type GraphemeToPhonemizerOptions,
} from './grapheme-to-phonemizer.js';

export {
  AgentFlow,
  Dialog,
  DialogCancelled,
  DialogRestart,
  DialogNoMatch,
  spellOut,
  type AskOptions,
  type ConfirmOptions,
  type FlowFn,
  type GlobalHandler,
  type UnmatchedHandler,
} from './agent-flow.js';
