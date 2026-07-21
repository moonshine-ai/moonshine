/**
 * @moonshine-ai/moonshine-wasm — idiomatic WebAssembly binding for Moonshine
 * Voice. Mirrors the object model of the Python, Swift, and Android bindings.
 *
 * Phase 1 (STT): {@link Transcriber}, {@link Stream}, {@link MicrophoneTranscriber}.
 * Phase 2 (TTS): {@link TextToSpeech}, {@link GraphemeToPhonemizer}.
 * Phase 3 (Intent + dialog): {@link IntentRecognizer}, {@link DialogFlow}.
 */
export { loadMoonshineModule, resetMoonshineModule, type LoadModuleOptions, type MoonshineModule, } from './module.js';
export { MoonshineError, MoonshineUnknownError, MoonshineInvalidHandleError, MoonshineInvalidArgumentError, MoonshineDownloadError, MoonshineErrorCode, } from './errors.js';
export { ModelArch, EmbeddingModelArch, TranscribeFlags, modelArchToString, stringToModelArch, } from './enums.js';
export type { WordTiming, SpeakerSpan, TranscriptLine, Transcript, IntentMatch, TtsSynthesisResult, } from './types.js';
export type { TranscriptEvent, TranscriptEventListener, LineStarted, LineUpdated, LineTextChanged, LineSpeakersChanged, LineCompleted, TranscriptErrorEvent, } from './events.js';
export { AssetDownloader, type AssetDownloaderOptions, type DownloadedAsset, } from './asset-downloader.js';
export { Transcriber, type TranscriberLoadOptions, type TranscriberFromBytes, type TranscriberFromCatalog, } from './transcriber.js';
export { Stream } from './stream.js';
export { MicrophoneTranscriber, type MicrophoneTranscriberOptions, } from './microphone-transcriber.js';
export { TextToSpeech, type TextToSpeechOptions, type TtsFromAssets, type TtsFromCatalog, } from './text-to-speech.js';
export { GraphemeToPhonemizer, type GraphemeToPhonemizerOptions, } from './grapheme-to-phonemizer.js';
export { IntentRecognizer, type IntentRecognizerOptions, type IntentPhrase, } from './intent-recognizer.js';
export { DialogFlow, Dialog, DialogCancelled, DialogRestart, InputMode, spellOut, type Prompt, type Say, type Ask, type Confirm, type Choose, type FlowFn, type GlobalHandler, type DialogFlowOptions, } from './dialog-flow.js';
//# sourceMappingURL=index.d.ts.map