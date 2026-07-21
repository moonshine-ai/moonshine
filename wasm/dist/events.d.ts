/**
 * Transcript event model, matching the C++/Python/Swift event semantics: a
 * streaming transcript is diffed into discrete line events that a
 * {@link TranscriptEventListener} can react to.
 *
 * The C++ event model is the source of truth; the {@link diffTranscript}
 * helper reproduces its LineStarted / LineTextChanged / LineCompleted /
 * LineSpeakersChanged sequencing from successive transcript snapshots.
 */
import type { Transcript, TranscriptLine } from './types.js';
export interface TranscriptEvent {
    readonly line: TranscriptLine;
}
export interface LineStarted extends TranscriptEvent {
}
export interface LineUpdated extends TranscriptEvent {
}
export interface LineTextChanged extends TranscriptEvent {
}
export interface LineSpeakersChanged extends TranscriptEvent {
}
export interface LineCompleted extends TranscriptEvent {
}
export interface TranscriptErrorEvent {
    readonly error: Error;
    readonly line?: TranscriptLine;
}
/**
 * Listener interface mirroring the other bindings. Implement the callbacks you
 * care about; all are optional.
 */
export interface TranscriptEventListener {
    onLineStarted?(event: LineStarted): void;
    onLineUpdated?(event: LineUpdated): void;
    onLineTextChanged?(event: LineTextChanged): void;
    onLineSpeakersChanged?(event: LineSpeakersChanged): void;
    onLineCompleted?(event: LineCompleted): void;
    onError?(event: TranscriptErrorEvent): void;
}
/** Tracks per-line state across snapshots so we only emit real transitions. */
export interface DiffState {
    /** Line id -> last seen text. */
    seenText: Map<number, string>;
    /** Line ids we've emitted `onLineStarted` for. */
    started: Set<number>;
    /** Line ids we've emitted `onLineCompleted` for. */
    completed: Set<number>;
}
export declare function createDiffState(): DiffState;
/**
 * Diffs a new transcript snapshot against prior state and dispatches the
 * resulting events to `listeners`. Returns nothing; mutates `state`.
 */
export declare function diffTranscript(transcript: Transcript, state: DiffState, listeners: readonly TranscriptEventListener[]): void;
export declare function dispatchError(listeners: readonly TranscriptEventListener[], error: Error, line?: TranscriptLine): void;
//# sourceMappingURL=events.d.ts.map