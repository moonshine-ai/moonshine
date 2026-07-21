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

export interface LineStarted extends TranscriptEvent {}
export interface LineUpdated extends TranscriptEvent {}
export interface LineTextChanged extends TranscriptEvent {}
export interface LineSpeakersChanged extends TranscriptEvent {}
export interface LineCompleted extends TranscriptEvent {}
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

export function createDiffState(): DiffState {
  return { seenText: new Map(), started: new Set(), completed: new Set() };
}

/**
 * Diffs a new transcript snapshot against prior state and dispatches the
 * resulting events to `listeners`. Returns nothing; mutates `state`.
 */
export function diffTranscript(
  transcript: Transcript,
  state: DiffState,
  listeners: readonly TranscriptEventListener[],
): void {
  for (const line of transcript.lines) {
    const id = line.id;

    if (!state.started.has(id)) {
      state.started.add(id);
      state.seenText.set(id, line.text);
      dispatch(listeners, (l) => l.onLineStarted?.({ line }));
    } else {
      const prevText = state.seenText.get(id);
      if (prevText !== line.text) {
        state.seenText.set(id, line.text);
        dispatch(listeners, (l) => l.onLineTextChanged?.({ line }));
      }
      if (line.isUpdated) {
        dispatch(listeners, (l) => l.onLineUpdated?.({ line }));
      }
    }

    if (line.haveSpeakersChanged) {
      dispatch(listeners, (l) => l.onLineSpeakersChanged?.({ line }));
    }

    if (line.isComplete && !state.completed.has(id)) {
      state.completed.add(id);
      dispatch(listeners, (l) => l.onLineCompleted?.({ line }));
    }
  }
}

export function dispatchError(
  listeners: readonly TranscriptEventListener[],
  error: Error,
  line?: TranscriptLine,
): void {
  dispatch(listeners, (l) => l.onError?.({ error, line }));
}

function dispatch(
  listeners: readonly TranscriptEventListener[],
  fn: (l: TranscriptEventListener) => void,
): void {
  for (const listener of listeners) {
    try {
      fn(listener);
    } catch {
      // A misbehaving listener must not break transcript delivery.
    }
  }
}
