/**
 * Plain, readonly value types returned by the binding. These mirror the C
 * structs in `core/moonshine-c-api.h` and the value types the Python/Swift/
 * Android bindings expose, but as idiomatic JS objects (no heap views).
 */
// --- Internal helpers: normalize raw embind objects into the clean types. ---
/** Reads an embind value that may be a native array or a bound vector. */
function readVector(raw) {
    if (raw == null)
        return [];
    if (Array.isArray(raw))
        return raw;
    // Embind register_vector<> object exposes size()/get().
    const vec = raw;
    if (typeof vec.size === 'function' && typeof vec.get === 'function') {
        const out = [];
        const n = vec.size();
        for (let i = 0; i < n; i++)
            out.push(vec.get(i));
        // Bound vectors must be explicitly freed to avoid leaking heap memory.
        raw.delete?.();
        return out;
    }
    return [];
}
/** @internal Normalizes a raw embind transcript into the public shape. */
export function normalizeTranscript(raw) {
    const rawLines = readVector(raw?.lines);
    const lines = rawLines.map((line) => ({
        text: line.text ?? '',
        startTime: line.startTime ?? 0,
        duration: line.duration ?? 0,
        id: line.id ?? 0,
        isComplete: !!line.isComplete,
        isUpdated: !!line.isUpdated,
        isNew: !!line.isNew,
        hasTextChanged: !!line.hasTextChanged,
        haveSpeakersChanged: !!line.haveSpeakersChanged,
        lastTranscriptionLatencyMs: line.lastTranscriptionLatencyMs ?? 0,
        words: readVector(line.words).map((w) => ({
            text: w.text ?? '',
            start: w.start ?? 0,
            end: w.end ?? 0,
            confidence: w.confidence ?? 0,
        })),
        speakerSpans: readVector(line.speakerSpans).map((s) => ({
            startTime: s.startTime ?? 0,
            duration: s.duration ?? 0,
            speakerId: s.speakerId ?? 0,
            speakerIndex: s.speakerIndex ?? 0,
            startChar: s.startChar ?? 0,
            endChar: s.endChar ?? 0,
        })),
    }));
    return { lines };
}
/** @internal Normalizes a raw embind intent-match vector. */
export function normalizeIntentMatches(raw) {
    return readVector(raw).map((m) => ({
        canonicalPhrase: m.canonicalPhrase ?? '',
        similarity: m.similarity ?? 0,
    }));
}
//# sourceMappingURL=types.js.map