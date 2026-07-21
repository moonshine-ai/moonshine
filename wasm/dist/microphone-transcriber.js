/**
 * Live microphone transcription, mirroring the `MicrophoneTranscriber` helpers
 * in the other bindings. Captures mic audio via WebAudio, downmixes to mono +
 * resamples to 16 kHz, and feeds a streaming {@link Stream}.
 *
 * Uses an AudioWorklet when available (off the main thread), falling back to a
 * ScriptProcessorNode. All model work still happens through the WASM binding.
 */
import { TranscribeFlags } from './enums.js';
import { Transcriber } from './transcriber.js';
const TARGET_SAMPLE_RATE = 16000;
export class MicrophoneTranscriber {
    transcriber;
    ownsTranscriber;
    flags;
    listeners;
    audioConstraints;
    mediaStream;
    audioContext;
    sourceNode;
    workletNode;
    scriptNode;
    stream;
    running = false;
    constructor(transcriber, ownsTranscriber, options) {
        this.transcriber = transcriber;
        this.ownsTranscriber = ownsTranscriber;
        this.flags = options.flags ?? TranscribeFlags.None;
        this.listeners = options.listeners ?? [];
        this.audioConstraints = options.audioConstraints ?? true;
    }
    /**
     * Creates a MicrophoneTranscriber. Either pass an existing `transcriber`, or
     * pass {@link TranscriberLoadOptions} to load one automatically.
     */
    static async load(options) {
        if (options.transcriber) {
            return new MicrophoneTranscriber(options.transcriber, false, options);
        }
        const transcriber = await Transcriber.load(options);
        return new MicrophoneTranscriber(transcriber, true, options);
    }
    addListener(listener) {
        this.listeners.push(listener);
        this.stream?.addListener(listener);
    }
    /** Starts capturing and transcribing. Resolves once audio is flowing. */
    async start() {
        if (this.running)
            return;
        this.running = true;
        this.mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: this.audioConstraints,
        });
        this.audioContext = new AudioContext();
        const inputSampleRate = this.audioContext.sampleRate;
        this.sourceNode = this.audioContext.createMediaStreamSource(this.mediaStream);
        this.stream = this.transcriber.createStream({ flags: this.flags });
        for (const listener of this.listeners)
            this.stream.addListener(listener);
        this.stream.start();
        const onChunk = (chunk) => {
            if (!this.running || !this.stream)
                return;
            const resampled = resampleTo16k(chunk, inputSampleRate);
            this.stream.addAudio(resampled, TARGET_SAMPLE_RATE, this.flags);
            this.stream.transcribe(this.flags);
        };
        if (this.audioContext.audioWorklet) {
            await this.setupWorklet(onChunk);
        }
        else {
            this.setupScriptProcessor(onChunk);
        }
    }
    /** Stops capture, flushes a final transcript, and releases audio resources. */
    async stop() {
        if (!this.running)
            return;
        this.running = false;
        this.workletNode?.disconnect();
        this.scriptNode?.disconnect();
        this.sourceNode?.disconnect();
        this.mediaStream?.getTracks().forEach((t) => t.stop());
        this.stream?.stop();
        await this.audioContext?.close();
        this.workletNode = undefined;
        this.scriptNode = undefined;
        this.sourceNode = undefined;
        this.mediaStream = undefined;
        this.audioContext = undefined;
    }
    /** Releases the stream (and the transcriber, if this instance loaded it). */
    close() {
        this.stream?.close();
        this.stream = undefined;
        if (this.ownsTranscriber)
            this.transcriber.close();
    }
    async setupWorklet(onChunk) {
        const ctx = this.audioContext;
        const url = URL.createObjectURL(new Blob([CAPTURE_WORKLET_SOURCE], { type: 'application/javascript' }));
        try {
            await ctx.audioWorklet.addModule(url);
        }
        finally {
            URL.revokeObjectURL(url);
        }
        this.workletNode = new AudioWorkletNode(ctx, 'moonshine-capture');
        this.workletNode.port.onmessage = (event) => onChunk(event.data);
        this.sourceNode.connect(this.workletNode);
        // Keep the graph alive without producing output.
        this.workletNode.connect(ctx.destination);
    }
    setupScriptProcessor(onChunk) {
        const ctx = this.audioContext;
        this.scriptNode = ctx.createScriptProcessor(4096, 1, 1);
        this.scriptNode.onaudioprocess = (event) => {
            onChunk(new Float32Array(event.inputBuffer.getChannelData(0)));
        };
        this.sourceNode.connect(this.scriptNode);
        this.scriptNode.connect(ctx.destination);
    }
}
/** Simple linear resampler to 16 kHz mono. */
function resampleTo16k(input, inputRate) {
    if (inputRate === TARGET_SAMPLE_RATE)
        return input;
    const ratio = inputRate / TARGET_SAMPLE_RATE;
    const outLength = Math.floor(input.length / ratio);
    const output = new Float32Array(outLength);
    for (let i = 0; i < outLength; i++) {
        const pos = i * ratio;
        const idx = Math.floor(pos);
        const frac = pos - idx;
        const a = input[idx] ?? 0;
        const b = input[idx + 1] ?? a;
        output[i] = a + (b - a) * frac;
    }
    return output;
}
/** AudioWorklet that forwards mono input frames to the main thread. */
const CAPTURE_WORKLET_SOURCE = `
class MoonshineCaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (input && input[0]) {
      // Copy: the underlying buffer is reused by the engine.
      this.port.postMessage(new Float32Array(input[0]));
    }
    return true;
  }
}
registerProcessor('moonshine-capture', MoonshineCaptureProcessor);
`;
//# sourceMappingURL=microphone-transcriber.js.map