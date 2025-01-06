import MoonshineModel from "./model";
import { MoonshineSettings } from "./constants";

interface MoonshineTranscriberCallbacks {
    onModelLoadStarted: () => any;

    onTranscribeStarted: () => any;

    onTranscribeStopped: () => any;

    onTranscriptionUpdated: (text: string | undefined) => any;
}

const defaultTranscriberCallbacks: MoonshineTranscriberCallbacks = {
    onModelLoadStarted() {
        console.log("MoonshineTranscriber.onModelLoadStarted()");
    },
    onTranscribeStarted: function () {
        console.log("MoonshineTranscriber.onTranscribeStarted()");
    },
    onTranscribeStopped: function () {
        console.log("MoonshineTranscriber.onTranscribeStopped()");
    },
    onTranscriptionUpdated: function (text: string | undefined) {
        console.log(
            "MoonshineTranscriber.onTranscriptionUpdated(" + text + ")"
        );
    },
};

/**
 * An interface for accessing the user's microphone, loading the moonshine model, streaming audio, and generating transcriptions.
 */
class MoonshineTranscriber {
    private static mediaRecorder: MediaRecorder | undefined = undefined;
    private static model: MoonshineModel | undefined = undefined;
    private static audioContext: AudioContext | undefined = undefined;
    private static modelURL: string;
    private callbacks: MoonshineTranscriberCallbacks;

    /**
     * Creates a transcriber for managing audio streaming from the mic and transcription with a {@link MoonshineModel}.
     *
     * @param callbacks - A set of optional callbacks used to trigger page behavior at different steps of the
     * transcription lifecycle. The following callbacks can be defined:
     *
     * onModelLoadStarted() - called when the model begins to load (or download, if hosted elsewhere)
     * 
     * onTranscribeStarted() - called once when transcription starts
     * 
     * onTranscribeStopped() - called once when transcription stops
     * 
     * onTranscriptionUpdated(text) - called every {@link MoonshineSettings.FRAME_SIZE} milliseconds while
     * transcription is active
     *
     * @param modelURL -the URL that the underlying {@link MoonshineModel} weights should be loaded from,
     * relative to {@link MoonshineSettings.BASE_ASSET_PATH}
     *
     * @example
     * This basic example demonstrates the use of the transcriber with custom callbacks:
     * 
     * ``` ts
     * import MoonshineTranscriber from "@usefulsensors/moonshine";
     * 
     * var transcriber = new MoonshineTranscriber(
     *      {
     *          onModelLoadStarted() {
     *              console.log("onModelLoadStarted()");
     *          },
     *          onTranscribeStarted() {
     *              console.log("onTranscribeStarted()");
     *          },
     *          onTranscribeStopped() {
     *              console.log("onTranscribeStopped()");
     *          },
     *          onTranscriptionUpdated(text: string | undefined) {
     *              console.log(
     *                  "onTranscriptionUpdated(" + text + ")"
     *              );
     *          },
     *      },
     *      "model/tiny"
     * );
     *
     * transcriber.start();
     * ```
     */
    public constructor(
        callbacks: Partial<MoonshineTranscriberCallbacks> = {},
        modelURL: string
    ) {
        this.callbacks = { ...defaultTranscriberCallbacks, ...callbacks };
        MoonshineTranscriber.modelURL = modelURL;
    }

    /**
     * Starts transcription.
     *
     * This will request microphone permissions (if not already provided), load the model (if not already loaded), and
     * generate an updated transcription every {@link MoonshineSettings.FRAME_SIZE} milliseconds. Transcription will stop
     * when {@link stop} is called, or when {@link MoonshineSettings.MAX_RECORD_MS} is passed (whichever comes first).
     */
    async start() {
        // set audio input device and create audio context
        if (!MoonshineTranscriber.mediaRecorder) {
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: true,
            });
            MoonshineTranscriber.mediaRecorder = new MediaRecorder(stream);
        }

        if (!MoonshineTranscriber.audioContext) {
            MoonshineTranscriber.audioContext = new AudioContext({
                sampleRate: 16000,
            });
        }

        // load model if not loaded
        if (!MoonshineTranscriber.model) {
            this.callbacks.onModelLoadStarted();
            MoonshineTranscriber.model = new MoonshineModel(
                MoonshineTranscriber.modelURL
            );
            await MoonshineTranscriber.model.loadModel();
        }

        let audioChunks: Blob[] = [];       // buffer of audio blobs from the user mic
        let commitChunk: boolean = false;   // flag to indicate whether the next generation should be committed (and buffer cleared)
        let transcript: string = "";        // running transcript collected from subsequent buffers

        // fires every MOONSHINE_FRAME_SIZE ms
        MoonshineTranscriber.mediaRecorder.ondataavailable = (event) => {
            let bufferSecs = Math.floor((audioChunks.length * MoonshineSettings.FRAME_SIZE) / 1000)
            if (bufferSecs >= MoonshineSettings.MAX_SPEECH_SECS) {
                // the next transcript should be "committed" and we should erase the buffer afterwards
                commitChunk = true
            }
            audioChunks.push(event.data);

            const audioBlob = new Blob(audioChunks, {
                type: "audio/wav",
            });

            audioBlob.arrayBuffer().then((arrayBuffer) => {
                MoonshineTranscriber.audioContext
                    ?.decodeAudioData(arrayBuffer)
                    .then((decoded) => {
                        let floatArray = new Float32Array(decoded.length);
                        if (floatArray.length > 16000 * 30) {
                            floatArray = floatArray.subarray(0, 16000 * 30);
                        }
                        decoded.copyFromChannel(floatArray, 0);
                        MoonshineTranscriber.model
                            ?.generate(floatArray)
                            .then((text) => {
                                if (text) {
                                    if (commitChunk) {
                                        transcript = transcript + " " + text
                                        this.callbacks.onTranscriptionUpdated(transcript);

                                        let headerBlob = audioChunks[0]
                                        // TODO lookback frames?
                                        audioChunks = []
                                        audioChunks.push(headerBlob)
                                        commitChunk = false
                                    }
                                    else {
                                        this.callbacks.onTranscriptionUpdated(transcript + " " + text);
                                    }
                                }
                            });
                    })
                    .catch(() => {});
            });
        };

        MoonshineTranscriber.mediaRecorder.start(MoonshineSettings.FRAME_SIZE);
        this.callbacks.onTranscribeStarted();

        const recorderTimeOut = setTimeout(() => {
            this.stop();
        }, MoonshineSettings.MAX_RECORD_MS);

        MoonshineTranscriber.mediaRecorder.onstop = () => {
            clearTimeout(recorderTimeOut);
            this.callbacks.onTranscribeStopped();
        };
    }

    /**
     * Stops transcription.
     */
    stop() {
        if (MoonshineTranscriber.mediaRecorder) {
            MoonshineTranscriber.mediaRecorder.stop();
        }
    }
}

export default MoonshineTranscriber;
