import MoonshineModel from "./model";
import { MoonshineSettings } from "./constants"


interface MoonshineTranscriberCallbacks {
    onModelLoadStarted: () => any

    onTranscribeStarted: () => any

    onTranscribeStopped: () => any

    onTranscriptionUpdated: (text: string | undefined) => any
}

const defaultTranscriberCallbacks: MoonshineTranscriberCallbacks = {
    onModelLoadStarted() {
        console.log("MoonshineTranscriber.onModelLoadStarted()")
    },
    onTranscribeStarted: function () {
        console.log("MoonshineTranscriber.onTranscribeStarted()")
    },
    onTranscribeStopped: function () {
        console.log("MoonshineTranscriber.onTranscribeStopped()")
    },
    onTranscriptionUpdated: function (text: string | undefined) {
        console.log("MoonshineTranscriber.onTranscriptionUpdated(" + text + ")")
    }
}

class MoonshineTranscriber {
    private static mediaRecorder: MediaRecorder | undefined = undefined
    private static model: MoonshineModel | undefined = undefined
    private static audioContext: AudioContext | undefined = undefined
    private static modelURL: String
    private callbacks: MoonshineTranscriberCallbacks

    public constructor(callbacks: Partial<MoonshineTranscriberCallbacks> = {}, modelURL: String) {
        this.callbacks = { ...defaultTranscriberCallbacks, ...callbacks }
        MoonshineTranscriber.modelURL = modelURL
    }

    async start() {
        // set audio input device and create audio context
        if (!MoonshineTranscriber.mediaRecorder) {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            MoonshineTranscriber.mediaRecorder = new MediaRecorder(stream);
        }

        if (!MoonshineTranscriber.audioContext) {
            MoonshineTranscriber.audioContext = new AudioContext({
                sampleRate: 16000,
            });
        }

        // load model if not loaded
        if (!MoonshineTranscriber.model) {
            this.callbacks.onModelLoadStarted()
            MoonshineTranscriber.model = new MoonshineModel(MoonshineTranscriber.modelURL)
            await MoonshineTranscriber.model.loadModel()
        }

        const audioChunks: Blob[] = [];

        // fires every MOONSHINE_FRAME_SIZE ms
        MoonshineTranscriber.mediaRecorder.ondataavailable = event => {
            audioChunks.push(event.data);

            const audioBlob = new Blob(audioChunks, { 
                type: 'audio/wav' 
            });

            audioBlob.arrayBuffer().then(arrayBuffer => {
                MoonshineTranscriber.audioContext?.decodeAudioData(arrayBuffer).then(decoded => {
                    let floatArray = new Float32Array(decoded.length)
                    // TODO segment array > x seconds and transcribe segments
                    if (floatArray.length > (16000 * 30)) {
                        floatArray = floatArray.subarray(0, 16000 * 30)
                    }
                    decoded.copyFromChannel(floatArray, 0)
                    MoonshineTranscriber.model?.generate(floatArray).then(text => {
                        if (text) {
                            this.callbacks.onTranscriptionUpdated(text)
                        }
                    })
                }).catch(() => {});
            })
        };

        MoonshineTranscriber.mediaRecorder.start(MoonshineSettings.FRAME_SIZE);
        this.callbacks.onTranscribeStarted()

        const recorderTimeOut = setTimeout(() => {
            this.stop()
        }, MoonshineSettings.MAX_RECORD_MS);

        MoonshineTranscriber.mediaRecorder.onstop = () => {
            clearTimeout(recorderTimeOut)
            this.callbacks.onTranscribeStopped()
        };
    }

    stop() {
        if (MoonshineTranscriber.mediaRecorder) {
            MoonshineTranscriber.mediaRecorder.stop()
        }
    }
}

export { MoonshineTranscriber }
