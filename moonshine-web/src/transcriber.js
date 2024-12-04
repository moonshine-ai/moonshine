import MoonshineModel from "./ts/model.js";
import { MoonshineEvents, MoonshineSettings } from "./constants.js"

var mediaRecorder = undefined
var model = undefined

async function startTranscription(origin, target) {
    // set audio input device
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);

    // load model if not loaded
    if (!model) {
        origin.dispatchEvent(new CustomEvent(MoonshineEvents.LOAD_STARTED))
        model = new MoonshineModel("moonshine/tiny")
        await model.loadModel()
    }

    const audioChunks = [];

    // fires every MOONSHINE_FRAME_SIZE ms
    mediaRecorder.ondataavailable = event => {
        audioChunks.push(event.data);

        const audioBlob = new Blob(audioChunks, { 
            type: 'audio/wav' 
        });

        const audioCTX = new AudioContext({
            sampleRate: 16000,
        });
        audioBlob.arrayBuffer().then(arrayBuffer => {
            audioCTX.decodeAudioData(arrayBuffer).then(decoded => {
                let floatArray = new Float32Array(decoded.length)
                // TODO segment array > x seconds and transcribe segments
                if (floatArray.length > (16000 * 30)) {
                    floatArray = floatArray.subarray(0, 16000 * 30)
                }
                decoded.copyFromChannel(floatArray, 0)
                model.generate(floatArray).then(text => {
                    target.value = text
                })
            });
        })
    };

    mediaRecorder.start(MoonshineSettings.FRAME_SIZE);
    origin.dispatchEvent(new CustomEvent(MoonshineEvents.TRANSCRIBE_STARTED))

    return new Promise(resolve => {
        // stop recording after 30 seconds
        const recorderTimeOut = setTimeout(() => {
            stopTranscription()
            resolve()
        }, MoonshineSettings.MAX_RECORD_MS);

        mediaRecorder.onstop = () => {
            origin.dispatchEvent(new CustomEvent(MoonshineEvents.TRANSCRIBE_STOPPED))
            clearTimeout(recorderTimeOut)
            resolve()
        };
    });
}

function stopTranscription() {
    if (mediaRecorder) {
        mediaRecorder.stop()
    }
}

export {
    startTranscription,
    stopTranscription, 
}