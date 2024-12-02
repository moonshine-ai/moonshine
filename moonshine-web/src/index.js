import MoonshineModel from "./model";


var moonshineModel = undefined
var moonshineMediaRecorder = undefined

const MoonshineSettings = {
    FRAME_SIZE: 500,
    LOOKBACK_FRAMES: 5,
    // MAX_SPEECH_SECS: 5,
    // MIN_REFRESH_SECS: 0.3,
    LOOKBACK_SIZE: 500 * 5
}

async function runMoonshineToggleTranscription(origin, target) {
    // load model if not loaded
    if (!moonshineModel) {
        origin.dispatchEvent(new CustomEvent("moonshineLoadStarted"))
        moonshineModel = new MoonshineModel("moonshine/tiny")
        await moonshineModel.loadModel()
    }
    // set audio input device
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    moonshineMediaRecorder = new MediaRecorder(stream);

    const audioChunks = [];

    // fires every MOONSHINE_FRAME_SIZE ms
    moonshineMediaRecorder.ondataavailable = event => {
        audioChunks.push(event.data);
    };

    moonshineMediaRecorder.start(MoonshineSettings.FRAME_SIZE);
    origin.dispatchEvent(new CustomEvent("moonshineRecordStarted"))

    return new Promise(resolve => {
        moonshineMediaRecorder.onstop = () => {
            const audioBlob = new Blob(audioChunks, { 
                type: 'audio/wav' 
            });
            origin.dispatchEvent(new CustomEvent("moonshineRecordStopped"))

            origin.dispatchEvent(new CustomEvent("moonshineTranscribeStarted"))
            const audioCTX = new AudioContext({
                sampleRate: 16000,
            });
            audioBlob.arrayBuffer().then(arrayBuffer => {
                audioCTX.decodeAudioData(arrayBuffer).then(decoded => {
                    let floatArray = new Float32Array(decoded.length)
                    if (floatArray.length > (16000 * 30)) {
                        floatArray = floatArray.subarray(0, 16000 * 30)
                    }
                    decoded.copyFromChannel(floatArray, 0)
                    moonshineModel.generate(floatArray).then(text => {
                        console.log("Transcription: " + text)
                        origin.dispatchEvent(new CustomEvent("moonshineTranscribeStopped"))
                        target.value = text
                        resolve(text)
                    })
                });
            })
        };
        // stop recording (if active) after 30 seconds
        setTimeout(() => {
            if (moonshineMediaRecorder) {
                moonshineMediaRecorder.stop()
            }
        }, 30000);
    });
}

async function runMoonshineLiveTranscription(origin, target) {
    // load model if not loaded
    if (!moonshineModel) {
        origin.dispatchEvent(new CustomEvent("moonshineLoadStarted"))
        moonshineModel = new MoonshineModel("moonshine/tiny")
        await moonshineModel.loadModel()
    }

    // set audio input device
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    moonshineMediaRecorder = new MediaRecorder(stream);

    const audioChunks = [];

    // fires every MOONSHINE_FRAME_SIZE ms
    moonshineMediaRecorder.ondataavailable = event => {
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
                if (floatArray.length > (16000 * 30)) {
                    floatArray = floatArray.subarray(0, 16000 * 30)
                }
                decoded.copyFromChannel(floatArray, 0)
                moonshineModel.generate(floatArray).then(text => {
                    target.value = text
                })
            });
        })
    };

    moonshineMediaRecorder.start(MoonshineSettings.FRAME_SIZE);
    origin.dispatchEvent(new CustomEvent("moonshineRecordStarted"))
    origin.dispatchEvent(new CustomEvent("moonshineTranscribeStarted"))

    return new Promise(resolve => {
        moonshineMediaRecorder.onstop = () => {
            origin.dispatchEvent(new CustomEvent("moonshineRecordStopped"))
            origin.dispatchEvent(new CustomEvent("moonshineTranscribeStopped"))
            resolve()
        };
        // stop recording (if active) after 30 seconds
        setTimeout(() => {
            if (moonshineMediaRecorder) {
                moonshineMediaRecorder.stop()
            }
        }, 30000);
    });
}

const lifecycleAttributes = ["loading", "recording", "transcribing", "idle"]

function initMoonshineLifecycleIcons(parentButton) {
    // inject innerHTML for lifecycle icons wherever inline overrides are not specified
    lifecycleAttributes.forEach(attr => {
        const iconElement = parentButton.querySelector(":scope > [data-moonshine-"+attr+"]")
        if (!iconElement) {
            let injectedIconElement = document.createElement("span")
            injectedIconElement.innerHTML = getMoonshineLifecycleInnerHTML(attr)
            injectedIconElement.setAttribute("data-moonshine-"+attr, "")
            parentButton.appendChild(injectedIconElement)
        }
        else {
            // inline override set; do nothing
        }
    })
    showMoonshineLifecycleIcon(parentButton, "idle")
}

function showMoonshineLifecycleIcon(parentButton, icon) {
    const hideAttributes = lifecycleAttributes.filter((attr) => attr != icon);

    hideAttributes.forEach(attr => {
        const hideElements = parentButton.querySelectorAll(":scope > [data-moonshine-"+attr+"]")
        hideElements.forEach(hideElement => {
            hideElement.style.display = "none"
        })
    })

    const showElements = parentButton.querySelectorAll(":scope > [data-moonshine-"+icon+"]")
    showElements.forEach(showElement => {
        showElement.style.display = "inline-block"
    })
}

function getMoonshineLifecycleInnerHTML(icon) {
    switch(icon) {
        case "loading":
            return "Loading";
        case "recording":
            return "Recording";
        case "transcribing":
            return "Transcribing";
        default:
        case "idle":
            return "Click me";
    }
}

const moonshineControlElements = document.querySelectorAll('[data-moonshine-target]');

moonshineControlElements.forEach(controlElement => {
    var targetElementSelector = controlElement.attributes["data-moonshine-target"].value
    var targetElements = document.querySelectorAll(targetElementSelector)
    initMoonshineLifecycleIcons(controlElement)
    targetElements.forEach(targetElement => {
        controlElement.addEventListener("click", () => {
            // if not transcribing, start transcribing
            if (!controlElement.attributes["data-moonshine-active"]) {
                moonshineControlElements.forEach(element => {
                    // disable other s2t buttons
                    if (element != controlElement) {
                        element.setAttribute("disabled", "")
                    }
                })
                controlElement.setAttribute("data-moonshine-active", "")
                // bind transcription type based on data-moonshine-live attribute
                if (controlElement.attributes["data-moonshine-live"]) {
                    runMoonshineLiveTranscription(controlElement, targetElement).then(() => {
                        controlElement.removeAttribute("data-moonshine-active")
                        moonshineMediaRecorder = undefined
                    })
                }
                else {
                    runMoonshineToggleTranscription(controlElement, targetElement).then(() => {
                        controlElement.removeAttribute("data-moonshine-active")
                        moonshineMediaRecorder = undefined
                    })
                }
            }
            // if transcribing, stop transcribing
            else {
                moonshineControlElements.forEach(element => {
                    // re-enable other s2t buttons
                    if (element != controlElement) {
                        element.removeAttribute("disabled")
                    }
                })
                moonshineMediaRecorder.stop()
                controlElement.removeAttribute("data-moonshine-active")
            }
        })
        controlElement.addEventListener("moonshineLoadStarted", () => {
            console.log("moonshineLoadStarted")
            showMoonshineLifecycleIcon(controlElement, "loading")
        })
        controlElement.addEventListener("moonshineRecordStarted", () => {
            console.log("moonshineRecordStarted")
            showMoonshineLifecycleIcon(controlElement, "recording")
        })
        controlElement.addEventListener("moonshineRecordStopped", () => {
            console.log("moonshineRecordStopped")
            showMoonshineLifecycleIcon(controlElement, "idle")
        })
        controlElement.addEventListener("moonshineTranscribeStarted", () => {
            console.log("moonshineTranscribeStarted")
            showMoonshineLifecycleIcon(controlElement, "transcribing")
        })
        controlElement.addEventListener("moonshineTranscribeStopped", () => {
            console.log("moonshineTranscribeStopped")
            showMoonshineLifecycleIcon(controlElement, "idle")
        })
    })
});
