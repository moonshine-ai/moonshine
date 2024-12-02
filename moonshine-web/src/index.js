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
    const globalDefinitionElement = document.querySelector("[data-moonshine-template]")
    if (globalDefinitionElement) {
        const definitionElement = globalDefinitionElement.querySelector("[data-moonshine-"+icon+"]")
        if (definitionElement) {
            return definitionElement.innerHTML
        }
    }
    switch(icon) {
        case "loading":
            return `
            <svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px"
                viewBox="0 0 1200 1200" width="100%" height="100%" xml:space="preserve">
                <g>
                    <path d="M852.9,572.3c-15.3-0.1-27.8,12.4-27.8,27.8s12.4,27.8,27.8,27.8h85.4c7.6,0,14.7-3.1,19.7-8.2c5-5,8.2-12.1,8.2-19.7
                        c0-15.4-12.4-27.8-27.8-27.8L852.9,572.3z"/>
                    <path d="M756.5,765.6c-10.9,10.9-10.9,28.5,0,39.4l60.4,60.4c10.9,10.9,28.5,10.9,39.4,0c10.9-10.9,10.9-28.5,0-39.4l-60.4-60.4
                        C785,754.7,767.3,754.7,756.5,765.6z"/>
                    <path d="M563.3,861.9v85.4c0,15.4,12.4,27.8,27.8,27.8c7.6,0,14.7-3.1,19.7-8.2c5-5,8.2-12.1,8.2-19.7v-85.4
                        c0-15.4-12.4-27.8-27.8-27.8C575.8,834.2,563.3,846.5,563.3,861.9L563.3,861.9z"/>
                    <path d="M425.7,765.6c-10.9-10.9-28.5-10.9-39.4,0L325.9,826c-10.9,10.9-10.9,28.5,0,39.4s28.5,10.9,39.4,0l60.4-60.4
                        C436.6,794.1,436.6,776.5,425.7,765.6z"/>
                    <path d="M329.4,628c7.6,0,14.7-3.1,19.7-8.2c5-5,8.2-12.1,8.2-19.7c0-15.4-12.4-27.8-27.8-27.8H244c-15.3-0.1-27.8,12.4-27.8,27.8
                        c0,15.4,12.4,27.8,27.8,27.8L329.4,628z"/>
                    <path d="M425.7,434.9c10.9-10.9,10.9-28.5,0-39.4l-60.4-60.4c-10.9-10.9-28.5-10.9-39.4,0s-10.9,28.5,0,39.4l60.4,60.4
                        C397.2,445.7,414.9,445.7,425.7,434.9z"/>
                    <path d="M796.3,434.4l60.4-60.4c10.9-10.9,10.9-28.5,0-39.4s-28.5-10.9-39.4,0L756.9,395c-10.9,10.9-10.9,28.5,0,39.4
                        S785.4,445.3,796.3,434.4z"/>
                    <path d="M619.4,252.7c0-15.4-12.4-27.8-27.8-27.8s-27.9,12.5-27.8,27.8v85.4c0,15.4,12.4,27.8,27.8,27.8c7.6,0,14.7-3.1,19.7-8.2
                        c5-5,8.2-12.1,8.2-19.7L619.4,252.7z"/>
                </g>
            </svg>`;
        case "recording":
            return `
            <svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px"
                viewBox="0 0 600 600" height="100%" width="100%" xml:space="preserve">
                <circle fill="red" cx="300" cy="299.9" r="151.2"/>
            </svg>`;
        case "transcribing":
            return `
            <svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px"
                viewBox="0 0 1200 1200" width="100%" height="100%" xml:space="preserve">
                <path d="M725.3,954.5c-3.4,0-6.7-0.6-10.1-1.9c-11.5-4.5-18.7-15.9-17.6-28.2l10.7-132.7H310.8c-52.3,0-94.8-42.6-94.8-94.8V340.3
                    c0-52.3,42.6-94.8,94.8-94.8h578.3c52.3,0,94.8,42.6,94.8,94.8v356.5c0,52.3-42.6,94.8-94.8,94.8h-24.6L747.2,943.5
                    c-5.4,6.9-13.5,10.8-22.1,10.8L725.3,954.5z M310.8,301.2c-21.6,0-39.2,17.5-39.2,39.1v356.5c0,21.6,17.5,39.1,39.2,39.1h427.5
                    c7.8,0,15.2,3.2,20.4,9c5.3,5.8,7.9,13.3,7.3,21.1l-5.6,69.2l68.3-88.5c5.3-6.8,13.5-10.8,22.1-10.8h38.3
                    c21.6,0,39.2-17.5,39.2-39.1V340.5c0-21.6-17.5-39.1-39.2-39.1L310.8,301.2L310.8,301.2z"/>
                <path d="M472,494.4c16.9,16.9,16.9,44.3,0,61.2c-16.9,16.9-44.3,16.9-61.2,0c-16.9-16.9-16.9-44.3,0-61.2
                    C427.6,477.5,455.1,477.5,472,494.4"/>
                <path d="M630.6,494.4c16.9,16.9,16.9,44.3,0,61.2c-16.9,16.9-44.3,16.9-61.2,0c-16.9-16.9-16.9-44.3,0-61.2
                    C586.3,477.4,613.7,477.4,630.6,494.4"/>
                <path d="M789.2,494.3c16.9,16.9,16.9,44.3,0,61.2c-16.9,16.9-44.3,16.9-61.2,0c-16.9-16.9-16.9-44.3,0-61.2
                    C744.9,477.4,772.3,477.4,789.2,494.3"/>
            </svg>`;
        default:
        case "idle":
            return `
            <svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px"
                viewBox="0 0 1200 1200" width="100%" height="100%" xml:space="preserve">
                <path d="M864.2,587.3c0-15.4-12.5-27.8-27.8-27.8s-27.8,12.5-27.8,27.8c0,114.9-93.5,208.5-208.5,208.5h-0.5
                    c-114.9,0-208.5-93.6-208.5-208.5c0-15.4-12.5-27.8-27.8-27.8s-27.8,12.5-27.8,27.8c0,136.3,103.8,248.9,236.4,262.8v78.2h-81.6
                    c-15.4,0-27.8,12.5-27.8,27.8s12.5,27.8,27.8,27.8h218.9c15.4,0,27.8-12.5,27.8-27.8s-12.5-27.8-27.8-27.8h-81.6v-78.2
                    c132.7-13.9,236.4-126.5,236.4-262.8L864.2,587.3z"/>
                <path d="M600,744c86.4,0,156.7-70.3,156.7-156.7V372.7c0-86.4-70.3-156.7-156.7-156.7s-156.7,70.3-156.7,156.7v214.5
                    C443.3,673.7,513.6,744,600,744z M498.9,372.7c0-55.7,45.4-101.1,101.1-101.1S701.1,317,701.1,372.7v214.6
                    c0,55.8-45.4,101.1-101.1,101.1s-101.1-45.4-101.1-101.1V372.7z"/>
            </svg>`;
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
