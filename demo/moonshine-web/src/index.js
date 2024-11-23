import Moonshine from "./moonshine.js"


function setTranscription(text) {
    console.log("setTranscription: " + text)
    document.getElementById("transcription").innerHTML = text
}

async function getRecordingDevice() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    return new MediaRecorder(stream);
}

async function startRecording(mediaRecorder) {
    const audioChunks = [];
  
    mediaRecorder.ondataavailable = event => {
        audioChunks.push(event.data);
    };
  
    mediaRecorder.start();
    console.log("Recording started");
  
    return new Promise(resolve => {
        mediaRecorder.onstop = () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            console.log("Recording stopped");
            resolve(audioBlob);
        };
        // max recording length is 30 seconds
        setTimeout(() => mediaRecorder.stop(), 30000);
    });
}

function toggleAudioPanel(isVisible) {
    var display = isVisible ? "inherit" : "none"
    document.getElementById("audioPanel").style = "display: " + display + ";"
}

function toggleControls(isEnabled) {
    document.getElementById("transcribe").disabled = !isEnabled
    document.getElementById("startRecord").disabled = !isEnabled
    document.getElementById("stopRecord").disabled = !isEnabled
    document.getElementById("models").disabled = !isEnabled
    document.getElementById("browse").disabled = !isEnabled
}

function setAudio(audioBlob) {
    var sound = document.getElementById("sound")
    sound.src = URL.createObjectURL(audioBlob)
}

window.onload = (event) => {
    toggleControls(false)
    var model_name = document.getElementById("models").value
    setTranscription("Loading " + model_name + "...")
    var moonshine = new Moonshine(model_name)
    moonshine.loadModel().then(() => {
        setTranscription("")
        toggleControls(true)
    });

    models.onchange = async function(e) {
        var selection = document.getElementById("models").value
        if (selection != model_name) {
            toggleControls(false)
            model_name = selection
            setTranscription("Loading " + model_name + "...")
            var moonshine = new Moonshine(model_name)
            moonshine.loadModel().then(() => {
                setTranscription("")
                toggleControls(true)
            });
        }
    }

    upload.onchange = function(e) {
        setAudio(this.files[0])
        toggleAudioPanel(true)
    }

    transcribe.onclick = async function(e) {
        var sound = document.getElementById("sound")
        if (sound.src) {
            toggleControls(false)
            const audioCTX = new AudioContext({
                sampleRate: 16000,
            });
            let file = await fetch(sound.src).then(r => r.blob())
            let data = await file.arrayBuffer()
            let decoded = await audioCTX.decodeAudioData(data);
            let floatArray = new Float32Array(decoded.length)
            if (floatArray.length > (16000 * 30)) {
                floatArray = floatArray.subarray(0, 16000 * 30)
                alert("Your audio is greater than 30 seconds in length. Moonshine will transcribe the first 30 seconds.")
            }
            decoded.copyFromChannel(floatArray, 0)
            moonshine.generate(floatArray).then((r) => {
                setTranscription(r)
                toggleControls(true)
            })
        }
    }

    var mediaRecorder = undefined
    startRecord.onclick = async function(e) {
        document.getElementById("startRecord").style = "display: none;"
        document.getElementById("stopRecord").style = "display: block;"

        if (!mediaRecorder) {
            mediaRecorder = await getRecordingDevice()
        }

        // fired when recording hits the time limit or when recording stops
        startRecording(mediaRecorder).then(audioBlob => {
            document.getElementById("startRecord").style = "display: block;"
            document.getElementById("stopRecord").style = "display: none;"
            setAudio(audioBlob)
            toggleAudioPanel(true)
        });
    }
    stopRecord.onclick = function(e) {
        document.getElementById("startRecord").style = "display: block;"
        document.getElementById("stopRecord").style = "display: none;"
        mediaRecorder.stop()
    }
};
