import { useState } from 'react';
import { MoonshineTranscriber, MoonshineLifecycle } from "@usefulsensors/moonshine";
import { IdleIcon, LoadingIcon, TranscribingIcon } from './Icons.js'

import "./css/SpeechInputButton.css";
import React from "react";

function SpeechInputButton({ onUpdate }) {
    const [lifecycle, setLifecycle] = useState(MoonshineLifecycle.idle)

    var transcriber = new MoonshineTranscriber(
        {
            onModelLoadStarted() {
                console.log("onModelLoadStarted()");
                setLifecycle(MoonshineLifecycle.loading)
            },
            onTranscribeStarted() {
                console.log("onTranscribeStarted()");
                setLifecycle(MoonshineLifecycle.transcribing)
            },
            onTranscribeStopped() {
                console.log("onTranscribeStopped()");
                setLifecycle(MoonshineLifecycle.idle)
            },
            onTranscriptionUpdated(text) {
                onUpdate(text)
            },
        },
        "model/tiny"
    );

    function handleClick() {
        switch (lifecycle) {
            case MoonshineLifecycle.idle:
                transcriber.start()
                break;
            case MoonshineLifecycle.transcribing:
                transcriber.stop();
                break;
            case MoonshineLifecycle.loading:
            default:
                break;
        }
    }

    function renderLifeCycleIcon() {
        switch (lifecycle) {
            case MoonshineLifecycle.loading:
                return ( 
                    <span data-moonshine-loading="">
                        <LoadingIcon />
                    </span> 
                )
            case MoonshineLifecycle.transcribing:
                return (
                    <span data-moonshine-transcribing="">
                        <TranscribingIcon />
                    </span>
                )
            case MoonshineLifecycle.idle:
            default:
                return (
                    <span data-moonshine-idle="">
                        <IdleIcon />
                    </span>
                )
        }
    }

    return (
        <div className="moonshine-button" onClick={handleClick}>
            { renderLifeCycleIcon() }
        </div>
    );
}

export default SpeechInputButton;
