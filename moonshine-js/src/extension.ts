var wasmURL = ""
var modelURL = ""

if (typeof chrome !== "undefined") {
    if (typeof browser !== "undefined") {
        // firefox
        wasmURL = browser.runtime.getURL("/wasm/")
        modelURL = browser.runtime.getURL("/model/tiny/")
    } else {
        // chrome
        wasmURL = chrome.runtime.getURL("/wasm/")
        modelURL = chrome.runtime.getURL("/model/tiny/")
    }
}

import * as ort from "onnxruntime-web"
ort.env.wasm.wasmPaths = wasmURL

import { autoInjectMoonshineControlElements } from "./auto"
autoInjectMoonshineControlElements(modelURL)