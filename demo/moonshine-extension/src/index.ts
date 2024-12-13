import { MoonshineSettings } from "@usefulsensors/moonshine"

if (typeof chrome !== "undefined") {
    if (typeof browser !== "undefined") {
        // firefox
        MoonshineSettings.BASE_ASSET_PATH = browser.runtime.getURL("/")
    } else {
        // chrome
        MoonshineSettings.BASE_ASSET_PATH = chrome.runtime.getURL("/")
    }
}

import { autoInjectMoonshineControlElements } from "@usefulsensors/moonshine"
setInterval(() => {
    autoInjectMoonshineControlElements("/model/tiny/")
}, 1000);
