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

import { MoonshineElementManager } from "@usefulsensors/moonshine"
var elementManager = new MoonshineElementManager("/model/tiny/")
setInterval(() => {
    // re-autoinject every second, since some elements may not exist on page load (e.g., in react-based sites)
    elementManager.autoInjectElements()
}, 1000);
