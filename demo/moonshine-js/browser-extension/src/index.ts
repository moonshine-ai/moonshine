import { MoonshineSettings } from "@usefulsensors/moonshine";

if (typeof chrome !== "undefined") {
    if (typeof browser !== "undefined") {
        // firefox
        MoonshineSettings.BASE_ASSET_PATH = browser.runtime.getURL("/");
    } else {
        // chrome
        MoonshineSettings.BASE_ASSET_PATH = chrome.runtime.getURL("/");
    }
}

import { MoonshineElementManager } from "@usefulsensors/moonshine";

chrome.storage.sync.get({ model: "tiny", enableEverywhere: false }, (settings) => {
    if (settings.enableEverywhere || window.location.host.includes("chatgpt.com") || window.location.host.includes("claude.ai") ) {
        var inputAreaSelectors: Array<string> | undefined = undefined;
        var styleSheet: string | undefined = undefined;
        var postInjectionFunction: Function | undefined = undefined;

        switch (window.location.host) {
            case "chatgpt.com":
                styleSheet = `
                .moonshine-container {
                    position: inherit !important;
                    display: inherit !important;
                }
                .moonshine-button {
                    top: 8px !important;
                    right: 11px !important;
                }
                @media (prefers-color-scheme: dark) {
                    .moonshine-button span svg path {
                        fill: white;
                    }
                }
                @media (prefers-color-scheme: light) {
                    .moonshine-button span svg path {
                        fill: black;
                    }
                }
                `;
                inputAreaSelectors = ["#prompt-textarea"];
                postInjectionFunction = (controlElement, targetInputElement) => {};
        }
        
        var elementManager = new MoonshineElementManager(
            "model/" + settings.model,
            styleSheet
        );
        setInterval(() => {
            // re-autoinject every second, since some elements may not exist on page load (e.g., in react-based sites)
            elementManager.autoInjectElements(inputAreaSelectors, postInjectionFunction);
            elementManager.initControlElements();
        }, 1000);
    }
})