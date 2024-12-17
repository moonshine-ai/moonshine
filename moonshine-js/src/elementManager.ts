import MoonshineTranscriber from "./transcriber";
import { MoonshineLifecycle } from "./constants";
import styles from "./css/base.css"
import IdleIcon from "./svg/idle.svg"
import LoadingIcon from "./svg/loading.svg"
import TranscribingIcon from "./svg/transcribing.svg"

function getRandomID() {
    const characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    const length = 8;
    let result = "";
    for (let i = 0; i < length; i++) {
        const randomIndex = Math.floor(Math.random() * characters.length);
        result += characters[randomIndex];
    }
    return result;
}

export default class MoonshineElementManager {
    private inputAreaSelectors: Array<string> = [
        "textarea",
        'input[type="text"], input[type="search"]',
        'div[contenteditable="true"]',
        'span[contenteditable="true"]',
    ];
    private styleSheet: string = styles;
    private boundControlElements: Array<string> = [];
    private modelURL: string = ""; // defaults to MoonshineSettings.BASE_ASSET_URL in MoonshineModel
    private postInjectionFunction: Function = (
        controlElement,
        targetInputElement
    ) => {
        const targetID = targetInputElement.id;
        // squeeze button into smaller inputs if they exceed the button's max height
        // note: need to get the injected button from the DOM to determine its actual height
        const computedButtonHeight = parseInt(
            window
                .getComputedStyle(controlElement)
                .getPropertyValue("max-height"),
            10
        );
        const inputRect = targetInputElement.getBoundingClientRect();

        // shrink the button if it is larger than the input field
        if (inputRect.height < computedButtonHeight) {
            console.log(inputRect.height)
            controlElement.style.height = inputRect.height + "px";
            controlElement.style.width = inputRect.height + "px";
        }

        // vertically center the button if the input height is close to (but greater than) the button height
        if (
            inputRect.height < 2 * computedButtonHeight &&
            inputRect.height > computedButtonHeight
        ) {
            controlElement.style.top =
                (inputRect.height - computedButtonHeight) / 2 + "px";
        }
        const container = controlElement.parentNode;
        const parentStyle = window.getComputedStyle(container.parentNode)

        if (parentStyle.getPropertyValue("display") == "flex") {
            container.style.display = "flex"
        }

        container.style.width = inputRect.width;
    };

    public constructor(
        modelURL?: string,
        inputAreaSelectors?: Array<string>,
        styleSheet?: string,
        postInjectionFunction?: Function
    ) {
        if (inputAreaSelectors) {
            this.inputAreaSelectors = inputAreaSelectors;
        }
        if (styleSheet) {
            this.styleSheet += styleSheet;
        }
        if (modelURL) {
            this.modelURL = modelURL;
        }
        if (postInjectionFunction) {
            this.postInjectionFunction = postInjectionFunction;
        }
        this.injectStyle();
    }

    public autoInjectElements() {
        // query selectors for each type of input element we want to add buttons to
        this.inputAreaSelectors.forEach((querySelector) => {
            const elements = document.querySelectorAll(querySelector);
            elements.forEach((element) => {
                if (
                    !document.querySelector(
                        '[data-moonshine-target="#' + element.id + '"]'
                    )
                ) {
                    this.wrapAndReinjectInputElement(element);
                    // the element should not be bound yet; if it is, the page may have reloaded since then so we need to remove it
                    if (this.boundControlElements.includes("#" + element.id)) {
                        const index = this.boundControlElements.indexOf(
                            "#" + element.id
                        );
                        if (index !== -1) {
                            this.boundControlElements.splice(index, 1);
                        }
                    }
                }
            });
        });
    }

    public initControlElements() {
        const moonshineControlElements = document.querySelectorAll(
            "[data-moonshine-target]"
        );

        moonshineControlElements.forEach((controlElement) => {
            var targetElementSelector =
                controlElement.attributes["data-moonshine-target"].value;
            if (!this.boundControlElements.includes(targetElementSelector)) {
                var targetElements = document.querySelectorAll(
                    targetElementSelector
                );
                MoonshineElementManager.initLifecycleIcons(controlElement);
                targetElements.forEach((targetElement) => {
                    var transcriber = new MoonshineTranscriber(
                        {
                            onModelLoadStarted() {
                                // disable other s2t buttons
                                moonshineControlElements.forEach((element) => {
                                    if (element != controlElement) {
                                        element.setAttribute("disabled", "");
                                    }
                                });
                                MoonshineElementManager.showLifecycleIcon(
                                    controlElement,
                                    MoonshineLifecycle.loading
                                );
                            },
                            onTranscribeStarted() {
                                // disable other s2t buttons
                                moonshineControlElements.forEach((element) => {
                                    if (element != controlElement) {
                                        element.setAttribute("disabled", "");
                                    }
                                });
                                controlElement.setAttribute(
                                    "data-moonshine-active",
                                    ""
                                );
                                MoonshineElementManager.showLifecycleIcon(
                                    controlElement,
                                    MoonshineLifecycle.transcribing
                                );
                            },
                            onTranscribeStopped() {
                                controlElement.removeAttribute(
                                    "data-moonshine-active"
                                );
                                MoonshineElementManager.showLifecycleIcon(
                                    controlElement,
                                    MoonshineLifecycle.idle
                                );

                                // re-enable other s2t buttons
                                moonshineControlElements.forEach((element) => {
                                    if (element != controlElement) {
                                        element.removeAttribute("disabled");
                                    }
                                });
                            },
                            onTranscriptionUpdated(text) {
                                targetElement.innerHTML = text;
                                targetElement.value = text;
                            },
                        },
                        this.modelURL
                    );
                    controlElement.addEventListener("click", () => {
                        // TODO fix for elements where the "disabled" attribute does not block click events (e.g., divs)
                        // if not transcribing, start transcribing
                        if (
                            !controlElement.attributes["data-moonshine-active"]
                        ) {
                            transcriber.start();
                        }
                        // if transcribing, stop transcribing
                        else {
                            transcriber.stop();
                            // const enterKeyEvent = new KeyboardEvent("keydown", {
                            //     key: "Enter",
                            //     code: "Enter",
                            //     which: 13,
                            //     keyCode: 13,
                            // });
                            // targetElement.dispatchEvent(enterKeyEvent);
                        }
                    });
                });
                this.boundControlElements.push(targetElementSelector);
            }
        });
    }

    static initLifecycleIcons(parentButton) {
        // inject innerHTML for lifecycle icons wherever inline overrides are not specified
        Object.values(MoonshineLifecycle).forEach((attr: string) => {
            const iconElement = parentButton.querySelector(
                ":scope > [data-moonshine-" + attr + "]"
            );
            if (!iconElement) {
                let injectedIconElement = document.createElement("span");
                injectedIconElement.innerHTML = this.getLifecycleInnerHTML(
                    MoonshineLifecycle[attr]
                );
                injectedIconElement.setAttribute("data-moonshine-" + attr, "");
                parentButton.appendChild(injectedIconElement);
            }
        });
        MoonshineElementManager.showLifecycleIcon(
            parentButton,
            MoonshineLifecycle.idle
        );
    }

    static showLifecycleIcon(parentButton, lifecycle: MoonshineLifecycle) {
        const hideAttributes = Object.values(MoonshineLifecycle).filter(
            (attr) => attr != lifecycle
        );

        hideAttributes.forEach((attr) => {
            const hideElements = parentButton.querySelectorAll(
                ":scope > [data-moonshine-" + attr + "]"
            );
            hideElements.forEach((hideElement) => {
                hideElement.style.display = "none";
            });
        });

        const showElements = parentButton.querySelectorAll(
            ":scope > [data-moonshine-" + lifecycle + "]"
        );
        showElements.forEach((showElement) => {
            showElement.style.display = "inline-block";
        });
    }

    static getLifecycleInnerHTML(lifecycle: MoonshineLifecycle) {
        const globalDefinitionElement = document.querySelector(
            "[data-moonshine-template]"
        );
        if (globalDefinitionElement) {
            const definitionElement = globalDefinitionElement.querySelector(
                "[data-moonshine-" + lifecycle + "]"
            );
            if (definitionElement) {
                return definitionElement.innerHTML;
            }
        }
        switch (lifecycle) {
            case MoonshineLifecycle.loading:
                return LoadingIcon;
            case MoonshineLifecycle.transcribing:
                return TranscribingIcon;
            default:
            case MoonshineLifecycle.idle:
                return IdleIcon;
        }
    }

    private wrapAndReinjectInputElement(inputElement: Element) {
        const targetID = inputElement.id ? inputElement.id : getRandomID();

        const container = document.createElement("div");
        container.className = "moonshine-container";

        const button = document.createElement("div");
        button.className = "moonshine-button";

        button.setAttribute("data-moonshine-target", "#" + targetID);
        if (!inputElement.id) {
            inputElement.id = targetID;
        }

        inputElement.parentNode?.replaceChild(container, inputElement);
        container.appendChild(inputElement);
        container.appendChild(button);

        this.postInjectionFunction(button, inputElement);
    }

    private injectStyle() {
        const styleElement = document.createElement("style");
        styleElement.type = "text/css";
        document.head.appendChild(styleElement);
        styleElement.innerHTML = this.styleSheet;
    }
}
