import MoonshineTranscriber from "./transcriber";
import { MoonshineLifecycle } from "./constants";
import styles from "./css/base.css";
import IdleIcon from "./svg/idle.svg";
import LoadingIcon from "./svg/loading.svg";
import TranscribingIcon from "./svg/transcribing.svg";

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

/**
 * Handles the injection, initialization, and management of HTML elements, CSS styles, and JS event listeners
 * that integrate a speech-to-text UI on pages.
 *
 * Automatic and custom integration with the UI are implemented here. {@link autoInjectElements} injects HTML for adding
 * speech-to-text buttons to input text areas, while {@link initControlElements} binds event listeners for these buttons.
 */
export default class MoonshineElementManager {
    private boundControlElements: Array<string> = [];
    private modelURL: string; // defaults to MoonshineSettings.BASE_ASSET_URL in MoonshineModel
    private defaultPostInjectionFunction: Function = (
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
        const parentStyle = window.getComputedStyle(container.parentNode);

        if (parentStyle.getPropertyValue("display") == "flex") {
            container.style.display = "flex";
        }

        container.style.width = inputRect.width;
    };

    /**
     * @param modelURL (Optional) the URL that the MoonshineModel weights should be loaded from, relative to {@link MoonshineSettings.BASE_ASSET_PATH}
     * @param styleSheet (Optional) additional CSS styles to append to the base stylesheet
     */
    public constructor(modelURL: string = "", styleSheet: string = styles) {
        this.modelURL = modelURL;
        this.injectStyle(styleSheet);
    }

    /**
     * Injects HTML elements to automatically add speech-to-text to a page.
     * 
     * @param inputAreaSelectors (Optional) a list of CSS selectors that point to elements that should have speech-to-text capabilities added to them. The default value for this argument points to a general-purpose set of input types, but this can be overridden to point to, e.g., specific elements by id, or elements that are not input fields but we might otherwise want to output transcriptions to.
     * @param postInjectionFunction (Optional) a function to run after injecting elements. This is useful when we need to apply additional dynamic styling to the elements using, e.g., computed dimensions of elements as they are rendered on the page. Takes the injected button element and target input element as arguments.
     */
    public autoInjectElements(
        inputAreaSelectors: Array<string> = [
            "textarea",
            'input[type="text"], input[type="search"]',
            'div[contenteditable="true"]',
            'span[contenteditable="true"]',
        ],
        postInjectionFunction: Function = this.defaultPostInjectionFunction
    ) {
        // query selectors for each type of input element we want to add buttons to
        inputAreaSelectors.forEach((querySelector) => {
            const elements = document.querySelectorAll(querySelector);
            elements.forEach((element) => {
                if (
                    !document.querySelector(
                        '[data-moonshine-target="#' + element.id + '"]'
                    )
                ) {
                    this.wrapAndReinjectInputElement(
                        element,
                        postInjectionFunction
                    );
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

    /**
     * Initializes speech-to-text capabilities for all elements on the page that are pointed to by a button or other element with a `data-moonshine-target` specified. 
     * 
     * This should be run in both the "custom" case (where we've manually added button elements in the page that point to other elements to output to) and in the "automatic" case (where we've automatically injected buttons in the DOM and set their targets to other elements to output to).
     */
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

    /**
     * Injects HTML for the lifecycle icons (loading, transcribing, idle) of a speech-to-text button. 
     * 
     * If icon overrides have been specified on the page with the `data-moonshine-{loading,transcribing,idle}` attribute, use this; otherwise, use the icons returned by {@link getLifecycleInnerHTML}
     * @param parentButton The button element to inject icon HTML
     */
    static initLifecycleIcons(parentButton: Element) {
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

    /**
     * Displays the appropriate icon for the given lifecycle step (i.e., idle, loading, transcribing) on the given button.
     * 
     * @param parentButton The element to show the specified icon for
     * @param lifecycle The {@link MoonshineLifecycle} to display the icon for
     */
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

    /**
     * Get the appropriate icon HTML for the given step of the lifecycle. If a `data-moonshine-template` button is specified somewhere on the page (which includes the HTML that should be globally applied to all speech-to-text buttons), the content of that element is used; otherwise, the default icons are used.
     * @param lifecycle the {@link MoonshineLifecycle} step to get the icon HTML for
     * @returns a string of inline HTML for the icon
     */
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

    private wrapAndReinjectInputElement(
        inputElement: Element,
        postInjectionFunction: Function
    ) {
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

        postInjectionFunction(button, inputElement);
    }

    private injectStyle(styleSheet: string) {
        const styleElement = document.createElement("style");
        styleElement.type = "text/css";
        document.head.appendChild(styleElement);
        styleElement.innerHTML = styleSheet;
    }
}
