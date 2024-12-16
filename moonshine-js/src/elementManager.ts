import MoonshineTranscriber from "./transcriber";
import { MoonshineLifecycle } from "./constants";
import { WatchDirectoryFlags } from "typescript";

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
    // TODO fetching this from .css rather than inlining would be preferable
    private styleSheet: string = `
        .moonshine-container {
            position: relative;
            display: inline-block;
            width: 100%;
            overflow: hidden;
        }
    
        .moonshine-button {
            position: absolute;
            max-width: 32px;
            max-height: 32px;
            top: 0;
            right: 0;
            cursor: pointer;
        }
    
        @keyframes loading {
            from {
                transform: rotate(0deg);
            }
            to {
                transform: rotate(360deg);
            }
        }
    
        *[data-moonshine-loading] svg {
            animation: loading 2s linear infinite;
            transform-origin: 50% 50%;
        }
    
        @keyframes transcribing {
            from {
                transform: translateY(0);
            }
            25% {
                transform: translateY(-5%);
            }
            50%, 100% {
                transform: translateY(0);
            }
        }
    
        *[data-moonshine-transcribing] .moonshine-transcribe-1 {
            animation: transcribing 0.75s linear infinite;
        }
    
        *[data-moonshine-transcribing] .moonshine-transcribe-2 {
            animation: transcribing 0.75s linear infinite;
            animation-delay: 0.25s;
        }
    
        *[data-moonshine-transcribing] .moonshine-transcribe-3 {
            animation: transcribing 0.75s linear infinite;
            animation-delay: 0.5s;
        }
        `;
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
                    // the element should not be bound yet; if it is, the page may have reloaded since then
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
                // // add a mutation observer to check if this element gets removed (so we can potentially re-add it later)
                // var mutationObserver = new MutationObserver(
                //     (mutationsList, observer) => {
                //         console.log(mutationsList);
                //         mutationsList.forEach((mutationRecord) => {
                //             if (mutationRecord.removedNodes) {
                //                 mutationRecord.removedNodes.forEach((node) => {
                //                     if (
                //                         this.boundControlElements.includes(
                //                             "#" + node.id
                //                         )
                //                     ) {
                //                         const index =
                //                             this.boundControlElements.indexOf(
                //                                 "#" + node.id
                //                             );
                //                         if (index !== -1) {
                //                             this.boundControlElements.splice(
                //                                 index,
                //                                 1
                //                             );
                //                         }
                //                     }
                //                 });
                //             }
                //         });
                //     }
                // );
                // mutationObserver.observe(controlElement.parentNode.parentNode, {
                //     childList: true,
                //     subtree: true,
                // });
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
        // TODO fetch these rather than returning inline svg
        switch (lifecycle) {
            case MoonshineLifecycle.loading:
                return `
                  <svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px"
                      viewBox="0 0 1200 1200" width="100%" height="100%" xml:space="preserve">
                      <path d="M627.8,861.8c0.1-15.3-12.4-27.8-27.8-27.8s-27.8,12.4-27.8,27.8v85.4c0,7.6,3.1,14.7,8.2,19.7c5,5,12.1,8.2,19.7,8.2
                          c15.4,0,27.8-12.4,27.8-27.8L627.8,861.8z"/>
                      <path d="M434.5,765.4c-10.9-10.9-28.5-10.9-39.4,0l-60.4,60.4c-10.9,10.9-10.9,28.5,0,39.4c10.9,10.9,28.5,10.9,39.4,0l60.4-60.4
                          C445.4,793.9,445.4,776.3,434.5,765.4z"/>
                      <path d="M338.2,572.2h-85.4c-15.4,0-27.8,12.4-27.8,27.8c0,7.6,3.1,14.7,8.2,19.7c5,5,12.1,8.2,19.7,8.2h85.4
                          c15.4,0,27.8-12.4,27.8-27.8C365.9,584.7,353.6,572.2,338.2,572.2L338.2,572.2z"/>
                      <path d="M434.5,434.7c10.9-10.9,10.9-28.5,0-39.4l-60.4-60.4c-10.9-10.9-28.5-10.9-39.4,0c-10.9,10.9-10.9,28.5,0,39.4l60.4,60.4
                          C406,445.5,423.6,445.5,434.5,434.7z"/>
                      <path d="M572,338.4c0,7.6,3.1,14.7,8.2,19.7c5,5,12.1,8.2,19.7,8.2c15.4,0,27.8-12.4,27.8-27.8V253c0.1-15.3-12.4-27.8-27.8-27.8
                          S572,237.5,572,253L572,338.4z"/>
                      <path d="M765.2,434.7c10.9,10.9,28.5,10.9,39.4,0l60.4-60.4c10.9-10.9,10.9-28.5,0-39.4c-10.9-10.9-28.5-10.9-39.4,0l-60.4,60.4
                          C754.3,406.2,754.3,423.8,765.2,434.7z"/>
                      <path d="M765.7,805.2l60.4,60.4c10.9,10.9,28.5,10.9,39.4,0s10.9-28.5,0-39.4l-60.4-60.4c-10.9-10.9-28.5-10.9-39.4,0
                          S754.8,794.4,765.7,805.2z"/>
                      <path d="M947.4,628.3c15.4,0,27.8-12.4,27.8-27.8c0-15.4-12.5-27.9-27.8-27.8H862c-15.4,0-27.8,12.4-27.8,27.8
                          c0,7.6,3.1,14.7,8.2,19.7c5,5,12.1,8.2,19.7,8.2L947.4,628.3z"/>
                  </svg>`;
            case MoonshineLifecycle.transcribing:
                return `
                  <svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px"
                      viewBox="0 0 1200 1200" width="100%" height="100%" xml:space="preserve">
                      <path d="M725.3,954.5c-3.4,0-6.7-0.6-10.1-1.9c-11.5-4.5-18.7-15.9-17.6-28.2l10.7-132.7H310.8c-52.3,0-94.8-42.6-94.8-94.8V340.3
                          c0-52.3,42.6-94.8,94.8-94.8h578.3c52.3,0,94.8,42.6,94.8,94.8v356.5c0,52.3-42.6,94.8-94.8,94.8h-24.6L747.2,943.5
                          c-5.4,6.9-13.5,10.8-22.1,10.8L725.3,954.5z M310.8,301.2c-21.6,0-39.2,17.5-39.2,39.1v356.5c0,21.6,17.5,39.1,39.2,39.1h427.5
                          c7.8,0,15.2,3.2,20.4,9c5.3,5.8,7.9,13.3,7.3,21.1l-5.6,69.2l68.3-88.5c5.3-6.8,13.5-10.8,22.1-10.8h38.3
                          c21.6,0,39.2-17.5,39.2-39.1V340.5c0-21.6-17.5-39.1-39.2-39.1L310.8,301.2L310.8,301.2z"/>
                      <path class="moonshine-transcribe-1" d="M472,494.4c16.9,16.9,16.9,44.3,0,61.2c-16.9,16.9-44.3,16.9-61.2,0c-16.9-16.9-16.9-44.3,0-61.2
                          C427.6,477.5,455.1,477.5,472,494.4"/>
                      <path class="moonshine-transcribe-2" d="M630.6,494.4c16.9,16.9,16.9,44.3,0,61.2c-16.9,16.9-44.3,16.9-61.2,0c-16.9-16.9-16.9-44.3,0-61.2
                          C586.3,477.4,613.7,477.4,630.6,494.4"/>
                      <path class="moonshine-transcribe-3" d="M789.2,494.3c16.9,16.9,16.9,44.3,0,61.2c-16.9,16.9-44.3,16.9-61.2,0c-16.9-16.9-16.9-44.3,0-61.2
                          C744.9,477.4,772.3,477.4,789.2,494.3"/>
                  </svg>`;
            default:
            case MoonshineLifecycle.idle:
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
