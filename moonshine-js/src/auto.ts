import { initMoonshineControlElements } from "./common";


function autoInjectMoonshineControlElements(modelURL: String) {
  const styles = `
  .moonshine-container {
    position: relative;
    display: inline-block;
    width: 100%;
  }

  .moonshine-button {
    position: absolute;
    max-width: 32px;
    max-height: 32px;
    top: 0;
    right: 0;
  }
`;

  const styleElement = document.createElement("style");
  styleElement.type = "text/css";
  document.head.appendChild(styleElement);
  styleElement.innerHTML = styles;

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

  function wrapAndReinjectInputElement(inputElement: Element) {
    const container = document.createElement("div");
    container.className = "moonshine-container";

    const button = document.createElement("div");
    button.className = "moonshine-button";

    const targetID = inputElement.id ? inputElement.id : getRandomID();
    button.setAttribute("data-moonshine-target", "#" + targetID);
    if (!inputElement.id) {
      inputElement.id = targetID;
    }

    inputElement.parentNode?.replaceChild(container, inputElement);
    container.appendChild(inputElement);
    container.appendChild(button);

    // squeeze button into smaller inputs if they exceed the button's max height
    // note: need to get the injected button from the DOM to determine its actual height
    const injectedButton = document.querySelector(
      '[data-moonshine-target="#' + targetID + '"]'
    );
    const computedButtonHeight = parseInt(
      window.getComputedStyle(injectedButton).getPropertyValue("max-height"),
      10
    );
    const inputRect = inputElement.getBoundingClientRect();

    if (inputRect.height < computedButtonHeight) {
      button.style.height = inputRect.height + "px";
      button.style.width = inputRect.height + "px";
    }

    // vertically center the button if the input height is close to (but greater than) the button height
    if (
      inputRect.height < 2 * computedButtonHeight &&
      inputRect.height > computedButtonHeight
    ) {
      button.style.top = (inputRect.height - computedButtonHeight) / 2 + "px";
    }

    container.style.width = inputRect.width;
  }

  // query selectors for each type of input element we want to add buttons to
  const injectionQuerySelectors = [
    "textarea",
    'input[type="text"], input[type="search"]',
    'div[contenteditable="true"]',
    'span[contenteditable="true"]',
  ];

  injectionQuerySelectors.forEach((querySelector) => {
    const elements = document.querySelectorAll(querySelector);
    elements.forEach((element) => {
      console.log("Injected Moonshine at " + element);
      wrapAndReinjectInputElement(element);
    });
  });

  initMoonshineControlElements(modelURL);
}

export { autoInjectMoonshineControlElements }