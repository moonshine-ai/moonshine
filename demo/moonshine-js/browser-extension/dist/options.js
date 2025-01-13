const saveOptions = () => {
    const modelSelected = document.getElementById("model").value;
    const enableEverywhereChecked = document.getElementById("enableEverywhere").checked;

    chrome.storage.sync.set(
        { model: modelSelected, enableEverywhere: enableEverywhereChecked },
        () => {}
    );
};

const restoreOptions = () => {
    chrome.storage.sync.get(
        { model: "tiny",  enableEverywhere: false },
        (items) => {
            document.getElementById("model").value = 
                items.model
            document.getElementById("enableEverywhere").checked =
                items.enableEverywhere
        }
    );
};

document.addEventListener("DOMContentLoaded", restoreOptions);
document
    .getElementById("save")
    .addEventListener("click", saveOptions);