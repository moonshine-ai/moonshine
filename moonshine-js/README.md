# moonshine-js

Moonshine Web is a JavaScript package that provides quick and easy integration of client-side, on-device speech-to-text on web pages and in web applications.

_Note: This package is currently in alpha, and breaking changes may occur between versions. User feedback and developer contributions are welcome._

## Quickstart

You can easily include `moonshine-web` on your site using our CDN-hosted solution. Simply add the following inside the `<head>` tag:

```html
<head>
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js"></script>
    <script type="module" src="https://cdn.jsdelivr.net/npm/useful-moonshine-web@.../dist/moonshine.min.js"></script>
</head>
```

The first script includes the required `onnxruntime-web` package.

## Usage

Using `moonshine-web` requires adding the following elements to your page:

1. _Target elements_. These are elements that you want to output the results of speech-to-text to.
2. _Control elements_, specified by adding a `data-moonshine-target` attribute that specifies a corresponding target element selector. These are clickable elements, like `<button>`s or `<div>`s, that should trigger speech-to-text.

Control elements point to corresponding target elements that should recieve the speech-to-text output. The following is a minimal example:

```html
<textarea id="myTextArea"></textarea>
<button data-moonshine-target="#myTextArea"></button>
```

When a page visitor clicks the button, mic access will be requested, the model will be loaded on their device, and live speech-to-text will begin. As the visitor speaks, the transcription will be updated and output to the target `<textarea>`.

### Lifecycle Icons

During each phase of the speech-to-text lifecycle (idle, model loading, and transcribing), a corresponding event is dispatched on the currently-active control element, which changes the icon currently displayed. We provide default icons; however, you may wish to override these to better match the design of your site. You have two options for doing so:

#### Option 1: Inline Override

You can specify what to display at each step of the lifecycle by adding inline `<span>` tags to a control element, as follows:

```html
<textarea id="myTextArea"></textarea>
<button data-moonshine-target="#myTextArea">
    <span data-moonshine-idle>This is custom content that will show while waiting to be clicked to start speech-to-text...</span>
    <span data-moonshine-loading>...and while loading the model...</span>
    <span data-moonshine-transcribing>...and during transcription!</span>
</button>
```

Note that inline overrides will only apply to the control element for which they are defined. If you wish to use a global override that automatically applies to all control elements on the page, read on...

#### Option 2: Global Override

To override the styling for every Moonshine control element on the page, simply add the `data-moonshine-template` attribute to the element that you want to serve as the "template" for all other control elements on the page:

```html
<textarea id="myTextArea"></textarea>
<button data-moonshine-target="#myTextArea" data-moonshine-template>
    <span data-moonshine-idle>This is custom content that will show while waiting to be clicked to start speech-to-text...</span>
    <span data-moonshine-loading>...and while loading the model...</span>
    <span data-moonshine-transcribing>...and during transcription!</span>
</button>
<textarea id="myOtherTextArea"></textarea>
<button data-moonshine-target="#myOtherTextArea">
    <!--- The content from the other button will be automatically filled in here; no need to define it again. -->
</button>
```

## TODO
* [x] IIFE- and CDN-based quickstart implementation

* [ ] Vanilla component-based impementation

* [ ] Component-based implementation for popular frontend frameworks (e.g., React)

## Developer Setup

1. Install `flask`, JS dependencies, and get models:

```shell
pip install Flask
npm install
npm run get-models
```

Flask is used to run a minimal web server for serving up the `example/index.html` page, minified JS bundle, and `.ort` files. 

2. Run dev server:

```shell
npm run dev
```

3. Open http://localhost:5000 in browser.
