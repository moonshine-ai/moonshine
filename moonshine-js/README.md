# moonshine-js

This package provides quick and easy integration of client-side, on-device speech-to-text on web pages and in web applications with Useful Sensors' Moonshine models. It includes the following:

- 🌙 **Easy UI integration**: add a single `<script>` that automatically adds speech-to-text to all text inputs on a page, or design your own UI integration in a few easy steps.
- 🌙 **Simple client-side speech-to-text interfaces**: just import [a single class]() to handle user mic permissions, model loading, and audio transcription, or use our [lower-level model implementation]() to easily transcribe audio from other sources.

If you are looking for a quick integration with your React application, check out [`moonshine-react`]().

_Note: This package is currently in beta, and breaking changes may occur between versions. User feedback and developer contributions are welcome._

## Quickstart

### 1. Include via CDN
You can include `moonshine-js` on your site using our CDN-hosted solutions. We provide a few options for rapid integration via CDN:

#### Auto: Automatically adds a speech-to-text button to all input fields on your page. 
Include the following `<script>` tag:

```html
<head>
    <script type="module" src="https://cdn.jsdelivr.net/npm/@usefulsensors/moonshine-js@latest/dist/moonshine.auto.min.js"></script>
</head>
```

#### Manual: Allows you to customize your speech-to-text UI with your own HTML layout. 
Include the following `<script>` tag:

```html
<head>
    <script type="module" src="https://cdn.jsdelivr.net/npm/@usefulsensors/moonshine-js@latest/dist/moonshine.manual.min.js"></script>
</head>
```

Now you can specify which text inputs should be speech-enabled and customize their appearance. Add a `data-moonshine-target` attribute to any clickable element that you want to trigger speech-to-text, and set its value to a CSS selector of the element(s) that should be updated with the transcription output. For example:

```html
<textarea id="myTextArea"></textarea>
<button data-moonshine-target="#myTextArea"></button>
```

The manual option also provides options for customizing your layout; [see some examples here]().

### 2. Install with `npm`
You can also install `moonshine-js` via `npm`, giving you greater control over how you build your application. Just run:

```
npm install @usefulsensors/moonshine-js
```

There are two key modules provided by `moonshine-js`:

- `MoonshineTranscriber`: handles user mic access, starting/stopping transcription of streaming audio, and transcription generation. 
- `MoonshineModel`: lower-level implementation of transcription generation from input audio using our ONNX models.

[Read the docs]() for more information.
