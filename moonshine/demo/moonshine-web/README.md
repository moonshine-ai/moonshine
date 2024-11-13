# moonshine-web

This directory is a self-contained demo of the Moonshine models running directly on a user's device in a web browser using `onnxruntime-web`. You can try this demo out in our [HuggingFace space](https://huggingface.co/spaces/UsefulSensors/moonshine-web) or, alternatively, install and run it on your own device by following the instructions below. If you want to run Moonshine in the browser in your own projects, `src/moonshine.js` provides a bare-bones implementation of inferences using the ONNX models.

## Installation

You must have Node.js (or another JavaScript toolkit like [Bun](https://bun.sh/)) installed to get started. Install [Node.js](https://nodejs.org/en) if you don't have it already.

Once you have your JavaScript toolkit installed, clone the `moonshine` repo and navigate to this directory:

```shell
git clone git@github.com:usefulsensors/moonshine.git
cd moonshine/moonshine/demo/moonshine-web
```

Then install the project's dependencies:

```shell
npm install
```

The demo expects the Moonshine Tiny and Base ONNX models to be available in `public/moonshine/tiny` and `public/moonshine/base`, respectively. To preserve space, they are not included here. However, we've included a helper script that you can run to conveniently download them from HuggingFace:

```shell
npm run get-models
```

This project uses Vite for bundling and development. Run the following to start a development server and open the demo in your web browser:

```shell
npm run dev
```
