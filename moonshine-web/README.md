# moonshine-web

Dev version of JS library for using Moonshine on the web.

## Setup

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
