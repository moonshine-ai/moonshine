# @gerrit0/mini-shiki

This is a re-bundled version of [Shiki](https://shiki.style/) which strips out
the dependencies which aren't necessary for [TypeDoc](https://typedoc.org/)'s usage.

## Why?

Compare Shiki's dependency tree:

<img src="static/shiki-dependency-tree.svg">

To this package's dependency tree:

<img src="static/mini-shiki-dependency-tree.svg">

The Shiki maintainers [have declined](https://github.com/shikijs/shiki/issues/844) to split
up the package in a way which makes it possible to avoid these dependencies when just relying
on shikijs published packages.

## Releases

This package will be released and keep the same major/minor version numbers as Shiki.
Patch versions will generally be the same as Shiki, but may differ if adjustments are
necessary to fix compatibility issues.

## ESM / CommonJS

This package is ESM, but does not use top level await, so may be `require`d in
Node 23, or Node 20.17+ with the `--experimental-require-module` flag.

## Usage

```js
import {
  codeToTokensWithThemes,
  createShikiInternal,
  createOnigurumaEngine,
  bundledLanguages,
  bundledThemes,
  loadBuiltinWasm,
} from "@gerrit0/mini-shiki";

await loadBuiltinWasm();
const shiki = await createShikiInternal({
  engine: createOnigurumaEngine(),
  langs: [bundledLanguages.typescript],
  themes: [bundledThemes["light-plus"]],
});

const lines = codeToTokensWithThemes(shiki, "console.log('Hello world!')", {
  themes: { light: "light-plus" },
  lang: "typescript",
});
```
