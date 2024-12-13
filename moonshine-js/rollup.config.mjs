import { nodeResolve } from '@rollup/plugin-node-resolve';
import terser from '@rollup/plugin-terser';
import typescript from '@rollup/plugin-typescript';

export default [
  // moonshine module for development/npm
  {
    input: "src/index.ts",
    output: {
      file: "dist/moonshine.min.js",
      format: "es"
    },
    plugins: [nodeResolve({browser: true}), typescript(), terser()]
  },
  // manual moonshine-ify from CDN
  {
    input: "src/manual.ts",
    output: {
      file: "dist/moonshine.manual.min.js",
      format: "iife",
      name: "UsefulMoonshine",
    },
    plugins: [nodeResolve({browser: true}), typescript(), terser()]
  },
  // auto moonshine-ify from CDN
  {
    input: "src/auto.ts",
    output: {
      file: "dist/moonshine.auto.min.js",
      format: "iife",
      name: "UsefulMoonshine",
    },
    plugins: [nodeResolve({browser: true}), typescript(), terser()]
  }
];
