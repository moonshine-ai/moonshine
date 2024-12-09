import { nodeResolve } from '@rollup/plugin-node-resolve';
import terser from '@rollup/plugin-terser';
import typescript from '@rollup/plugin-typescript';

export default [
  // manual moonshine-ify
  {
    input: "src/manual.ts",
    external: ["onnxruntime-web"], // assumes onnxruntime-web <script> will be included separately
    output: {
      file: "dist/moonshine.min.js",
      format: "iife",
      name: "UsefulMoonshine",
    },
    plugins: [nodeResolve({browser: true}), typescript(), terser()]
  },
  // auto moonshine-ify
  {
    input: "src/auto.ts",
    output: {
      file: "dist/moonshine.auto.min.js",
      format: "iife",
      name: "UsefulMoonshine",
    },
    plugins: [nodeResolve({browser: true}), typescript(), terser()]
  },  
  // browser extension
  {
    input: "src/extension.ts",
    output: {
      file: "extension/moonshine.extension.min.js",
      format: "iife",
      name: "UsefulMoonshine",
    },
    plugins: [nodeResolve({browser: true}), typescript(), terser()]
  },
];
