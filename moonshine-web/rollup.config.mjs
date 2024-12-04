import { nodeResolve } from '@rollup/plugin-node-resolve';
import terser from '@rollup/plugin-terser';
import typescript from '@rollup/plugin-typescript';

export default [
  // manual moonshine-ify (using data-moonshine-target)
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
];
