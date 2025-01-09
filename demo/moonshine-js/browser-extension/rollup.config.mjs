import { nodeResolve } from '@rollup/plugin-node-resolve';
import terser from '@rollup/plugin-terser';
import typescript from '@rollup/plugin-typescript';

export default [
  {
    input: "src/index.ts",
    output: {
      file: "dist/moonshine.extension.min.js",
      format: "iife",
      name: "UsefulMoonshine",
    },
    plugins: [nodeResolve({browser: true}), typescript(), terser()]
  },
];
