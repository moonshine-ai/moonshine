import { nodeResolve } from '@rollup/plugin-node-resolve';
import terser from '@rollup/plugin-terser';

export default {
  input: "src/index.js",
  external: ["onnxruntime-web"], // assumes onnxruntime-web <script> will be included separately
  output: {
    file: "dist/moonshine.min.js",
    format: "iife",
    name: "UsefulMoonshine",
  },
  plugins: [nodeResolve({browser: true}), terser()]
};
