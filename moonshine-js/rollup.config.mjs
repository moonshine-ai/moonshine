import { nodeResolve } from "@rollup/plugin-node-resolve";
import terser from "@rollup/plugin-terser";
import typescript from "@rollup/plugin-typescript";
import css from "rollup-plugin-import-css";
import svg from "rollup-plugin-svg-import";

const plugins = [
    nodeResolve({ browser: true }),
    typescript(),
    svg({
      stringify: true
    }),
    css(),
    terser(),
];

export default [
    // moonshine module for development/npm
    {
        input: "src/index.ts",
        output: {
            file: "dist/moonshine.min.js",
            format: "es",
        },
        plugins: plugins,
    },
    // manual moonshine-ify from CDN
    {
        input: "src/manual.ts",
        output: {
            file: "dist/moonshine.manual.min.js",
            format: "iife",
            name: "UsefulMoonshine",
        },
        plugins: plugins,
    },
    // auto moonshine-ify from CDN
    {
        input: "src/auto.ts",
        output: {
            file: "dist/moonshine.auto.min.js",
            format: "iife",
            name: "UsefulMoonshine",
        },
        plugins: plugins,
    },
];
