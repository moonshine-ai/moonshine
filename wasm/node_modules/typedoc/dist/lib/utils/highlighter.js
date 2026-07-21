import * as shiki from "@gerrit0/mini-shiki";
import * as JSX from "./jsx.js";
import { unique } from "./array.js";
import assert from "assert";
const aliases = new Map();
for (const lang of shiki.bundledLanguagesInfo) {
    for (const alias of lang.aliases || []) {
        aliases.set(alias, lang.id);
    }
}
const plaintextLanguages = ["txt", "text"];
const supportedLanguages = unique([
    ...plaintextLanguages,
    ...aliases.keys(),
    ...shiki.bundledLanguagesInfo.map((lang) => lang.id),
]).sort();
const supportedThemes = Object.keys(shiki.bundledThemes);
class DoubleHighlighter {
    highlighter;
    light;
    dark;
    schemes = new Map();
    constructor(highlighter, light, dark) {
        this.highlighter = highlighter;
        this.light = light;
        this.dark = dark;
    }
    supports(lang) {
        return this.highlighter.getLoadedLanguages().includes(lang);
    }
    highlight(code, lang) {
        const tokens = shiki.codeToTokensWithThemes(this.highlighter, code, {
            themes: { light: this.light, dark: this.dark },
            lang: lang,
        });
        const docEls = [];
        for (const line of tokens) {
            for (const token of line) {
                docEls.push(JSX.createElement("span", { class: this.getClass(token.variants) }, token.content));
            }
            docEls.push(JSX.createElement("br", null));
        }
        docEls.pop(); // Remove last <br>
        docEls.pop(); // Remove last <br>
        return JSX.renderElement(JSX.createElement(JSX.Fragment, null, docEls));
    }
    getStyles() {
        const style = [":root {"];
        const lightRules = [];
        const darkRules = [];
        let i = 0;
        for (const key of this.schemes.keys()) {
            const [light, dark] = key.split(" | ");
            style.push(`    --light-hl-${i}: ${light};`);
            style.push(`    --dark-hl-${i}: ${dark};`);
            lightRules.push(`    --hl-${i}: var(--light-hl-${i});`);
            darkRules.push(`    --hl-${i}: var(--dark-hl-${i});`);
            i++;
        }
        style.push(`    --light-code-background: ${this.highlighter.getTheme(this.light).bg};`);
        style.push(`    --dark-code-background: ${this.highlighter.getTheme(this.dark).bg};`);
        lightRules.push(`    --code-background: var(--light-code-background);`);
        darkRules.push(`    --code-background: var(--dark-code-background);`);
        style.push("}", "");
        style.push("@media (prefers-color-scheme: light) { :root {");
        style.push(...lightRules);
        style.push("} }", "");
        style.push("@media (prefers-color-scheme: dark) { :root {");
        style.push(...darkRules);
        style.push("} }", "");
        style.push(":root[data-theme='light'] {");
        style.push(...lightRules);
        style.push("}", "");
        style.push(":root[data-theme='dark'] {");
        style.push(...darkRules);
        style.push("}", "");
        for (i = 0; i < this.schemes.size; i++) {
            style.push(`.hl-${i} { color: var(--hl-${i}); }`);
        }
        style.push("pre, code { background: var(--code-background); }", "");
        return style.join("\n");
    }
    getClass(variants) {
        const key = `${variants["light"].color} | ${variants["dark"].color}`;
        let scheme = this.schemes.get(key);
        if (scheme == null) {
            scheme = `hl-${this.schemes.size}`;
            this.schemes.set(key, scheme);
        }
        return scheme;
    }
}
let shikiEngine;
let highlighter;
let ignoredLanguages;
export async function loadHighlighter(lightTheme, darkTheme, langs, ignoredLangs) {
    if (highlighter)
        return;
    ignoredLanguages = ignoredLangs;
    if (!shikiEngine) {
        await shiki.loadBuiltinWasm();
        shikiEngine = await shiki.createOnigurumaEngine();
    }
    const hl = await shiki.createShikiInternal({
        engine: shikiEngine,
        themes: [shiki.bundledThemes[lightTheme], shiki.bundledThemes[darkTheme]],
        langs: langs.map((lang) => shiki.bundledLanguages[lang]),
    });
    highlighter = new DoubleHighlighter(hl, lightTheme, darkTheme);
}
export function isSupportedLanguage(lang) {
    return ignoredLanguages?.includes(lang) || getSupportedLanguages().includes(lang);
}
export function getSupportedLanguages() {
    return supportedLanguages;
}
export function getSupportedThemes() {
    return supportedThemes;
}
export function isLoadedLanguage(lang) {
    return (plaintextLanguages.includes(lang) || ignoredLanguages?.includes(lang) || highlighter?.supports(lang) || false);
}
export function highlight(code, lang) {
    assert(highlighter, "Tried to highlight with an uninitialized highlighter");
    if (plaintextLanguages.includes(lang) || ignoredLanguages?.includes(lang)) {
        return JSX.renderElement(JSX.createElement(JSX.Fragment, null, code));
    }
    return highlighter.highlight(code, aliases.get(lang) ?? lang);
}
export function getStyles() {
    assert(highlighter, "Tried to highlight with an uninitialized highlighter");
    return highlighter.getStyles();
}
