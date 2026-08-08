import { ParameterHint, ParameterType, } from "./declaration.js";
import { getSupportedLanguages, getSupportedThemes } from "../highlighter.js";
function hasHint(parameter) {
    return ((parameter.type ?? ParameterType.String) === ParameterType.String &&
        "hint" in parameter);
}
/**
 * Prepare parameter information for the {@link toString} method.
 *
 * @param scope  The scope of the parameters whose help should be returned.
 * @returns The columns and lines for the help of the requested parameters.
 */
function getParameterHelp(options, i18n) {
    const parameters = options.getDeclarations();
    parameters.sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: "base" }));
    const names = [];
    const helps = [];
    let margin = 0;
    for (const parameter of parameters) {
        if (!parameter.help || parameter.configFileOnly) {
            continue;
        }
        let name = " --" + parameter.name;
        if (hasHint(parameter)) {
            name += " " + ParameterHint[parameter.hint].toUpperCase();
        }
        names.push(name);
        helps.push(typeof parameter.help === "string"
            ? parameter.help
            : parameter.help(i18n));
        margin = Math.max(name.length, margin);
    }
    return { names, helps, margin };
}
function toEvenColumns(values, maxLineWidth) {
    const columnWidth = values.reduce((acc, val) => Math.max(acc, val.length), 0) + 2;
    const numColumns = Math.max(1, Math.floor(maxLineWidth / columnWidth));
    let line = "";
    const out = [];
    for (let i = 0; i < values.length; ++i) {
        if (i !== 0 && i % numColumns === 0) {
            out.push(line);
            line = "";
        }
        line += values[i].padEnd(columnWidth);
    }
    if (line != "") {
        out.push(line);
    }
    return out;
}
export function getOptionsHelp(options, i18n) {
    const output = ["typedoc path/to/entry.ts", "", "Options:"];
    const columns = getParameterHelp(options, i18n);
    for (let i = 0; i < columns.names.length; i++) {
        const usage = columns.names[i];
        const description = columns.helps[i];
        output.push(usage.padEnd(columns.margin + 2) + description);
    }
    output.push("", "Supported highlighting languages:", ...toEvenColumns(getSupportedLanguages(), 80));
    output.push("", "Supported highlighting themes:", ...toEvenColumns(getSupportedThemes(), 80));
    return output.join("\n");
}
