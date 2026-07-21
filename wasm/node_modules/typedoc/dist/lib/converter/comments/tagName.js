/**
 * Determines the name of the parameter/template/property from the tag content
 * when processing `@param x`
 */
export function extractTagName(text) {
    let pos = skipWs(text, 0);
    // Extract the "name" part of the comment, this might also include a default
    // value if bracketed.
    let nameStart = pos;
    let bracketDepth = 0;
    let stringChar = "";
    while (pos < text.length && (bracketDepth > 0 || /\S/.test(text[pos]))) {
        if (stringChar) {
            if (text[pos] == stringChar) {
                stringChar = "";
                ++pos;
            }
            else if (text[pos] == "\\") {
                pos += 2;
            }
            else {
                ++pos;
            }
        }
        else {
            if ("\"'`".includes(text[pos])) {
                stringChar = text[pos];
            }
            else if (text[pos] == "[") {
                ++bracketDepth;
            }
            else if (text[pos] == "]") {
                --bracketDepth;
            }
            ++pos;
        }
    }
    let nameEnd = pos;
    if (text[nameStart] === "[") {
        nameStart = skipWs(text, nameStart + 1);
        nameEnd = skipWith(text, nameStart, /[^\s=\]]/);
    }
    // Skip any whitespace & an optional dash before the description
    pos = skipWith(text, pos, /[\s-]/);
    return {
        name: text.substring(nameStart, nameEnd),
        newText: text.substring(pos),
    };
}
function skipWs(text, pos) {
    return skipWith(text, pos, /\s/);
}
function skipWith(text, pos, reg) {
    while (pos < text.length && reg.test(text[pos])) {
        ++pos;
    }
    return pos;
}
