/**
 * Parser for declaration references, see the [TSDoc grammar](https://github.com/microsoft/tsdoc/blob/main/tsdoc/src/beta/DeclarationReference.grammarkdown)
 * for reference. TypeDoc **does not** support the full grammar today. This is intentional, since the TSDoc
 * specified grammar allows the user to construct nonsensical declaration references such as `abc![def!ghi]`
 *
 * @module
 */
export const MeaningKeywords = [
    "class", // SymbolFlags.Class
    "interface", // SymbolFlags.Interface
    "type", // SymbolFlags.TypeAlias
    "enum", // SymbolFlags.Enum
    "namespace", // SymbolFlags.Module
    "function", // SymbolFlags.Function
    "var", // SymbolFlags.Variable
    "constructor", // SymbolFlags.Constructor
    "member", // SymbolFlags.ClassMember | SymbolFlags.EnumMember
    "event", //
    "call", // SymbolFlags.Signature (for __call)
    "new", // SymbolFlags.Signature (for __new)
    "index", // SymbolFlags.Signature (for __index)
    "complex", // Any complex type
    // TypeDoc specific
    "getter",
    "setter",
];
export function meaningToString(meaning) {
    let result = "";
    if (meaning.keyword) {
        result += meaning.keyword;
    }
    else if (meaning.label) {
        result += meaning.label;
    }
    if (typeof meaning.index === "number") {
        result += `(${meaning.index})`;
    }
    return result;
}
// <TAB> <VT> <FF> <SP> <NBSP> <ZWNBSP> <USP>
const WhiteSpace = /[\t\u2B7F\u240C \u00A0\uFEFF\p{White_Space}]/u;
const LineTerminator = "\r\n\u2028\u2029";
const Punctuators = "{}()[]!.#~:,";
const FutureReservedPunctuator = "{}@";
const NavigationPunctuator = ".#~";
const DecimalDigit = "0123456789";
const HexDigit = DecimalDigit + "abcdefABCDEF";
const SingleEscapeCharacter = `'"\\bfnrtv`;
const EscapeCharacter = SingleEscapeCharacter + DecimalDigit + "xu";
const UserLabelStart = "ABCDEFGHIJKLMNOPQRSTUVWXYZ_";
const UserLabelCharacter = UserLabelStart + DecimalDigit;
const SingleEscapeChars = {
    "'": "'",
    '"': '"',
    "\\": "\\",
    b: "\b",
    f: "\f",
    n: "\n",
    r: "\r",
    t: "\t",
    v: "\v",
};
// EscapeSequence::
//     SingleEscapeCharacter
//     NonEscapeCharacter
//     `0` [lookahead != DecimalDigit]
//     HexEscapeSequence
//     UnicodeEscapeSequence
function parseEscapeSequence(source, pos, end) {
    // SingleEscapeCharacter
    if (SingleEscapeCharacter.includes(source[pos])) {
        return [SingleEscapeChars[source[pos]], pos + 1];
    }
    // NonEscapeCharacter:: SourceCharacter but not one of EscapeCharacter or LineTerminator
    if (!(EscapeCharacter + LineTerminator).includes(source[pos])) {
        return [source[pos], pos + 1];
    }
    // `0` [lookahead != DecimalDigit]
    if (source[pos] === "0" &&
        pos + 1 < end &&
        !DecimalDigit.includes(source[pos + 1])) {
        return ["\x00", pos + 1];
    }
    // HexEscapeSequence:: x HexDigit HexDigit
    if (source[pos] === "x" &&
        pos + 2 < end &&
        HexDigit.includes(source[pos + 1]) &&
        HexDigit.includes(source[pos + 2])) {
        return [
            String.fromCharCode(parseInt(source.substring(pos + 1, pos + 3), 16)),
            pos + 3,
        ];
    }
    return parseUnicodeEscapeSequence(source, pos, end);
}
// UnicodeEscapeSequence::
//     `u` HexDigit HexDigit HexDigit HexDigit
//     `u` `{` CodePoint `}`
// CodePoint:: > |HexDigits| but only if MV of |HexDigits| ≤ 0x10FFFF
function parseUnicodeEscapeSequence(source, pos, end) {
    if (source[pos] !== "u" || pos + 1 >= end) {
        return;
    }
    if (HexDigit.includes(source[pos + 1])) {
        if (pos + 4 >= end ||
            !HexDigit.includes(source[pos + 2]) ||
            !HexDigit.includes(source[pos + 3]) ||
            !HexDigit.includes(source[pos + 4])) {
            return;
        }
        return [
            String.fromCharCode(parseInt(source.substring(pos + 1, pos + 5), 16)),
            pos + 5,
        ];
    }
    if (source[pos + 1] === "{" &&
        pos + 2 < end &&
        HexDigit.includes(source[pos + 2])) {
        let lookahead = pos + 3;
        while (lookahead < end && HexDigit.includes(source[lookahead])) {
            lookahead++;
        }
        if (lookahead >= end || source[lookahead] !== "}")
            return;
        const codePoint = parseInt(source.substring(pos + 2, lookahead), 16);
        if (codePoint <= 0x10ffff) {
            return [String.fromCodePoint(codePoint), lookahead + 1];
        }
    }
}
// String:: `"` StringCharacters? `"`
// StringCharacters:: StringCharacter StringCharacters?
// StringCharacter::
//   SourceCharacter but not one of `"` or `\` or LineTerminator
//   `\` EscapeSequence
export function parseString(source, pos, end) {
    let result = "";
    if (source[pos++] !== '"')
        return;
    while (pos < end) {
        if (source[pos] === '"') {
            return [result, pos + 1];
        }
        if (LineTerminator.includes(source[pos]))
            return;
        if (source[pos] === "\\") {
            const esc = parseEscapeSequence(source, pos + 1, end);
            if (!esc)
                return;
            result += esc[0];
            pos = esc[1];
            continue;
        }
        result += source[pos++];
    }
}
// ModuleSource:: String | ModuleSourceCharacters
export function parseModuleSource(source, pos, end) {
    if (pos >= end)
        return;
    if (source[pos] === '"') {
        return parseString(source, pos, end);
    }
    let lookahead = pos;
    while (lookahead < end &&
        !('"!' + LineTerminator).includes(source[lookahead])) {
        lookahead++;
    }
    if (lookahead === pos)
        return;
    return [source.substring(pos, lookahead), lookahead];
}
// SymbolReference:
//     ComponentPath Meaning?
//     Meaning
export function parseSymbolReference(source, pos, end) {
    const path = parseComponentPath(source, pos, end);
    pos = path?.[1] ?? pos;
    const meaning = parseMeaning(source, pos, end);
    pos = meaning?.[1] ?? pos;
    if (path || meaning) {
        return [{ path: path?.[0], meaning: meaning?.[0] }, pos];
    }
}
// Component::
//     String
//     ComponentCharacters
//     `[` DeclarationReference `]` <--- THIS ONE IS NOT IMPLEMENTED.
export function parseComponent(source, pos, end) {
    if (pos < end && source[pos] === '"') {
        return parseString(source, pos, end);
    }
    let lookahead = pos;
    while (lookahead < end &&
        !('"' +
            Punctuators +
            FutureReservedPunctuator +
            LineTerminator).includes(source[lookahead]) &&
        !WhiteSpace.test(source[lookahead])) {
        lookahead++;
    }
    if (lookahead === pos)
        return;
    return [source.substring(pos, lookahead), lookahead];
}
// ComponentPath:
//     Component
//     ComponentPath `.` Component                      // Navigate via 'exports' of |ComponentPath|
//     ComponentPath `#` Component                      // Navigate via 'members' of |ComponentPath|
//     ComponentPath `~` Component                      // Navigate via 'locals' of |ComponentPath|
export function parseComponentPath(source, pos, end) {
    const components = [];
    let component = parseComponent(source, pos, end);
    if (!component)
        return;
    pos = component[1];
    components.push({ navigation: ".", path: component[0] });
    while (pos < end && NavigationPunctuator.includes(source[pos])) {
        const navigation = source[pos];
        pos++;
        component = parseComponent(source, pos, end);
        if (!component) {
            return;
        }
        pos = component[1];
        components.push({ navigation, path: component[0] });
    }
    return [components, pos];
}
// The TSDoc specification permits the first four branches of Meaning. TypeDoc adds the UserLabel
// branch so that the @label tag can be used with this form of declaration references.
// Meaning:
//     `:` MeaningKeyword                            // Indicates the meaning of a symbol (i.e. ':class')
//     `:` MeaningKeyword `(` DecimalDigits `)`      // Indicates an overloaded meaning (i.e. ':function(1)')
//     `:` `(` DecimalDigits `)`                     // Shorthand for an overloaded meaning (i.e. `:(1)`)
//     `:` DecimalDigits                             // Shorthand for an overloaded meaning (i.e. ':1')
//     `:` UserLabel                                 // Indicates a user defined label via {@label CUSTOM_LABEL}
//
// UserLabel:
//     UserLabelStart UserLabelCharacter*
export function parseMeaning(source, pos, end) {
    if (source[pos++] !== ":")
        return;
    const keyword = MeaningKeywords.find((kw) => pos + kw.length <= end && source.startsWith(kw, pos));
    if (keyword) {
        pos += keyword.length;
    }
    if (!keyword && UserLabelStart.includes(source[pos])) {
        let lookahead = pos + 1;
        while (lookahead < end &&
            UserLabelCharacter.includes(source[lookahead])) {
            lookahead++;
        }
        return [{ label: source.substring(pos, lookahead) }, lookahead];
    }
    if (pos + 1 < end &&
        source[pos] === "(" &&
        DecimalDigit.includes(source[pos + 1])) {
        let lookahead = pos + 1;
        while (lookahead < end && DecimalDigit.includes(source[lookahead])) {
            lookahead++;
        }
        if (lookahead < end && source[lookahead] === ")") {
            return [
                {
                    keyword,
                    index: parseInt(source.substring(pos + 1, lookahead)),
                },
                lookahead + 1,
            ];
        }
    }
    if (!keyword && pos < end && DecimalDigit.includes(source[pos])) {
        let lookahead = pos;
        while (lookahead < end && DecimalDigit.includes(source[lookahead])) {
            lookahead++;
        }
        return [
            {
                index: parseInt(source.substring(pos, lookahead)),
            },
            lookahead,
        ];
    }
    if (keyword) {
        return [{ keyword }, pos];
    }
}
// // NOTE: The following grammar is incorrect as |SymbolReference| and |ModuleSource| have an
// //       ambiguous parse. The correct solution is to use a cover grammar to parse
// //       |SymbolReference| until we hit a `!` and then reinterpret the grammar.
// DeclarationReference:
//   [empty]
//   SymbolReference                               // Shorthand reference to symbol
//   ModuleSource `!`                              // Reference to a module
//   ModuleSource `!` SymbolReference              // Reference to an export of a module
//   ModuleSource `!` `~` SymbolReference          // Reference to a local of a module
//   `!` SymbolReference                           // Reference to global symbol
export function parseDeclarationReference(source, pos, end) {
    let moduleSource;
    let symbolReference;
    let resolutionStart = "local";
    let topLevelLocalReference = false;
    const moduleSourceOrSymbolRef = parseModuleSource(source, pos, end);
    if (moduleSourceOrSymbolRef) {
        if (moduleSourceOrSymbolRef[1] < end &&
            source[moduleSourceOrSymbolRef[1]] === "!") {
            // We had a module source!
            pos = moduleSourceOrSymbolRef[1] + 1;
            resolutionStart = "global";
            moduleSource = moduleSourceOrSymbolRef[0];
            // We might be referencing a local of a module
            if (source[pos] === "~") {
                topLevelLocalReference = true;
                pos++;
            }
        }
    }
    else if (source[pos] === "!") {
        pos++;
        resolutionStart = "global";
    }
    const ref = parseSymbolReference(source, pos, end);
    if (ref) {
        symbolReference = ref[0];
        if (topLevelLocalReference && symbolReference.path?.length) {
            symbolReference.path[0].navigation = "~";
        }
        pos = ref[1];
    }
    if (!moduleSource && !symbolReference)
        return;
    return [
        {
            moduleSource,
            resolutionStart,
            symbolReference,
        },
        pos,
    ];
}
