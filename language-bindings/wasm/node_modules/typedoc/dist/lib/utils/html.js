import { assertNever } from "./general.js";
import { htmlEntities } from "./html-entities.js";
const htmlEntitiesTrie = {};
for (const [name, data] of Object.entries(htmlEntities)) {
    let current = htmlEntitiesTrie;
    for (let i = 0; i < name.length; ++i) {
        current.children ||= {};
        current = current.children[name.charCodeAt(i)] ||= {};
    }
    current.data = data;
}
const htmlEscapes = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
};
export function escapeHtml(html) {
    return html.replace(/[&<>'"]/g, (c) => htmlEscapes[c]);
}
/**
 * Replaces non-[URL code points](https://url.spec.whatwg.org/#url-code-points)
 * with an underscore. Also disallows some additional special characters which either
 * result in confusing file names or invalid file names on some platform.
 *
 * Ref: #2714
 */
export function createNormalizedUrl(url) {
    const codePoints = [...url].map((c) => c.codePointAt(0));
    for (let i = 0; i < codePoints.length; ++i) {
        if (isalnum(codePoints[i]))
            continue;
        switch (codePoints[i]) {
            // case Chars.BANG: // commonly used for shell history access
            // case Chars.DOLLAR: // variable reference in shells
            // case Chars.AMPERSAND: // confusing in urls with params
            // case Chars.APOSTROPHE: // sometimes permitted for quoting
            case Chars.LEFT_PAREN:
            case Chars.RIGHT_PAREN:
            // case Chars.ASTERISK: // not allowed in file names on windows
            case Chars.PLUS:
            case Chars.COMMA:
            case Chars.DASH:
            case Chars.DOT:
            // case Chars.SOLIDUS: // not allowed in file names
            // case Chars.COLON: // not allowed in file names on windows
            // case Chars.SEMICOLON: // appears suspiciously close to colon
            // case Chars.EQUALS: // confusing in urls with params
            // case Chars.QUESTION_MARK: // not allowed in file names on windows
            // case Chars.AT: // avoid confusing some (bad) software that thinks this is an email
            case Chars.UNDERSCORE:
                // case Chars.TILDE: // reference to $HOME on Linux
                continue;
        }
        if (0xa0 <= codePoints[i] && codePoints[i] <= 0x10fffd) {
            if (!isSurrogate(codePoints[i])) {
                continue;
            }
        }
        codePoints[i] = Chars.UNDERSCORE;
    }
    return String.fromCharCode(...codePoints);
}
var Chars;
(function (Chars) {
    Chars[Chars["EOF"] = -1] = "EOF";
    Chars[Chars["NULL"] = 0] = "NULL";
    Chars[Chars["TAB"] = 9] = "TAB";
    Chars[Chars["LF"] = 10] = "LF";
    Chars[Chars["FF"] = 12] = "FF";
    Chars[Chars["SPACE"] = 32] = "SPACE";
    Chars[Chars["NUMBER_SIGN"] = 35] = "NUMBER_SIGN";
    Chars[Chars["BANG"] = 33] = "BANG";
    Chars[Chars["QUOTATION_MARK"] = 34] = "QUOTATION_MARK";
    Chars[Chars["DOLLAR"] = 36] = "DOLLAR";
    Chars[Chars["AMPERSAND"] = 38] = "AMPERSAND";
    Chars[Chars["APOSTROPHE"] = 39] = "APOSTROPHE";
    Chars[Chars["LEFT_PAREN"] = 40] = "LEFT_PAREN";
    Chars[Chars["RIGHT_PAREN"] = 41] = "RIGHT_PAREN";
    Chars[Chars["ASTERISK"] = 42] = "ASTERISK";
    Chars[Chars["PLUS"] = 43] = "PLUS";
    Chars[Chars["COMMA"] = 44] = "COMMA";
    Chars[Chars["DASH"] = 45] = "DASH";
    Chars[Chars["DOT"] = 46] = "DOT";
    Chars[Chars["SOLIDUS"] = 47] = "SOLIDUS";
    Chars[Chars["ZERO"] = 48] = "ZERO";
    Chars[Chars["NINE"] = 57] = "NINE";
    Chars[Chars["COLON"] = 58] = "COLON";
    Chars[Chars["SEMICOLON"] = 59] = "SEMICOLON";
    Chars[Chars["LESS_THAN"] = 60] = "LESS_THAN";
    Chars[Chars["EQUALS"] = 61] = "EQUALS";
    Chars[Chars["GREATER_THAN"] = 62] = "GREATER_THAN";
    Chars[Chars["QUESTION_MARK"] = 63] = "QUESTION_MARK";
    Chars[Chars["AT"] = 64] = "AT";
    Chars[Chars["UPPERCASE_A"] = 65] = "UPPERCASE_A";
    Chars[Chars["UPPERCASE_F"] = 70] = "UPPERCASE_F";
    Chars[Chars["UPPERCASE_X"] = 88] = "UPPERCASE_X";
    Chars[Chars["UPPERCASE_Z"] = 90] = "UPPERCASE_Z";
    Chars[Chars["UNDERSCORE"] = 95] = "UNDERSCORE";
    Chars[Chars["GRAVE_ACCENT"] = 96] = "GRAVE_ACCENT";
    Chars[Chars["LOWERCASE_A"] = 97] = "LOWERCASE_A";
    Chars[Chars["LOWERCASE_F"] = 102] = "LOWERCASE_F";
    Chars[Chars["LOWERCASE_X"] = 120] = "LOWERCASE_X";
    Chars[Chars["LOWERCASE_Z"] = 122] = "LOWERCASE_Z";
    Chars[Chars["TILDE"] = 126] = "TILDE";
})(Chars || (Chars = {}));
function isalpha(ch) {
    return Chars.LOWERCASE_A <= (ch | 0x20) && (ch | 0x20) <= Chars.LOWERCASE_Z;
}
function isdigit(ch) {
    return Chars.ZERO <= ch && ch <= Chars.NINE;
}
function isalnum(ch) {
    return isalpha(ch) || isdigit(ch);
}
function isxdigit(ch) {
    return (isdigit(ch) ||
        (Chars.LOWERCASE_A <= (ch | 0x20) && (ch | 0x20) <= Chars.LOWERCASE_F));
}
export var ParserState;
(function (ParserState) {
    ParserState[ParserState["BeforeAttributeName"] = 0] = "BeforeAttributeName";
    ParserState[ParserState["AfterAttributeName"] = 1] = "AfterAttributeName";
    ParserState[ParserState["BeforeAttributeValue"] = 2] = "BeforeAttributeValue";
    ParserState[ParserState["END"] = 3] = "END";
})(ParserState || (ParserState = {}));
/**
 * Parser for HTML attributes, each call to {@link step} will
 * pause the parser at key points used to extract relative links from markdown
 *
 * The parser will pause at the points marked with `^`:
 *
 * ```text
 * attr="value" attr='value' attr=value attr attr2 />
 *     ^       ^    ^       ^    ^     ^    ^     ^^
 *     BeforeValue  |       |    |     |    |     ||
 *             BeforeName   |    |     |    |     ||
 *                  BeforeValue  |     |    |     ||
 *                          BeforeName |    |     ||
 *                               BeforeValue|     ||
 *                                     BeforeName ||
 *                                          AfterName
 *                                                AfterName
 *                                                 END
 * ```
 */
export class HtmlAttributeParser {
    text;
    pos;
    state = ParserState.BeforeAttributeName;
    currentAttributeName = "";
    currentAttributeValueStart = -1;
    currentAttributeValueEnd = -1;
    currentAttributeValue = "";
    temporaryBuffer = [];
    characterReferenceCode = 0;
    constructor(text, pos = 0) {
        this.text = text;
        this.pos = pos;
    }
    step() {
        switch (this.state) {
            case ParserState.BeforeAttributeName:
                this.beforeAttributeName();
                return;
            case ParserState.AfterAttributeName:
                this.afterAttributeName();
                return;
            case ParserState.BeforeAttributeValue:
                this.beforeAttributeValue();
                return;
            case ParserState.END:
                return; // Do nothing
        }
        /* c8 ignore next */
        assertNever(this.state);
    }
    peek() {
        const ch = this.text.charCodeAt(this.pos);
        return isNaN(ch) ? Chars.EOF : ch;
    }
    consume() {
        const ch = this.peek();
        ++this.pos;
        return ch;
    }
    // https://html.spec.whatwg.org/multipage/parsing.html#before-attribute-name-state
    beforeAttributeName() {
        this.currentAttributeName = "";
        this.currentAttributeValue = "";
        loop: for (;;) {
            switch (this.consume()) {
                case Chars.TAB:
                case Chars.LF:
                case Chars.FF:
                case Chars.SPACE:
                    break;
                case Chars.SOLIDUS:
                case Chars.GREATER_THAN:
                case Chars.EOF:
                    --this.pos;
                    this.afterAttributeName();
                    break loop;
                case Chars.EQUALS:
                // Unexpected equals sign before attribute name parse error.
                // fall through
                default:
                    --this.pos;
                    this.attributeName();
                    break loop;
            }
        }
    }
    // https://html.spec.whatwg.org/multipage/parsing.html#attribute-name-state
    attributeName() {
        const startPos = this.pos;
        loop: for (;;) {
            const ch = this.consume();
            switch (ch) {
                case Chars.TAB:
                case Chars.LF:
                case Chars.FF:
                case Chars.SPACE:
                case Chars.SOLIDUS:
                case Chars.GREATER_THAN:
                case Chars.EOF:
                    --this.pos;
                    this.state = ParserState.AfterAttributeName;
                    break loop;
                case Chars.EQUALS:
                    this.state = ParserState.BeforeAttributeValue;
                    break loop;
                case Chars.QUOTATION_MARK:
                case Chars.APOSTROPHE:
                case Chars.LESS_THAN:
                // This is an unexpected-character-in-attribute-name parse error. Treat it as per the "anything else" entry below.
                // fall through
                default:
                    // Do nothing, we collect the attribute name after the loop
                    break;
            }
        }
        if (this.state === ParserState.BeforeAttributeValue) {
            this.currentAttributeName = this.text
                .substring(startPos, this.pos - 1)
                .toLowerCase();
        }
        else {
            this.currentAttributeName = this.text
                .substring(startPos, this.pos)
                .toLowerCase();
        }
    }
    // https://html.spec.whatwg.org/multipage/parsing.html#after-attribute-name-state
    afterAttributeName() {
        loop: for (;;) {
            switch (this.consume()) {
                case Chars.TAB:
                case Chars.LF:
                case Chars.FF:
                case Chars.SPACE:
                    break; // Ignore the character
                case Chars.SOLIDUS:
                    this.state = ParserState.END; // self-closing start tag state
                    break loop;
                case Chars.EQUALS:
                    this.state = ParserState.BeforeAttributeValue;
                    break loop;
                case Chars.GREATER_THAN:
                    this.state = ParserState.END; // data state
                    break loop;
                case Chars.EOF:
                    this.state = ParserState.END; // eof-in-tag parse error
                    break loop;
                default:
                    --this.pos;
                    this.attributeName();
                    break loop;
            }
        }
    }
    // https://html.spec.whatwg.org/multipage/parsing.html#before-attribute-value-state
    beforeAttributeValue() {
        loop: for (;;) {
            switch (this.consume()) {
                case Chars.TAB:
                case Chars.LF:
                case Chars.FF:
                case Chars.SPACE:
                    break; // Ignore the character
                case Chars.QUOTATION_MARK:
                    this.attributeValueDoubleQuoted();
                    break loop;
                case Chars.APOSTROPHE:
                    this.attributeValueSingleQuoted();
                    break loop;
                case Chars.GREATER_THAN:
                    this.state = ParserState.END; // missing-attribute-value parse error
                    break loop;
                default:
                    --this.pos;
                    this.attributeValueUnquoted();
                    break loop;
            }
        }
    }
    // https://html.spec.whatwg.org/multipage/parsing.html#attribute-value-(double-quoted)-state
    attributeValueDoubleQuoted() {
        this.currentAttributeValueStart = this.pos;
        loop: for (;;) {
            switch (this.consume()) {
                case Chars.QUOTATION_MARK:
                    this.currentAttributeValueEnd = this.pos - 1;
                    this.afterAttributeValueQuoted();
                    break loop;
                case Chars.AMPERSAND:
                    this.characterReference();
                    break;
                case Chars.NULL:
                    this.currentAttributeValue += String.fromCharCode(0xfffd);
                    break;
                case Chars.EOF:
                    this.currentAttributeValueEnd = this.pos;
                    this.state = ParserState.END; // eof-in-tag parse error
                    break loop;
                default:
                    this.currentAttributeValue += this.text[this.pos - 1];
                    break;
            }
        }
    }
    // https://html.spec.whatwg.org/multipage/parsing.html#attribute-value-(single-quoted)-state
    attributeValueSingleQuoted() {
        this.currentAttributeValueStart = this.pos;
        loop: for (;;) {
            switch (this.consume()) {
                case Chars.APOSTROPHE:
                    this.currentAttributeValueEnd = this.pos - 1;
                    this.afterAttributeValueQuoted();
                    break loop;
                case Chars.AMPERSAND:
                    this.characterReference();
                    break;
                case Chars.NULL:
                    this.currentAttributeValue += String.fromCharCode(0xfffd);
                    break;
                case Chars.EOF:
                    this.currentAttributeValueEnd = this.pos;
                    this.state = ParserState.END; // eof-in-tag parse error
                    break loop;
                default:
                    this.currentAttributeValue += this.text[this.pos - 1];
                    break;
            }
        }
    }
    // https://html.spec.whatwg.org/multipage/parsing.html#attribute-value-(unquoted)-state
    attributeValueUnquoted() {
        this.currentAttributeValueStart = this.pos;
        loop: for (;;) {
            switch (this.consume()) {
                case Chars.TAB:
                case Chars.LF:
                case Chars.FF:
                case Chars.SPACE:
                    this.currentAttributeValueEnd = this.pos - 1;
                    this.state = ParserState.BeforeAttributeName;
                    break loop;
                case Chars.AMPERSAND:
                    this.characterReference();
                    break;
                case Chars.GREATER_THAN:
                    this.currentAttributeValueEnd = this.pos;
                    this.state = ParserState.END;
                    break loop;
                case Chars.NULL:
                    this.currentAttributeValue += String.fromCharCode(0xfffd);
                    break;
                case Chars.EOF:
                    this.currentAttributeValueEnd = this.pos;
                    this.state = ParserState.END; // eof-in-tag parse error
                    break loop;
                case Chars.QUOTATION_MARK:
                case Chars.APOSTROPHE:
                case Chars.LESS_THAN:
                case Chars.EQUALS:
                case Chars.GRAVE_ACCENT:
                // This is an unexpected-character-in-unquoted-attribute-value parse error. Treat it as per the "anything else" entry below.
                // fall through
                default:
                    this.currentAttributeValue += this.text[this.pos - 1];
                    break;
            }
        }
    }
    // https://html.spec.whatwg.org/multipage/parsing.html#after-attribute-value-(quoted)-state
    afterAttributeValueQuoted() {
        switch (this.consume()) {
            case Chars.TAB:
            case Chars.LF:
            case Chars.FF:
            case Chars.SPACE:
                this.state = ParserState.BeforeAttributeName;
                break;
            case Chars.SOLIDUS:
            case Chars.GREATER_THAN:
            case Chars.EOF:
                this.state = ParserState.END;
                break;
            default:
                // This is a missing-whitespace-between-attributes parse error. Reconsume in the before attribute name state.
                --this.pos;
                this.state = ParserState.BeforeAttributeName;
                break;
        }
    }
    // https://html.spec.whatwg.org/multipage/parsing.html#character-reference-state
    characterReference() {
        this.temporaryBuffer = [Chars.AMPERSAND];
        const next = this.consume();
        if (isalnum(next)) {
            --this.pos;
            this.namedCharacterReference();
        }
        else if (next == Chars.NUMBER_SIGN) {
            this.temporaryBuffer.push(next);
            this.numericCharacterReference();
        }
        else {
            --this.pos;
            this.flushTemporaryBuffer();
        }
    }
    // https://html.spec.whatwg.org/multipage/parsing.html#named-character-reference-state
    // Intentionally only handling part of an attribute
    namedCharacterReference() {
        // Consume the maximum number of characters possible, where the consumed
        // characters are one of the identifiers in the first column of the named
        // character references table. Append each character to the temporary buffer
        // when it's consumed.
        let currentTrie = htmlEntitiesTrie;
        for (;;) {
            const ch = this.consume();
            this.temporaryBuffer.push(ch);
            if (currentTrie.children && ch in currentTrie.children) {
                currentTrie = currentTrie.children[ch];
            }
            else {
                --this.pos;
                this.temporaryBuffer.pop();
                const lastChar = this.temporaryBuffer[this.temporaryBuffer.length - 1];
                // If there is a match
                if (currentTrie.data) {
                    // If the character reference was consumed as part of an attribute,
                    // and the last character matched is not a U+003B SEMICOLON character (;),
                    // and the next input character is either a U+003D EQUALS SIGN character (=)
                    // or an ASCII alphanumeric, then, for historical reasons, flush code points
                    // consumed as a character reference and switch to the return state.
                    if (lastChar != Chars.SEMICOLON &&
                        (this.peek() == Chars.EQUALS || isalpha(this.peek()))) {
                        this.flushTemporaryBuffer();
                        return;
                    }
                    else {
                        // missing-semicolon-after-character-reference parse error
                        this.temporaryBuffer = currentTrie.data.p;
                        this.flushTemporaryBuffer();
                        return;
                    }
                }
                else {
                    this.flushTemporaryBuffer();
                    this.ambiguousAmpersand();
                    return;
                }
            }
        }
    }
    // https://html.spec.whatwg.org/multipage/parsing.html#ambiguous-ampersand-state
    ambiguousAmpersand() {
        const ch = this.consume();
        if (isalnum(ch)) {
            this.currentAttributeValue += String.fromCharCode(ch);
        }
        else {
            --this.pos;
            return;
        }
    }
    // https://html.spec.whatwg.org/multipage/parsing.html#numeric-character-reference-state
    numericCharacterReference() {
        this.characterReferenceCode = 0;
        const ch = this.consume();
        switch (ch) {
            case Chars.LOWERCASE_X:
            case Chars.UPPERCASE_X:
                this.temporaryBuffer.push(ch);
                this.hexadecimalCharacterReferenceStart();
                break;
            default:
                --this.pos;
                this.decimalCharacterReferenceStart();
                break;
        }
    }
    // https://html.spec.whatwg.org/multipage/parsing.html#hexadecimal-character-reference-start-state
    hexadecimalCharacterReferenceStart() {
        const ch = this.consume();
        if (isxdigit(ch)) {
            --this.pos;
            this.hexadecimalCharacterReference();
        }
        else {
            --this.pos;
            this.flushTemporaryBuffer();
        }
    }
    // https://html.spec.whatwg.org/multipage/parsing.html#decimal-character-reference-start-state
    decimalCharacterReferenceStart() {
        const ch = this.consume();
        if (isdigit(ch)) {
            --this.pos;
            this.decimalCharacterReference();
        }
        else {
            --this.pos;
            this.flushTemporaryBuffer();
        }
    }
    // https://html.spec.whatwg.org/multipage/parsing.html#hexadecimal-character-reference-state
    hexadecimalCharacterReference() {
        for (;;) {
            const ch = this.consume();
            if (isdigit(ch)) {
                this.characterReferenceCode *= 16;
                this.characterReferenceCode += ch - 0x30;
            }
            else if (Chars.UPPERCASE_A <= ch && ch <= Chars.UPPERCASE_F) {
                this.characterReferenceCode *= 16;
                this.characterReferenceCode += ch - 0x37;
            }
            else if (Chars.LOWERCASE_A <= ch && ch <= Chars.LOWERCASE_F) {
                this.characterReferenceCode *= 16;
                this.characterReferenceCode += ch - 0x57;
            }
            else if (ch === Chars.SEMICOLON) {
                this.numericCharacterReferenceEndState();
                return;
            }
            else {
                --this.pos;
                this.numericCharacterReferenceEndState();
                return;
            }
        }
    }
    // https://html.spec.whatwg.org/multipage/parsing.html#decimal-character-reference-state
    decimalCharacterReference() {
        for (;;) {
            const ch = this.consume();
            if (isdigit(ch)) {
                this.characterReferenceCode *= 10;
                this.characterReferenceCode += ch - 0x30;
            }
            else if (ch === Chars.SEMICOLON) {
                this.numericCharacterReferenceEndState();
                return;
            }
            else {
                --this.pos;
                this.numericCharacterReferenceEndState();
                return;
            }
        }
    }
    // https://html.spec.whatwg.org/multipage/parsing.html#numeric-character-reference-end-state
    numericCharacterReferenceEndState() {
        if (this.characterReferenceCode == 0) {
            // null-character-reference parse error
            this.characterReferenceCode = 0xfffd;
        }
        if (this.characterReferenceCode > 0x10ffff) {
            // character-reference-outside-unicode-range parse error
            this.characterReferenceCode = 0xfffd;
        }
        if (isSurrogate(this.characterReferenceCode)) {
            // surrogate-character-reference parse error
            this.characterReferenceCode = 0xfffd;
        }
        // If the number is a noncharacter, then this is a noncharacter-character-reference parse error.
        // ... and do nothing, so don't bother checking.
        // Handle replacements
        this.characterReferenceCode =
            characterReferenceCodePointReplacements.get(this.characterReferenceCode) ?? this.characterReferenceCode;
        this.temporaryBuffer = [this.characterReferenceCode];
        this.flushTemporaryBuffer();
    }
    flushTemporaryBuffer() {
        this.currentAttributeValue += String.fromCodePoint(...this.temporaryBuffer);
        this.temporaryBuffer = [];
    }
}
// https://infra.spec.whatwg.org/#leading-surrogate
function isLeadingSurrogate(ch) {
    return 0xd800 <= ch && ch <= 0xdbff;
}
// https://infra.spec.whatwg.org/#trailing-surrogate
function isTrailingSurrogate(ch) {
    return 0xdc00 <= ch && ch <= 0xdfff;
}
// https://infra.spec.whatwg.org/#surrogate
function isSurrogate(ch) {
    return isLeadingSurrogate(ch) || isTrailingSurrogate(ch);
}
const characterReferenceCodePointReplacements = new Map([
    [0x80, 0x20ac], // EURO SIGN (€)
    [0x82, 0x201a], // SINGLE LOW-9 QUOTATION MARK (‚)
    [0x83, 0x0192], // LATIN SMALL LETTER F WITH HOOK (ƒ)
    [0x84, 0x201e], // DOUBLE LOW-9 QUOTATION MARK („)
    [0x85, 0x2026], // HORIZONTAL ELLIPSIS (…)
    [0x86, 0x2020], // DAGGER (†)
    [0x87, 0x2021], // DOUBLE DAGGER (‡)
    [0x88, 0x02c6], // MODIFIER LETTER CIRCUMFLEX ACCENT (ˆ)
    [0x89, 0x2030], // PER MILLE SIGN (‰)
    [0x8a, 0x0160], // LATIN CAPITAL LETTER S WITH CARON (Š)
    [0x8b, 0x2039], // SINGLE LEFT-POINTING ANGLE QUOTATION MARK (‹)
    [0x8c, 0x0152], // LATIN CAPITAL LIGATURE OE (Œ)
    [0x8e, 0x017d], // LATIN CAPITAL LETTER Z WITH CARON (Ž)
    [0x91, 0x2018], // LEFT SINGLE QUOTATION MARK (‘)
    [0x92, 0x2019], // RIGHT SINGLE QUOTATION MARK (’)
    [0x93, 0x201c], // LEFT DOUBLE QUOTATION MARK (“)
    [0x94, 0x201d], // RIGHT DOUBLE QUOTATION MARK (”)
    [0x95, 0x2022], // BULLET (•)
    [0x96, 0x2013], // EN DASH (–)
    [0x97, 0x2014], // EM DASH (—)
    [0x98, 0x02dc], // SMALL TILDE (˜)
    [0x99, 0x2122], // TRADE MARK SIGN (™)
    [0x9a, 0x0161], // LATIN SMALL LETTER S WITH CARON (š)
    [0x9b, 0x203a], // SINGLE RIGHT-POINTING ANGLE QUOTATION MARK (›)
    [0x9c, 0x0153], // LATIN SMALL LIGATURE OE (œ)
    [0x9e, 0x017e], // LATIN SMALL LETTER Z WITH CARON (ž)
    [0x9f, 0x0178], // LATIN CAPITAL LETTER Y WITH DIAERESIS (Ÿ)
]);
