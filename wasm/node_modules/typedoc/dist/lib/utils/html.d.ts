export declare function escapeHtml(html: string): string;
/**
 * Replaces non-[URL code points](https://url.spec.whatwg.org/#url-code-points)
 * with an underscore. Also disallows some additional special characters which either
 * result in confusing file names or invalid file names on some platform.
 *
 * Ref: #2714
 */
export declare function createNormalizedUrl(url: string): string;
export declare const enum ParserState {
    BeforeAttributeName = 0,
    AfterAttributeName = 1,
    BeforeAttributeValue = 2,
    END = 3
}
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
export declare class HtmlAttributeParser {
    readonly text: string;
    pos: number;
    state: ParserState;
    currentAttributeName: string;
    currentAttributeValueStart: number;
    currentAttributeValueEnd: number;
    currentAttributeValue: string;
    private temporaryBuffer;
    private characterReferenceCode;
    constructor(text: string, pos?: number);
    step(): void;
    private peek;
    private consume;
    beforeAttributeName(): void;
    attributeName(): void;
    afterAttributeName(): void;
    beforeAttributeValue(): void;
    attributeValueDoubleQuoted(): void;
    attributeValueSingleQuoted(): void;
    attributeValueUnquoted(): void;
    afterAttributeValueQuoted(): void;
    characterReference(): void;
    namedCharacterReference(): void;
    ambiguousAmpersand(): void;
    numericCharacterReference(): void;
    hexadecimalCharacterReferenceStart(): void;
    decimalCharacterReferenceStart(): void;
    hexadecimalCharacterReference(): void;
    decimalCharacterReference(): void;
    numericCharacterReferenceEndState(): void;
    private flushTemporaryBuffer;
}
