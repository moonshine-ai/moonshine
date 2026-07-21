import type { ReflectionSymbolId } from "../../models/index.js";
export declare enum TokenSyntaxKind {
    Text = "text",
    NewLine = "new_line",
    OpenBrace = "open_brace",
    CloseBrace = "close_brace",
    Tag = "tag",
    Code = "code",
    TypeAnnotation = "type"
}
export interface Token {
    kind: TokenSyntaxKind;
    text: string;
    pos: number;
    tsLinkTarget?: ReflectionSymbolId;
    tsLinkText?: string;
}
