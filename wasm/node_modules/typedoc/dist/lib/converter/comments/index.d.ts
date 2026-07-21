import ts from "typescript";
import { Comment, ReflectionKind } from "../../models/index.js";
import { type Logger } from "../../utils/index.js";
import type { CommentStyle, JsDocCompatibility } from "../../utils/options/declaration.js";
import type { FileRegistry } from "../../models/FileRegistry.js";
export interface CommentParserConfig {
    blockTags: Set<string>;
    inlineTags: Set<string>;
    modifierTags: Set<string>;
    jsDocCompatibility: JsDocCompatibility;
    suppressCommentWarningsInDeclarationFiles: boolean;
    useTsLinkResolution: boolean;
    commentStyle: CommentStyle;
}
export declare function clearCommentCache(): void;
export declare function getComment(symbol: ts.Symbol, kind: ReflectionKind, config: CommentParserConfig, logger: Logger, checker: ts.TypeChecker, files: FileRegistry): Comment | undefined;
export declare function getNodeComment(node: ts.Node, moduleComment: boolean, config: CommentParserConfig, logger: Logger, checker: ts.TypeChecker | undefined, files: FileRegistry): Comment | undefined;
export declare function getFileComment(file: ts.SourceFile, config: CommentParserConfig, logger: Logger, checker: ts.TypeChecker | undefined, files: FileRegistry): Comment | undefined;
export declare function getSignatureComment(declaration: ts.SignatureDeclaration | ts.JSDocSignature, config: CommentParserConfig, logger: Logger, checker: ts.TypeChecker, files: FileRegistry): Comment | undefined;
export declare function getJsDocComment(declaration: ts.JSDocPropertyLikeTag | ts.JSDocCallbackTag | ts.JSDocTypedefTag | ts.JSDocTemplateTag | ts.JSDocEnumTag, config: CommentParserConfig, logger: Logger, checker: ts.TypeChecker | undefined, files: FileRegistry): Comment | undefined;
