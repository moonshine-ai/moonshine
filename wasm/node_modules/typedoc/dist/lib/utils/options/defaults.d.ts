/**
 * Defaults values for TypeDoc options
 * @module
 */
import type { BundledLanguage } from "@gerrit0/mini-shiki";
import type { EnumKeys } from "../enum.js";
import type { ReflectionKind } from "../../models/index.js";
export declare const excludeNotDocumentedKinds: readonly EnumKeys<typeof ReflectionKind>[];
export declare const excludeTags: readonly `@${string}`[];
export declare const blockTags: readonly `@${string}`[];
export declare const inlineTags: readonly `@${string}`[];
export declare const modifierTags: readonly `@${string}`[];
export declare const cascadedModifierTags: readonly `@${string}`[];
export declare const notRenderedTags: readonly `@${string}`[];
export declare const highlightLanguages: readonly BundledLanguage[];
export declare const ignoredHighlightLanguages: readonly string[];
export declare const sort: readonly string[];
export declare const kindSortOrder: readonly EnumKeys<typeof ReflectionKind>[];
export declare const requiredToBeDocumented: readonly EnumKeys<typeof ReflectionKind>[];
