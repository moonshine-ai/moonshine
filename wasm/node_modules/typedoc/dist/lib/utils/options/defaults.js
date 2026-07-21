import * as TagDefaults from "./tsdoc-defaults.js";
export const excludeNotDocumentedKinds = [
    "Module",
    "Namespace",
    "Enum",
    // Not including enum member here by default
    "Variable",
    "Function",
    "Class",
    "Interface",
    "Constructor",
    "Property",
    "Method",
    "CallSignature",
    "IndexSignature",
    "ConstructorSignature",
    "Accessor",
    "GetSignature",
    "SetSignature",
    "TypeAlias",
    "Reference",
];
export const excludeTags = [
    "@override",
    "@virtual",
    "@privateRemarks",
    "@satisfies",
    "@overload",
    "@inline",
];
export const blockTags = TagDefaults.blockTags;
export const inlineTags = TagDefaults.inlineTags;
export const modifierTags = TagDefaults.modifierTags;
export const cascadedModifierTags = [
    "@alpha",
    "@beta",
    "@experimental",
];
export const notRenderedTags = [
    "@showCategories",
    "@showGroups",
    "@hideCategories",
    "@hideGroups",
    "@expand",
    "@summary",
    "@group",
    "@groupDescription",
    "@category",
    "@categoryDescription",
];
export const highlightLanguages = [
    "bash",
    "console",
    "css",
    "html",
    "javascript",
    "json",
    "jsonc",
    "json5",
    "yaml",
    "tsx",
    "typescript",
];
export const ignoredHighlightLanguages = [];
export const sort = [
    "kind",
    "instance-first",
    "alphabetical-ignoring-documents",
];
export const kindSortOrder = [
    "Document",
    "Project",
    "Module",
    "Namespace",
    "Enum",
    "EnumMember",
    "Class",
    "Interface",
    "TypeAlias",
    "Constructor",
    "Property",
    "Variable",
    "Function",
    "Accessor",
    "Method",
    "Reference",
];
export const requiredToBeDocumented = [
    "Enum",
    "EnumMember",
    "Variable",
    "Function",
    "Class",
    "Interface",
    "Property",
    "Method",
    "Accessor",
    "TypeAlias",
];
