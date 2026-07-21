import type { BuiltinTranslatableStringConstraints } from "./translatable.js" with { "resolution-mode": "import" };
declare function buildTranslation<T extends BuiltinTranslatableStringConstraints>(translations: T): T;
declare function buildIncompleteTranslation<T extends Partial<BuiltinTranslatableStringConstraints>>(translations: T): T;
declare const _default: {
    buildTranslation: typeof buildTranslation;
    buildIncompleteTranslation: typeof buildIncompleteTranslation;
};
export = _default;
