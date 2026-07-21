import { ok } from "assert";
import { nicePath } from "../utils/paths.js";
import { discoverAllReferenceTypes } from "../utils/reflections.js";
function makeIntentionallyExportedHelper(project, intentional, logger) {
    const used = new Set();
    const processed = intentional.map((v) => {
        const index = v.lastIndexOf(":");
        if (index === -1) {
            return ["", v];
        }
        return [v.substring(0, index), v.substring(index + 1)];
    });
    return {
        has(type, typeName) {
            ok(!type.reflection);
            // If it isn't declared anywhere, we can't produce a good error message about where
            // the non-exported symbol is, so even if it isn't ignored, pretend it is. In practice,
            // this will happen incredibly rarely, since symbols without declarations are very rare.
            // I know of only two instances:
            // 1. `undefined` in `globalThis`
            // 2. Properties on non-homomorphic mapped types, e.g. the symbol for "foo" on `Record<"foo", 1>`
            // There might be others, so still check this here rather than asserting, but print a debug log
            // so that we can possibly improve this in the future.
            if (!type.package) {
                logger.verbose(`The type ${type.qualifiedName} has no declarations, implicitly allowing missing export.`);
                return true;
            }
            // Don't produce warnings for third-party symbols.
            if (type.package !== project.packageName) {
                return true;
            }
            for (const [index, [file, name]] of processed.entries()) {
                if (typeName === name &&
                    type.symbolId.fileName.endsWith(file)) {
                    used.add(index);
                    return true;
                }
            }
            return false;
        },
        getUnused() {
            return intentional.filter((_, i) => !used.has(i));
        },
    };
}
export function validateExports(project, logger, intentionallyNotExported) {
    const intentional = makeIntentionallyExportedHelper(project, intentionallyNotExported, logger);
    const warned = new Set();
    for (const { type, owner } of discoverAllReferenceTypes(project, true)) {
        const uniqueId = type.symbolId?.getStableKey();
        if (!type.reflection &&
            !type.externalUrl &&
            !type.isIntentionallyBroken() &&
            !intentional.has(type, type.qualifiedName) &&
            !warned.has(uniqueId) &&
            !project.symbolIdHasBeenRemoved(type.symbolId)) {
            warned.add(uniqueId);
            logger.warn(logger.i18n.type_0_defined_in_1_is_referenced_by_2_but_not_included_in_docs(type.qualifiedName, nicePath(type.symbolId.fileName), owner.getFriendlyFullName()));
        }
    }
    const unusedIntentional = intentional.getUnused();
    if (unusedIntentional.length) {
        logger.warn(logger.i18n.invalid_intentionally_not_exported_symbols_0(unusedIntentional.join("\n\t")));
    }
}
