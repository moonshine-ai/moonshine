import { ReflectionKind } from "../models/index.js";
export function validateMergeModuleWith(project, logger) {
    for (const refl of project.getReflectionsByKind(ReflectionKind.SomeModule)) {
        if (refl.comment?.getTag("@mergeModuleWith")) {
            logger.warn(logger.i18n.reflection_0_has_unused_mergeModuleWith_tag(refl.getFriendlyFullName()));
        }
    }
    if (project.comment?.getTag("@mergeModuleWith")) {
        logger.warn(logger.i18n.reflection_0_has_unused_mergeModuleWith_tag("<project>"));
    }
}
