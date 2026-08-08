import { ok } from "assert";
import { ApplicationEvents } from "../../application-events.js";
import { Comment, ReflectionKind, } from "../../models/index.js";
import { ConverterComponent } from "../components.js";
import { ConverterEvents } from "../converter-events.js";
/**
 * Handles `@mergeModuleWith` tags in comments
 * Warnings if resolution failed are emitted during the validation step, not here.
 */
export class MergeModuleWithPlugin extends ConverterComponent {
    constructor(owner) {
        super(owner);
        // Note: This happens before grouping/categorizing
        this.owner.on(ConverterEvents.RESOLVE_BEGIN, this.onResolveBegin.bind(this), 10000);
        this.application.on(ApplicationEvents.REVIVE, this.onRevive.bind(this), 10000);
    }
    onResolveBegin(context) {
        this.onRevive(context.project);
    }
    onRevive(project) {
        const mods = project.getReflectionsByKind(ReflectionKind.SomeModule);
        for (const refl of mods) {
            this.checkAndMerge(refl);
        }
    }
    checkAndMerge(refl) {
        ok(refl.isDeclaration());
        const tag = refl.comment?.getTag("@mergeModuleWith");
        if (!tag)
            return;
        const project = refl.project;
        const targetStr = Comment.combineDisplayParts(tag.content);
        const target = targetStr === "<project>"
            ? project
            : project.getChildByName(targetStr);
        if (!target?.isDeclaration() && !target?.isProject()) {
            return;
        }
        let tempRefl = refl;
        while (tempRefl !== project) {
            if (tempRefl === target) {
                this.application.logger.warn(this.application.i18n.reflection_0_tried_to_merge_into_child_1(refl.getFriendlyFullName(), target.getFriendlyFullName()));
            }
            tempRefl = tempRefl.parent;
        }
        this.application.logger.verbose(`Merging ${refl.getFullName()} into ${target.getFullName()}`);
        project.mergeModules(refl, target);
    }
}
