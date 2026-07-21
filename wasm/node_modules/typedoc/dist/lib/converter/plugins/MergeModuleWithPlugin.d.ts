import { ConverterComponent } from "../components.js";
import type { Converter } from "../converter.js";
/**
 * Handles `@mergeModuleWith` tags in comments
 * Warnings if resolution failed are emitted during the validation step, not here.
 */
export declare class MergeModuleWithPlugin extends ConverterComponent {
    constructor(owner: Converter);
    private onResolveBegin;
    private onRevive;
    private checkAndMerge;
}
