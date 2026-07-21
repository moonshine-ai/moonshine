import { type DeclarationReflection, type DocumentReflection } from "../../models/index.js";
import { ConverterComponent } from "../components.js";
import type { Converter } from "../converter.js";
/**
 * A handler that sorts and categorizes the found reflections in the resolving phase.
 *
 * The handler sets the ´category´ property of all reflections.
 */
export declare class CategoryPlugin extends ConverterComponent {
    sortFunction: (reflections: Array<DeclarationReflection | DocumentReflection>) => void;
    accessor defaultCategory: string;
    accessor categoryOrder: string[];
    accessor categorizeByGroup: boolean;
    static defaultCategory: string;
    static WEIGHTS: string[];
    constructor(owner: Converter);
    private onRevive;
    /**
     * Triggered when the converter begins converting a project.
     */
    private setup;
    /**
     * Triggered when the converter has finished resolving a project.
     *
     * @param context  The context object describing the current state the converter is in.
     */
    private onEndResolve;
    private categorize;
    private groupCategorize;
    private lumpCategorize;
    /**
     * Create a categorized representation of the given list of reflections.
     *
     * @param reflections  The reflections that should be categorized.
     * @returns An array containing all children of the given reflection categorized
     */
    private getReflectionCategories;
    /**
     * Callback used to sort categories by name.
     *
     * @param a The left reflection to sort.
     * @param b The right reflection to sort.
     * @returns The sorting weight.
     */
    private static sortCatCallback;
    static getCategories(reflection: DeclarationReflection | DocumentReflection): Set<string>;
}
