import { ContainerReflection, type DeclarationReflection, type DocumentReflection } from "../../models/reflections/index.js";
import { ReflectionGroup } from "../../models/ReflectionGroup.js";
import { ConverterComponent } from "../components.js";
import type { Converter } from "../converter.js";
import type { Internationalization } from "../../internationalization/index.js";
/**
 * A handler that sorts and groups the found reflections in the resolving phase.
 *
 * The handler sets the `groups` property of all container reflections.
 */
export declare class GroupPlugin extends ConverterComponent {
    sortFunction: (reflections: Array<DeclarationReflection | DocumentReflection>) => void;
    accessor groupOrder: string[];
    accessor sortEntryPoints: boolean;
    accessor groupReferencesByType: boolean;
    static WEIGHTS: string[];
    constructor(owner: Converter);
    /**
     * Triggered when the converter has finished resolving a project.
     *
     * @param context  The context object describing the current state the converter is in.
     */
    private onEndResolve;
    private onRevive;
    private setup;
    private group;
    /**
     * Extracts the groups for a given reflection.
     *
     * @privateRemarks
     * If you change this, also update extractCategories in CategoryPlugin accordingly.
     */
    getGroups(reflection: DeclarationReflection | DocumentReflection): Set<string>;
    static getGroups(reflection: DeclarationReflection | DocumentReflection, groupReferencesByType: boolean, internationalization: Internationalization): Set<string>;
    /**
     * Create a grouped representation of the given list of reflections.
     *
     * Reflections are grouped by kind and sorted by weight and name.
     *
     * @param reflections  The reflections that should be grouped.
     * @returns An array containing all children of the given reflection grouped by their kind.
     */
    getReflectionGroups(parent: ContainerReflection, reflections: Array<DeclarationReflection | DocumentReflection>): ReflectionGroup[];
    /**
     * Callback used to sort groups by name.
     */
    static sortGroupCallback(a: ReflectionGroup, b: ReflectionGroup): number;
}
