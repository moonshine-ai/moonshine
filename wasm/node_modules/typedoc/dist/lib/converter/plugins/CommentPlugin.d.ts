import { ConverterComponent } from "../components.js";
import type { Converter } from "../converter.js";
/**
 * Handles most behavior triggered by comments. `@group` and `@category` are handled by their respective plugins, but everything else is here.
 *
 * How it works today
 * ==================
 * During conversion:
 * - Handle visibility flags (`@private`, `@protected`. `@public`)
 * - Handle module renames (`@module`)
 * - Remove excluded tags & comment discovery tags (`@module`, `@packageDocumentation`)
 * - Copy comments for type parameters from the parent container (for classes/interfaces)
 *
 * Resolve begin:
 * - Remove hidden reflections
 *
 * Resolve:
 * - Apply `@label` tag
 * - Copy comments on signature containers to the signature if signatures don't already have a comment
 *   and then remove the comment on the container.
 * - Copy comments to parameters and type parameters (for signatures)
 * - Apply `@group` and `@category` tags
 *
 * Resolve end:
 * - Copy auto inherited comments from heritage clauses
 * - Handle `@inheritDoc`
 * - Resolve `@link` tags to point to target reflections
 *
 * How it should work
 * ==================
 * During conversion:
 * - Handle visibility flags (`@private`, `@protected`. `@public`)
 * - Handle module renames (`@module`)
 * - Remove excluded tags & comment discovery tags (`@module`, `@packageDocumentation`)
 *
 * Resolve begin (100):
 * - Copy auto inherited comments from heritage clauses
 * - Apply `@label` tag
 *
 * Resolve begin (75)
 * - Handle `@inheritDoc`
 *
 * Resolve begin (50)
 * - Copy comments on signature containers to the signature if signatures don't already have a comment
 *   and then remove the comment on the container.
 * - Copy comments for type parameters from the parent container (for classes/interfaces)
 *
 * Resolve begin (25)
 * - Remove hidden reflections
 *
 * Resolve:
 * - Copy comments to parameters and type parameters (for signatures)
 * - Apply `@group` and `@category` tags
 *
 * Resolve end:
 * - Resolve `@link` tags to point to target reflections
 *
 */
export declare class CommentPlugin extends ConverterComponent {
    accessor excludeTags: `@${string}`[];
    accessor cascadedModifierTags: `@${string}`[];
    accessor excludeInternal: boolean;
    accessor excludePrivate: boolean;
    accessor excludeProtected: boolean;
    accessor excludeNotDocumented: boolean;
    accessor excludeCategories: string[];
    accessor defaultCategory: string;
    private _excludeKinds;
    private get excludeNotDocumentedKinds();
    constructor(owner: Converter);
    /**
     * Apply all comment tag modifiers to the given reflection.
     *
     * @param reflection  The reflection the modifiers should be applied to.
     * @param comment  The comment that should be searched for modifiers.
     */
    private applyModifiers;
    /**
     * Triggered when the converter has created a type parameter reflection.
     *
     * @param context  The context object describing the current state the converter is in.
     * @param reflection  The reflection that is currently processed.
     */
    private onCreateTypeParameter;
    /**
     * Triggered when the converter has created a declaration or signature reflection.
     *
     * Invokes the comment parser.
     *
     * @param context  The context object describing the current state the converter is in.
     * @param reflection  The reflection that is currently processed.
     * @param node  The node that is currently processed if available.
     */
    private onDeclaration;
    /**
     * Triggered when the converter begins resolving a project.
     *
     * @param context  The context object describing the current state the converter is in.
     */
    private onBeginResolve;
    /**
     * Triggered when the converter resolves a reflection.
     *
     * Cleans up comment tags related to signatures like `@param` or `@returns`
     * and moves their data to the corresponding parameter reflections.
     *
     * This hook also copies over the comment of function implementations to their
     * signatures.
     *
     * @param context  The context object describing the current state the converter is in.
     * @param reflection  The reflection that is currently resolved.
     */
    private onResolve;
    private moveSignatureParamComments;
    private removeExcludedTags;
    private cascadeModifiers;
    /**
     * Determines whether or not a reflection has been hidden
     *
     * @param reflection Reflection to check if hidden
     */
    private isHidden;
    private excludedByCategory;
    private validateParamTags;
}
