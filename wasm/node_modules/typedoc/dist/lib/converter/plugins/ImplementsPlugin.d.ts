import { ConverterComponent } from "../components.js";
import type { Converter } from "../converter.js";
/**
 * A plugin that detects interface implementations of functions and
 * properties on classes and links them.
 */
export declare class ImplementsPlugin extends ConverterComponent {
    private resolved;
    private postponed;
    private revivingSerialized;
    constructor(owner: Converter);
    /**
     * Mark all members of the given class to be the implementation of the matching interface member.
     */
    private analyzeImplements;
    private analyzeInheritance;
    private onResolveEnd;
    private onRevive;
    private resolve;
    private tryResolve;
    private doResolve;
    private getExtensionInfo;
    private onSignature;
    /**
     * Responsible for setting the {@link DeclarationReflection.inheritedFrom},
     * {@link DeclarationReflection.overwrites}, and {@link DeclarationReflection.implementationOf}
     * properties on the provided reflection temporarily, these links will be replaced
     * during the resolve step with links which actually point to the right place.
     */
    private onDeclaration;
    /**
     * Responsible for copying comments from "parent" reflections defined
     * in either a base class or implemented interface to the child class.
     */
    private handleInheritedComments;
    /**
     * Copy the comment of the source reflection to the target reflection with a JSDoc style copy
     * function. The TSDoc copy function is in the InheritDocPlugin.
     */
    private copyComment;
}
