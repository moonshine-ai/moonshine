import ts from "typescript";
import { ApplicationEvents } from "../../application-events.js";
import { DeclarationReflection, ReflectionFlag, ReflectionKind, SignatureReflection, } from "../../models/reflections/index.js";
import { ReferenceType, ReflectionType, } from "../../models/types.js";
import { filterMap, zip } from "../../utils/array.js";
import { ConverterComponent } from "../components.js";
import { getHumanName } from "../../utils/index.js";
import { ConverterEvents } from "../converter-events.js";
/**
 * A plugin that detects interface implementations of functions and
 * properties on classes and links them.
 */
export class ImplementsPlugin extends ConverterComponent {
    resolved = new WeakSet();
    postponed = new WeakMap();
    revivingSerialized = false;
    constructor(owner) {
        super(owner);
        this.owner.on(ConverterEvents.RESOLVE_END, this.onResolveEnd.bind(this));
        this.owner.on(ConverterEvents.CREATE_DECLARATION, this.onDeclaration.bind(this), -1000);
        this.owner.on(ConverterEvents.CREATE_SIGNATURE, this.onSignature.bind(this), 1000);
        this.application.on(ApplicationEvents.REVIVE, this.onRevive.bind(this));
    }
    /**
     * Mark all members of the given class to be the implementation of the matching interface member.
     */
    analyzeImplements(project, classReflection, interfaceReflection) {
        this.handleInheritedComments(classReflection, interfaceReflection);
        if (!interfaceReflection.children) {
            return;
        }
        interfaceReflection.children.forEach((interfaceMember) => {
            const classMember = findMatchingMember(interfaceMember, classReflection);
            if (!classMember) {
                return;
            }
            const interfaceMemberName = interfaceReflection.name + "." + interfaceMember.name;
            classMember.implementationOf =
                ReferenceType.createResolvedReference(interfaceMemberName, interfaceMember, project);
            const intSigs = interfaceMember.signatures ||
                interfaceMember.type?.visit({
                    reflection: (r) => r.declaration.signatures,
                });
            const clsSigs = classMember.signatures ||
                classMember.type?.visit({
                    reflection: (r) => r.declaration.signatures,
                });
            if (intSigs && clsSigs) {
                for (const [clsSig, intSig] of zip(clsSigs, intSigs)) {
                    if (clsSig.implementationOf) {
                        const target = intSig.parent.kindOf(ReflectionKind.FunctionOrMethod)
                            ? intSig
                            : intSig.parent.parent;
                        clsSig.implementationOf =
                            ReferenceType.createResolvedReference(clsSig.implementationOf.name, target, project);
                    }
                }
            }
            this.handleInheritedComments(classMember, interfaceMember);
        });
    }
    analyzeInheritance(project, reflection) {
        const extendedTypes = filterMap(reflection.extendedTypes ?? [], (type) => {
            return type instanceof ReferenceType &&
                type.reflection instanceof DeclarationReflection
                ? type
                : void 0;
        });
        for (const parent of extendedTypes) {
            this.handleInheritedComments(reflection, parent.reflection);
            for (const parentMember of parent.reflection.children ?? []) {
                const child = findMatchingMember(parentMember, reflection);
                if (child) {
                    const key = child.overwrites
                        ? "overwrites"
                        : "inheritedFrom";
                    for (const [childSig, parentSig] of zip(child.signatures ?? [], parentMember.signatures ?? [])) {
                        childSig[key] = ReferenceType.createResolvedReference(`${parent.name}.${parentMember.name}`, parentSig, project);
                    }
                    child[key] = ReferenceType.createResolvedReference(`${parent.name}.${parentMember.name}`, parentMember, project);
                    this.handleInheritedComments(child, parentMember);
                }
            }
        }
    }
    onResolveEnd(context) {
        this.resolve(context.project);
    }
    onRevive(project) {
        this.revivingSerialized = true;
        this.resolve(project);
        this.revivingSerialized = false;
    }
    resolve(project) {
        for (const id in project.reflections) {
            const refl = project.reflections[id];
            if (refl instanceof DeclarationReflection) {
                this.tryResolve(project, refl);
            }
        }
    }
    tryResolve(project, reflection) {
        const requirements = filterMap([
            ...(reflection.implementedTypes ?? []),
            ...(reflection.extendedTypes ?? []),
        ], (type) => {
            return type instanceof ReferenceType ? type.reflection : void 0;
        });
        if (requirements.every((req) => this.resolved.has(req))) {
            this.doResolve(project, reflection);
            this.resolved.add(reflection);
            for (const refl of this.postponed.get(reflection) ?? []) {
                this.tryResolve(project, refl);
            }
            this.postponed.delete(reflection);
        }
        else {
            for (const req of requirements) {
                const future = this.postponed.get(req) ?? new Set();
                future.add(reflection);
                this.postponed.set(req, future);
            }
        }
    }
    doResolve(project, reflection) {
        if (reflection.kindOf(ReflectionKind.Class) &&
            reflection.implementedTypes) {
            reflection.implementedTypes.forEach((type) => {
                if (!(type instanceof ReferenceType)) {
                    return;
                }
                if (type.reflection &&
                    type.reflection.kindOf(ReflectionKind.ClassOrInterface)) {
                    this.analyzeImplements(project, reflection, type.reflection);
                }
            });
        }
        if (reflection.kindOf(ReflectionKind.ClassOrInterface) &&
            reflection.extendedTypes) {
            this.analyzeInheritance(project, reflection);
        }
    }
    getExtensionInfo(context, reflection) {
        if (!reflection || !reflection.kindOf(ReflectionKind.Inheritable)) {
            return;
        }
        // Need this because we re-use reflections for type literals.
        if (!reflection.parent?.kindOf(ReflectionKind.ClassOrInterface)) {
            return;
        }
        const symbol = context.project.getSymbolFromReflection(reflection.parent);
        if (!symbol) {
            return;
        }
        const declaration = symbol
            .getDeclarations()
            ?.find((n) => ts.isClassDeclaration(n) || ts.isInterfaceDeclaration(n));
        if (!declaration) {
            return;
        }
        return { symbol, declaration };
    }
    onSignature(context, reflection) {
        this.onDeclaration(context, reflection.parent);
    }
    /**
     * Responsible for setting the {@link DeclarationReflection.inheritedFrom},
     * {@link DeclarationReflection.overwrites}, and {@link DeclarationReflection.implementationOf}
     * properties on the provided reflection temporarily, these links will be replaced
     * during the resolve step with links which actually point to the right place.
     */
    onDeclaration(context, reflection) {
        const info = this.getExtensionInfo(context, reflection);
        if (!info) {
            return;
        }
        if (reflection.kind === ReflectionKind.Constructor) {
            const ctor = info.declaration.members.find(ts.isConstructorDeclaration);
            constructorInheritance(context, reflection, info.declaration, ctor);
            return;
        }
        const childType = reflection.flags.isStatic
            ? context.checker.getTypeOfSymbolAtLocation(info.symbol, info.declaration)
            : context.checker.getDeclaredTypeOfSymbol(info.symbol);
        const property = findProperty(reflection, childType);
        if (!property) {
            // We're probably broken... but I don't think this should be fatal.
            context.logger.warn(`Failed to retrieve${reflection.flags.isStatic ? " static" : ""} member "${reflection.escapedName ?? reflection.name}" of "${reflection.parent?.name}" for inheritance analysis. Please report a bug.`);
            return;
        }
        // Need to check both extends and implements clauses.
        out: for (const clause of info.declaration.heritageClauses ?? []) {
            // No point checking implemented types for static members, they won't exist.
            if (reflection.flags.isStatic &&
                clause.token === ts.SyntaxKind.ImplementsKeyword) {
                continue;
            }
            for (const expr of clause.types) {
                const parentType = context.checker.getTypeAtLocation(reflection.flags.isStatic ? expr.expression : expr);
                const parentProperty = findProperty(reflection, parentType);
                if (parentProperty) {
                    const isInherit = property
                        .getDeclarations()
                        ?.some((d) => d.parent !== info.declaration) ??
                        true;
                    createLink(context, reflection, clause, expr, parentProperty, isInherit);
                    // Can't always break because we need to also set `implementationOf` if we
                    // inherit from a base class and also implement an interface.
                    if (clause.token === ts.SyntaxKind.ImplementsKeyword) {
                        break out;
                    }
                }
            }
        }
    }
    /**
     * Responsible for copying comments from "parent" reflections defined
     * in either a base class or implemented interface to the child class.
     */
    handleInheritedComments(child, parent) {
        this.copyComment(child, parent);
        if (parent.kindOf(ReflectionKind.Property) &&
            child.kindOf(ReflectionKind.Accessor)) {
            if (child.getSignature) {
                this.copyComment(child.getSignature, parent);
                child.getSignature.implementationOf = child.implementationOf;
            }
            if (child.setSignature) {
                this.copyComment(child.setSignature, parent);
                child.setSignature.implementationOf = child.implementationOf;
            }
        }
        if (parent.kindOf(ReflectionKind.Accessor) &&
            child.kindOf(ReflectionKind.Accessor)) {
            if (parent.getSignature && child.getSignature) {
                this.copyComment(child.getSignature, parent.getSignature);
            }
            if (parent.setSignature && child.setSignature) {
                this.copyComment(child.setSignature, parent.setSignature);
            }
        }
        if (parent.kindOf(ReflectionKind.FunctionOrMethod) &&
            parent.signatures &&
            child.signatures) {
            for (const [cs, ps] of zip(child.signatures, parent.signatures)) {
                this.copyComment(cs, ps);
            }
        }
        else if (parent.kindOf(ReflectionKind.Property) &&
            parent.type instanceof ReflectionType &&
            parent.type.declaration.signatures &&
            child.signatures) {
            for (const [cs, ps] of zip(child.signatures, parent.type.declaration.signatures)) {
                this.copyComment(cs, ps);
            }
        }
    }
    /**
     * Copy the comment of the source reflection to the target reflection with a JSDoc style copy
     * function. The TSDoc copy function is in the InheritDocPlugin.
     */
    copyComment(target, source) {
        if (!shouldCopyComment(target, source, this.revivingSerialized)) {
            return;
        }
        target.comment = source.comment.clone();
        if (target instanceof DeclarationReflection &&
            source instanceof DeclarationReflection) {
            for (const [tt, ts] of zip(target.typeParameters || [], source.typeParameters || [])) {
                this.copyComment(tt, ts);
            }
        }
        if (target instanceof SignatureReflection &&
            source instanceof SignatureReflection) {
            for (const [tt, ts] of zip(target.typeParameters || [], source.typeParameters || [])) {
                this.copyComment(tt, ts);
            }
            for (const [pt, ps] of zip(target.parameters || [], source.parameters || [])) {
                this.copyComment(pt, ps);
            }
        }
    }
}
function constructorInheritance(context, reflection, childDecl, constructorDecl) {
    const extendsClause = childDecl.heritageClauses?.find((cl) => cl.token === ts.SyntaxKind.ExtendsKeyword);
    if (!extendsClause)
        return;
    const name = `${extendsClause.types[0].getText()}.constructor`;
    const key = constructorDecl ? "overwrites" : "inheritedFrom";
    reflection[key] ??= ReferenceType.createBrokenReference(name, context.project);
    for (const sig of reflection.signatures ?? []) {
        sig[key] ??= ReferenceType.createBrokenReference(name, context.project);
    }
}
function findProperty(reflection, parent) {
    return parent.getProperties().find((prop) => {
        return reflection.escapedName
            ? prop.escapedName === reflection.escapedName
            : prop.name === reflection.name;
    });
}
function createLink(context, reflection, clause, expr, symbol, isInherit) {
    const project = context.project;
    const name = `${expr.expression.getText()}.${getHumanName(symbol.name)}`;
    link(reflection);
    link(reflection.getSignature);
    link(reflection.setSignature);
    for (const sig of reflection.indexSignatures || []) {
        link(sig);
    }
    for (const sig of reflection.signatures ?? []) {
        link(sig);
    }
    // Intentionally create broken links here. These will be replaced with real links during
    // resolution if we can do so. We create broken links rather than real links because in the
    // case of an inherited symbol, we'll end up referencing a single symbol ID rather than one
    // for each class.
    function link(target) {
        if (!target)
            return;
        if (clause.token === ts.SyntaxKind.ImplementsKeyword) {
            target.implementationOf ??= ReferenceType.createBrokenReference(name, project);
            return;
        }
        if (isInherit) {
            target.setFlag(ReflectionFlag.Inherited);
            target.inheritedFrom ??= ReferenceType.createBrokenReference(name, project);
        }
        else {
            target.overwrites ??= ReferenceType.createBrokenReference(name, project);
        }
    }
}
function shouldCopyComment(target, source, revivingSerialized) {
    if (!source.comment) {
        return false;
    }
    if (target.comment) {
        // If we're reviving, then the revived project might have a better comment
        // on source, so copy it.
        if (revivingSerialized && source.comment.similarTo(target.comment)) {
            return true;
        }
        // We might still want to copy, if the child has a JSDoc style inheritDoc tag.
        const tag = target.comment.getTag("@inheritDoc");
        if (!tag || tag.name) {
            return false;
        }
    }
    return true;
}
function findMatchingMember(toMatch, container) {
    return container.children?.find((child) => child.name == toMatch.name &&
        child.flags.isStatic === toMatch.flags.isStatic);
}
