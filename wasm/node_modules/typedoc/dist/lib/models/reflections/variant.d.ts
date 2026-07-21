import type { DeclarationReflection } from "./declaration.js";
import type { DocumentReflection } from "./document.js";
import type { ParameterReflection } from "./parameter.js";
import type { ProjectReflection } from "./project.js";
import type { ReferenceReflection } from "./reference.js";
import type { SignatureReflection } from "./signature.js";
import type { TypeParameterReflection } from "./type-parameter.js";
/**
 * A map of known {@link Reflection} concrete subclasses.
 * This is used during deserialization to reconstruct serialized objects.
 */
export interface ReflectionVariant {
    declaration: DeclarationReflection;
    param: ParameterReflection;
    project: ProjectReflection;
    reference: ReferenceReflection;
    signature: SignatureReflection;
    typeParam: TypeParameterReflection;
    document: DocumentReflection;
}
export type SomeReflection = ReflectionVariant[keyof ReflectionVariant];
