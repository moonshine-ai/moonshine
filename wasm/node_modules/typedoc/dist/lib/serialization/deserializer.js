import { ok } from "assert";
import { ArrayType, ConditionalType, DeclarationReflection, DocumentReflection, IndexedAccessType, InferredType, IntersectionType, IntrinsicType, LiteralType, MappedType, NamedTupleMember, OptionalType, ParameterReflection, PredicateType, ProjectReflection, QueryType, ReferenceReflection, ReferenceType, Reflection, ReflectionKind, ReflectionType, RestType, SignatureReflection, TemplateLiteralType, TupleType, TypeOperatorType, TypeParameterReflection, UnionType, UnknownType, } from "../models/index.js";
import { insertPrioritySorted } from "../utils/array.js";
/**
 * Deserializes TypeDoc's JSON output back to {@link Reflection} instances.
 *
 * @group Common
 * @summary Deserializes TypeDoc's JSON output
 */
export class Deserializer {
    application;
    deferred = [];
    deserializers = [];
    activeReflection = [];
    constructor(application) {
        this.application = application;
    }
    get logger() {
        return this.application.logger;
    }
    reflectionBuilders = {
        declaration(parent, obj) {
            return new DeclarationReflection(obj.name, obj.kind, parent);
        },
        document(parent, obj) {
            return new DocumentReflection(obj.name, parent, [], {});
        },
        param(parent, obj) {
            return new ParameterReflection(obj.name, obj.kind, parent);
        },
        project() {
            throw new Error("Not supported, use Deserializer.reviveProject(s) instead.");
        },
        reference(parent, obj) {
            // Ugly, but we don't have a reference yet!
            return new ReferenceReflection(obj.name, 
            /* target */ parent, parent);
        },
        signature(parent, obj) {
            return new SignatureReflection(obj.name, obj.kind, parent);
        },
        typeParam(parent, obj) {
            return new TypeParameterReflection(obj.name, parent, void 0);
        },
    };
    typeBuilders = {
        array(obj, de) {
            return new ArrayType(de.reviveType(obj.elementType));
        },
        conditional(obj, de) {
            return new ConditionalType(de.reviveType(obj.checkType), de.reviveType(obj.extendsType), de.reviveType(obj.trueType), de.reviveType(obj.falseType));
        },
        indexedAccess(obj, de) {
            return new IndexedAccessType(de.reviveType(obj.objectType), de.reviveType(obj.indexType));
        },
        inferred(obj, de) {
            return new InferredType(obj.name, de.reviveType(obj.constraint));
        },
        intersection(obj, de) {
            return new IntersectionType(obj.types.map((t) => de.reviveType(t)));
        },
        intrinsic(obj) {
            return new IntrinsicType(obj.name);
        },
        literal(obj) {
            if (obj.value && typeof obj.value === "object") {
                return new LiteralType(BigInt(`${obj.value.negative ? "-" : ""}${obj.value.value}`));
            }
            return new LiteralType(obj.value);
        },
        mapped(obj, de) {
            return new MappedType(obj.parameter, de.reviveType(obj.parameterType), de.reviveType(obj.templateType), obj.readonlyModifier, obj.optionalModifier, de.reviveType(obj.nameType));
        },
        optional(obj, de) {
            return new OptionalType(de.reviveType(obj.elementType));
        },
        predicate(obj, de) {
            return new PredicateType(obj.name, obj.asserts, de.reviveType(obj.targetType));
        },
        query(obj, de) {
            return new QueryType(de.reviveType(obj.queryType));
        },
        reference(obj) {
            // Correct reference will be restored in fromObject
            return ReferenceType.createResolvedReference(obj.name, -2, null);
        },
        reflection(obj, de) {
            return new ReflectionType(de.revive(obj.declaration, (o) => de.constructReflection(o)));
        },
        rest(obj, de) {
            return new RestType(de.reviveType(obj.elementType));
        },
        templateLiteral(obj, de) {
            return new TemplateLiteralType(obj.head, obj.tail.map(([t, s]) => [de.reviveType(t), s]));
        },
        tuple(obj, de) {
            return new TupleType(obj.elements?.map((t) => de.reviveType(t)) || []);
        },
        namedTupleMember(obj, de) {
            return new NamedTupleMember(obj.name, obj.isOptional, de.reviveType(obj.element));
        },
        typeOperator(obj, de) {
            return new TypeOperatorType(de.reviveType(obj.target), obj.operator);
        },
        union(obj, de) {
            return new UnionType(obj.types.map((t) => de.reviveType(t)));
        },
        unknown(obj) {
            return new UnknownType(obj.name);
        },
    };
    /**
     * Only set when deserializing.
     */
    projectRoot;
    oldIdToNewId = {};
    oldFileIdToNewFileId = {};
    project;
    addDeserializer(de) {
        insertPrioritySorted(this.deserializers, de);
    }
    /**
     * Revive a single project into the structure it was originally created with.
     * This is generally not appropriate for merging multiple projects since projects may
     * contain reflections in their root, not inside a module.
     */
    reviveProject(name, projectObj, options) {
        ok(this.deferred.length === 0, "Deserializer.defer was called when not deserializing");
        const project = new ProjectReflection(name || projectObj.name, options.registry);
        if (options.addProjectDocuments) {
            this.application.converter.addProjectDocuments(project);
        }
        this.project = project;
        this.projectRoot = options.projectRoot;
        this.oldIdToNewId = { [projectObj.id]: project.id };
        this.oldFileIdToNewFileId = {};
        this.fromObject(project, projectObj);
        const deferred = this.deferred;
        this.deferred = [];
        for (const def of deferred) {
            def(project);
        }
        ok(this.deferred.length === 0, "Work may not be double deferred when deserializing.");
        ok(this.activeReflection.length === 0, "Imbalanced reflection deserialization");
        this.project = undefined;
        this.projectRoot = undefined;
        this.oldIdToNewId = {};
        this.oldFileIdToNewFileId = {};
        return project;
    }
    reviveProjects(name, projects, options) {
        if (projects.length === 1 &&
            !this.application.options.getValue("alwaysCreateEntryPointModule")) {
            return this.reviveProject(name, projects[0], options);
        }
        const project = new ProjectReflection(name, options.registry);
        if (options.addProjectDocuments) {
            this.application.converter.addProjectDocuments(project);
        }
        this.project = project;
        this.projectRoot = options.projectRoot;
        for (const proj of projects) {
            ok(this.deferred.length === 0, "Deserializer.defer was called when not deserializing");
            const projModule = new DeclarationReflection(proj.name, ReflectionKind.Module, project);
            project.registerReflection(projModule, undefined, undefined);
            project.addChild(projModule);
            this.oldIdToNewId = { [proj.id]: projModule.id };
            this.oldFileIdToNewFileId = {};
            this.fromObject(projModule, proj);
            const deferred = this.deferred;
            this.deferred = [];
            for (const def of deferred) {
                def(project);
            }
            ok(this.deferred.length === 0, "Work may not be double deferred when deserializing.");
            ok(this.activeReflection.length === 0, "Imbalanced reflection deserialization");
        }
        this.oldIdToNewId = {};
        this.oldFileIdToNewFileId = {};
        this.project = undefined;
        this.projectRoot = undefined;
        return project;
    }
    revive(source, creator) {
        if (source) {
            const revived = creator(source);
            this.fromObject(revived, source);
            return revived;
        }
    }
    reviveMany(sourceArray, creator) {
        if (sourceArray) {
            return sourceArray.map((item) => {
                const revived = creator(item);
                this.fromObject(revived, item);
                return revived;
            });
        }
    }
    reviveType(obj) {
        return this.revive(obj, (o) => this.constructType(o));
    }
    constructReflection(obj) {
        ok(this.activeReflection.length > 0);
        const result = this.reflectionBuilders[obj.variant](this.activeReflection[this.activeReflection.length - 1], obj);
        this.oldIdToNewId[obj.id] = result.id;
        this.project.registerReflection(result, undefined, undefined);
        return result;
    }
    constructType(obj) {
        const result = this.typeBuilders[obj.type](obj, this);
        return result;
    }
    fromObject(receiver, obj) {
        if (receiver instanceof Reflection) {
            this.activeReflection.push(receiver);
        }
        receiver.fromObject(this, obj);
        for (const de of this.deserializers) {
            if (de.supports(receiver, obj)) {
                de.fromObject(receiver, obj);
            }
        }
        if (receiver instanceof Reflection) {
            this.activeReflection.pop();
        }
    }
    /**
     * Defers work until the initial pass of serialization has been completed.
     * This can be used to set up references which cannot be immediately restored.
     *
     * May only be called when deserializing.
     */
    defer(cb) {
        this.deferred.push(cb);
    }
}
