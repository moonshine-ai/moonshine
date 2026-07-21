import { ReflectionKind, DeclarationReflection, } from "../../models/reflections/index.js";
import { ReferenceType } from "../../models/types.js";
import { ConverterComponent } from "../components.js";
import { ApplicationEvents } from "../../application-events.js";
import { ConverterEvents } from "../converter-events.js";
/**
 * Responsible for adding `implementedBy` / `implementedFrom`
 */
export class TypePlugin extends ConverterComponent {
    reflections = new Set();
    constructor(owner) {
        super(owner);
        this.owner.on(ConverterEvents.RESOLVE, this.onResolve.bind(this));
        this.owner.on(ConverterEvents.RESOLVE_END, this.onResolveEnd.bind(this));
        this.owner.on(ConverterEvents.END, () => this.reflections.clear());
        this.application.on(ApplicationEvents.REVIVE, this.onRevive.bind(this), 100);
    }
    onRevive(project) {
        for (const id in project.reflections) {
            this.resolve(project, project.reflections[id]);
        }
        this.finishResolve(project);
        this.reflections.clear();
    }
    onResolve(context, reflection) {
        this.resolve(context.project, reflection);
    }
    resolve(project, reflection) {
        if (!(reflection instanceof DeclarationReflection))
            return;
        if (reflection.kindOf(ReflectionKind.ClassOrInterface)) {
            this.postpone(reflection);
            walk(reflection.implementedTypes, (target) => {
                this.postpone(target);
                target.implementedBy ||= [];
                if (!target.implementedBy.some((t) => t.reflection === reflection)) {
                    target.implementedBy.push(ReferenceType.createResolvedReference(reflection.name, reflection, project));
                }
            });
            walk(reflection.extendedTypes, (target) => {
                this.postpone(target);
                target.extendedBy ||= [];
                if (!target.extendedBy.some((t) => t.reflection === reflection)) {
                    target.extendedBy.push(ReferenceType.createResolvedReference(reflection.name, reflection, project));
                }
            });
        }
        function walk(types, callback) {
            if (!types) {
                return;
            }
            types.forEach((type) => {
                if (!(type instanceof ReferenceType)) {
                    return;
                }
                if (!type.reflection ||
                    !(type.reflection instanceof DeclarationReflection)) {
                    return;
                }
                callback(type.reflection);
            });
        }
    }
    postpone(reflection) {
        this.reflections.add(reflection);
    }
    onResolveEnd(context) {
        this.finishResolve(context.project);
    }
    finishResolve(project) {
        this.reflections.forEach((reflection) => {
            if (reflection.implementedBy) {
                reflection.implementedBy.sort((a, b) => {
                    if (a.name === b.name) {
                        return 0;
                    }
                    return a.name > b.name ? 1 : -1;
                });
            }
            let root;
            let hierarchy;
            function push(types) {
                const level = { types: types };
                if (hierarchy) {
                    hierarchy.next = level;
                    hierarchy = level;
                }
                else {
                    root = hierarchy = level;
                }
            }
            if (reflection.extendedTypes) {
                push(reflection.extendedTypes);
            }
            push([
                ReferenceType.createResolvedReference(reflection.name, reflection, project),
            ]);
            hierarchy.isTarget = true;
            if (reflection.extendedBy) {
                push(reflection.extendedBy);
            }
            // No point setting up a hierarchy if there is no hierarchy to display
            if (root.next) {
                reflection.typeHierarchy = root;
            }
        });
    }
}
