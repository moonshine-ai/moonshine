import type { ProjectReflection } from "../models/index.js";
import type { ProjectReflection as JSONProjectReflection } from "./schema.js";
/**
 * An event emitted by the {@link Serializer} class at the very beginning and
 * ending of the a project serialization process.
 *
 * @see {@link Serializer.EVENT_BEGIN}
 * @see {@link Serializer.EVENT_END}
 */
export declare class SerializeEvent {
    /**
     * The project the renderer is currently processing.
     */
    readonly project: ProjectReflection;
    output: JSONProjectReflection | undefined;
    constructor(project: ProjectReflection, output?: JSONProjectReflection);
}
