/**
 * An event emitted by the {@link Serializer} class at the very beginning and
 * ending of the a project serialization process.
 *
 * @see {@link Serializer.EVENT_BEGIN}
 * @see {@link Serializer.EVENT_END}
 */
export class SerializeEvent {
    /**
     * The project the renderer is currently processing.
     */
    project;
    output;
    constructor(project, output) {
        this.project = project;
        this.output = output;
    }
}
