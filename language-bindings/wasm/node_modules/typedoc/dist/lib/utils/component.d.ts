import type { Application } from "../application.js";
import { EventDispatcher } from "./events.js";
/**
 * Exposes a reference to the root Application component.
 */
export interface ComponentHost {
    readonly application: Application;
}
export interface Component<E extends Record<keyof E, unknown[]> = {}> extends AbstractComponent<ComponentHost, E> {
}
/**
 * Component base class.  Has an owner (unless it's the application root component),
 * can dispatch events to its children, and has access to the root Application component.
 *
 * @template O type of component's owner.
 */
export declare abstract class AbstractComponent<O extends ComponentHost, E extends Record<keyof E, unknown[]>> extends EventDispatcher<E> implements ComponentHost {
    /**
     * The owner of this component instance.
     */
    private _componentOwner;
    /**
     * The name of this component as set by the `@Component` decorator.
     */
    componentName: string;
    /**
     * Create new Component instance.
     */
    constructor(owner: O);
    /**
     * Return the application / root component instance.
     */
    get application(): Application;
    /**
     * Return the owner of this component.
     */
    get owner(): O;
}
