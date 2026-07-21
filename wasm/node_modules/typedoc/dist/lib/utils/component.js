import { EventDispatcher } from "./events.js";
/**
 * Component base class.  Has an owner (unless it's the application root component),
 * can dispatch events to its children, and has access to the root Application component.
 *
 * @template O type of component's owner.
 */
export class AbstractComponent extends EventDispatcher {
    /**
     * The owner of this component instance.
     */
    _componentOwner;
    /**
     * The name of this component as set by the `@Component` decorator.
     */
    componentName;
    /**
     * Create new Component instance.
     */
    constructor(owner) {
        super();
        this._componentOwner = owner;
    }
    /**
     * Return the application / root component instance.
     */
    get application() {
        if (this._componentOwner === null) {
            return this;
        }
        return this._componentOwner.application;
    }
    /**
     * Return the owner of this component.
     */
    get owner() {
        return this._componentOwner === null
            ? this
            : this._componentOwner;
    }
}
