/**
 * Custom JSX module designed specifically for TypeDoc's needs.
 * When overriding a default TypeDoc theme output, your implementation must create valid {@link Element}
 * instances, which can be most easily done by using TypeDoc's JSX implementation. To use it, set up
 * your tsconfig with the following compiler options:
 * ```json
 * {
 *     "jsx": "react",
 *     "jsxFactory": "JSX.createElement",
 *     "jsxFragmentFactory": "JSX.Fragment"
 * }
 * ```
 * @summary Custom JSX module designed specifically for TypeDoc's needs.
 * @module
 */
import type { IntrinsicElements, JsxElement, JsxChildren, JsxComponent, JsxHtmlGlobalProps } from "./jsx.elements.js";
export type { JsxElement as Element, JsxChildren as Children, JsxComponent, } from "./jsx.elements.js";
export { JsxFragment as Fragment } from "./jsx.elements.js";
/**
 * Used to inject HTML directly into the document.
 */
export declare function Raw(_props: {
    html: string;
}): null;
/**
 * TypeScript's rules for looking up the JSX.IntrinsicElements and JSX.Element
 * interfaces are incredibly strange. It will find them if they are included as
 * a namespace under the createElement function, or globally, or, apparently, if
 * a JSX namespace is declared at the same scope as the factory function.
 * Hide this in the docs, hopefully someday TypeScript improves this and allows
 * looking adjacent to the factory function and we can get rid of this phantom namespace.
 * @hidden
 */
export declare namespace JSX {
    export { IntrinsicElements, JsxElement as Element, JsxHtmlGlobalProps as IntrinsicAttributes, };
}
/**
 * JSX factory function to create an "element" that can later be rendered with {@link renderElement}
 * @param tag
 * @param props
 * @param children
 */
export declare function createElement(tag: string | JsxComponent<any>, props: object | null, ...children: JsxChildren[]): JsxElement;
export declare function setRenderSettings(options: {
    pretty: boolean;
}): void;
export declare function renderElement(element: JsxElement | null | undefined): string;
/**
 * Render an element to text, stripping out any HTML tags.
 * This is roughly equivalent to getting `innerText` on a rendered element.
 * @internal
 */
export declare function renderElementToText(element: JsxElement | null | undefined): string;
