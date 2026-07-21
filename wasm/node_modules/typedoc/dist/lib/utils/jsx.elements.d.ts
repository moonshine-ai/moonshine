export interface IntrinsicElements {
    a: JsxAElementProps;
    abbr: JsxHtmlGlobalProps;
    address: JsxHtmlGlobalProps;
    area: JsxAreaElementProps;
    article: JsxHtmlGlobalProps;
    aside: JsxHtmlGlobalProps;
    audio: JsxAudioElementProps;
    b: JsxHtmlGlobalProps;
    base: JsxBaseElementProps;
    bdi: JsxHtmlGlobalProps;
    bdo: JsxHtmlGlobalProps;
    blockquote: JsxBlockquoteElementProps;
    body: JsxHtmlGlobalProps;
    br: JsxBrElementProps;
    button: JsxButtonElementProps;
    canvas: JsxCanvasElementProps;
    caption: JsxHtmlGlobalProps;
    cite: JsxHtmlGlobalProps;
    code: JsxHtmlGlobalProps;
    col: JsxColElementProps;
    colgroup: JsxColgroupElementProps;
    data: JsxDataElementProps;
    datalist: JsxHtmlGlobalProps;
    dd: JsxHtmlGlobalProps;
    del: JsxDelElementProps;
    details: JsxDetailsElementProps;
    dfn: JsxHtmlGlobalProps;
    dialog: JsxDialogElementProps;
    div: JsxHtmlGlobalProps;
    dl: JsxHtmlGlobalProps;
    dt: JsxHtmlGlobalProps;
    em: JsxHtmlGlobalProps;
    embed: JsxEmbedElementProps;
    fieldset: JsxFieldsetElementProps;
    figcaption: JsxHtmlGlobalProps;
    figure: JsxHtmlGlobalProps;
    footer: JsxHtmlGlobalProps;
    form: JsxFormElementProps;
    h1: JsxFormElementProps;
    h2: JsxFormElementProps;
    h3: JsxFormElementProps;
    h4: JsxFormElementProps;
    h5: JsxFormElementProps;
    h6: JsxFormElementProps;
    head: JsxHtmlGlobalProps;
    header: JsxHtmlGlobalProps;
    hgroup: JsxHtmlGlobalProps;
    hr: JsxHtmlGlobalProps;
    html: JsxHtmlElementProps;
    i: JsxHtmlGlobalProps;
    iframe: JsxIframeElementProps;
    img: JsxImgElementProps;
    input: JsxInputElementProps;
    ins: JsxInsElementProps;
    kbd: JsxHtmlGlobalProps;
    label: JsxLabelElementProps;
    legend: JsxHtmlGlobalProps;
    li: JsxLiElementProps;
    link: JsxLinkElementProps;
    main: JsxHtmlGlobalProps;
    map: JsxMapElementProps;
    mark: JsxHtmlGlobalProps;
    meta: JsxMetaElementProps;
    meter: JsxMeterElementProps;
    nav: JsxHtmlGlobalProps;
    noscript: JsxHtmlGlobalProps;
    object: JsxObjectElementProps;
    ol: JsxOlElementProps;
    optgroup: JsxOptgroupElementProps;
    option: JsxOptionElementProps;
    output: JsxOutputElementProps;
    p: JsxHtmlGlobalProps;
    param: JsxParamElementProps;
    picture: JsxHtmlGlobalProps;
    portal: JsxPortalElementProps;
    pre: JsxHtmlGlobalProps;
    progress: JsxProgressElementProps;
    q: JsxQElementProps;
    rp: JsxHtmlGlobalProps;
    rt: JsxHtmlGlobalProps;
    ruby: JsxHtmlGlobalProps;
    s: JsxHtmlGlobalProps;
    samp: JsxHtmlGlobalProps;
    script: JsxScriptElementProps;
    section: JsxHtmlGlobalProps;
    select: JsxSelectElementProps;
    slot: JsxSlotElementProps;
    small: JsxHtmlGlobalProps;
    source: JsxSourceElementProps;
    span: JsxHtmlGlobalProps;
    strong: JsxHtmlGlobalProps;
    style: JsxStyleElementProps;
    sub: JsxHtmlGlobalProps;
    summary: JsxHtmlGlobalProps;
    sup: JsxHtmlGlobalProps;
    table: JsxHtmlGlobalProps;
    tbody: JsxHtmlGlobalProps;
    td: JsxTdElementProps;
    template: JsxHtmlGlobalProps;
    textarea: JsxTextareaElementProps;
    tfoot: JsxHtmlGlobalProps;
    th: JsxThElementProps;
    thead: JsxHtmlGlobalProps;
    time: JsxTimeElementProps;
    title: JsxHtmlGlobalProps;
    tr: JsxHtmlGlobalProps;
    track: JsxTrackElementProps;
    u: JsxHtmlGlobalProps;
    ul: JsxHtmlGlobalProps;
    var: JsxHtmlGlobalProps;
    video: JsxVideoElementProps;
    wbr: JsxHtmlGlobalProps;
    svg: JsxSvgElementProps;
    g: JsxGElementProps;
    path: JsxPathElementProps;
    rect: JsxRectElementProps;
    circle: JsxCircleElementProps;
    ellipse: JsxEllipseElementProps;
    polygon: JsxPolygonElementProps;
    polyline: JsxPolylineElementProps;
    line: JsxLineElementProps;
    use: JsxUseElementProps;
    text: JsxTextElementProps;
}
export declare function JsxFragment(): never;
export type JsxComponent<P> = (props: P) => JsxElement | null | undefined;
export interface JsxElement {
    tag: typeof JsxFragment | string | JsxComponent<any>;
    props: object | null;
    children: JsxChildren[];
}
export type JsxChildren = JsxElement | string | number | null | undefined | JsxChildren[];
/**
 * The common properties that may appear on any HTML element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/HTML/Global_attributes#list_of_global_attributes
 */
export interface JsxHtmlGlobalProps {
    accessKey?: string;
    autocapitalize?: string;
    children?: JsxChildren;
    class?: string;
    contentEditable?: string;
    [data: `data-${string}`]: string;
    dir?: string;
    draggable?: boolean;
    enterKeyHint?: string;
    hidden?: boolean;
    id?: string;
    inputMode?: string;
    is?: string;
    itemId?: string;
    itemProp?: string;
    itemRef?: string;
    itemScope?: string;
    itemType?: string;
    lang?: string;
    nonce?: string;
    part?: string;
    role?: string;
    slot?: string;
    spellcheck?: boolean;
    style?: string;
    tabIndex?: number;
    title?: string;
    translate?: boolean;
    /**
     * Default: 'auto'. true and 'auto' are equivalent
     *
     * See [MDN](https://developer.mozilla.org/en-US/docs/Web/API/Popover_API) for more details
     */
    popover?: boolean | "auto" | "manual";
    /**
     * It must be the popover element id, see [MDN](https://developer.mozilla.org/en-US/docs/Web/API/Popover_API)
     */
    popovertarget?: string;
    popovertargetaction?: "hide" | "show" | "toggle";
}
/**
 * Properties permitted on the `<a>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/a
 */
export interface JsxAElementProps extends JsxHtmlGlobalProps {
    download?: string;
    href?: string;
    hreflang?: string;
    ping?: string;
    referrerPolicy?: string;
    rel?: string;
    target?: string;
    type?: string;
}
/**
 * Properties permitted on the `<area>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/area
 */
export interface JsxAreaElementProps extends JsxHtmlGlobalProps {
    alt?: string;
    coords?: string;
    download?: string;
    href?: string;
    hreflang?: string;
    ping?: string;
    referrerPolicy?: string;
    rel?: string;
    shape?: "rect" | "circle" | "poly" | "default";
    target?: "_self" | "_blank" | "_parent" | "_top";
}
/**
 * Properties permitted on the `<audio>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/audio
 */
export interface JsxAudioElementProps extends JsxHtmlGlobalProps {
    autoplay?: boolean;
    controls?: boolean;
    crossOrigin?: "anonymous" | "use-credentials";
    loop?: boolean;
    muted?: boolean;
    preload?: "none" | "metadata" | "auto" | "";
    src?: string;
}
/**
 * Properties permitted on the `<base>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/base
 */
export interface JsxBaseElementProps extends JsxHtmlGlobalProps {
    href?: string;
    target?: string;
}
/**
 * Properties permitted on the `<blockquote>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/blockquote
 */
export interface JsxBlockquoteElementProps extends JsxHtmlGlobalProps {
    cite?: string;
}
/**
 * Properties permitted on the `<br>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/br
 */
export interface JsxBrElementProps extends JsxHtmlGlobalProps {
    clear?: string;
}
/**
 * Properties permitted on the `<button>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/button
 */
export interface JsxButtonElementProps extends JsxHtmlGlobalProps {
    autofocus?: boolean;
    disabled?: boolean;
    form?: string;
    formAction?: string;
    formEnctype?: "application/x-www-form-urlencoded" | "multipart/form-data" | "text/plain";
    formMethod?: "get" | "post";
    formNoValidate?: boolean;
    formTarget?: "_self" | "_blank" | "_parent" | "_top";
    name?: string;
    type?: "submit" | "reset" | "button";
    value?: string;
}
/**
 * Properties permitted on the `<canvas>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/canvas
 */
export interface JsxCanvasElementProps extends JsxHtmlGlobalProps {
    height?: number;
    width?: number;
}
/**
 * Properties permitted on the `<col>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/col
 */
export interface JsxColElementProps extends JsxHtmlGlobalProps {
    span?: number;
}
/**
 * Properties permitted on the `<colgroup>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/colgroup
 */
export interface JsxColgroupElementProps extends JsxHtmlGlobalProps {
    span?: number;
}
/**
 * Properties permitted on the `<data>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/data
 */
export interface JsxDataElementProps extends JsxHtmlGlobalProps {
    value?: string;
}
/**
 * Properties permitted on the `<del>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/del
 */
export interface JsxDelElementProps extends JsxHtmlGlobalProps {
    cite?: string;
    dateTime?: string;
}
/**
 * Properties permitted on the `<details>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/details
 */
export interface JsxDetailsElementProps extends JsxHtmlGlobalProps {
    open?: boolean;
}
/**
 * Properties permitted on the `<dialog>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/dialog
 */
export interface JsxDialogElementProps extends JsxHtmlGlobalProps {
    open?: boolean;
}
/**
 * Properties permitted on the `<embed>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/embed
 */
export interface JsxEmbedElementProps extends JsxHtmlGlobalProps {
    height?: number;
    src?: string;
    type?: string;
    width?: number;
}
/**
 * Properties permitted on the `<fieldset>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/fieldset
 */
export interface JsxFieldsetElementProps extends JsxHtmlGlobalProps {
    disabled?: boolean;
    form?: string;
    name?: string;
}
/**
 * Properties permitted on the `<form>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/form
 */
export interface JsxFormElementProps extends JsxHtmlGlobalProps {
    "accept-charset"?: string;
    action?: string;
    enctype?: "application/x-www-form-urlencoded" | "multipart/form-data" | "text/plain";
    autocomplete?: string;
    method?: "get" | "post" | "dialog";
    name?: string;
    noValidate?: boolean;
    rel?: string;
    target?: "_self" | "_blank" | "_parent" | "_top";
}
/**
 * Properties permitted on the `<html>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/html
 */
export interface JsxHtmlElementProps extends JsxHtmlGlobalProps {
    xmlns?: string;
}
/**
 * Properties permitted on the `<iframe>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/iframe
 */
export interface JsxIframeElementProps extends JsxHtmlGlobalProps {
    allow?: string;
    height?: number;
    name?: string;
    referrerPolicy?: "no-referrer" | "no-referrer-when-downgrade" | "origin" | "origin-when-cross-origin" | "same-origin" | "strict-origin" | "strict-origin-when-cross-origin" | "unsafe-url";
    sandbox?: string;
    src?: string;
    srcdoc?: string;
    width?: number;
}
/**
 * Properties permitted on the `<img>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/img
 */
export interface JsxImgElementProps extends JsxHtmlGlobalProps {
    alt?: string;
    crossOrigin?: "anonymous" | "use-credentials";
    decoding?: "async" | "sync" | "auto";
    height?: number;
    isMap?: boolean;
    referrerPolicy?: "no-referrer" | "no-referrer-when-downgrade" | "origin" | "origin-when-cross-origin" | "same-origin" | "strict-origin" | "strict-origin-when-cross-origin" | "unsafe-url";
    sizes?: string;
    src: string;
    srcset?: string;
    width?: number;
}
/**
 * Properties permitted on the `<input>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/input
 */
export interface JsxInputElementProps extends JsxHtmlGlobalProps {
    accept?: string;
    alt?: string;
    autocomplete?: string;
    autofocus?: boolean;
    capture?: string;
    checked?: boolean;
    dirName?: string;
    disabled?: boolean;
    form?: string;
    formAction?: string;
    formEnctype?: string;
    formMethod?: string;
    formNoValidate?: boolean;
    formTarget?: string;
    height?: number;
    list?: string;
    max?: number;
    maxLength?: number;
    min?: string;
    minLength?: number;
    multiple?: boolean;
    name?: string;
    pattern?: string;
    placeholder?: string;
    readOnly?: boolean;
    required?: boolean;
    size?: number;
    src?: string;
    step?: number;
    type?: "button" | "checkbox" | "color" | "date" | "datetime-local" | "email" | "file" | "hidden" | "image" | "month" | "number" | "password" | "radio" | "range" | "reset" | "search" | "submit" | "tel" | "text" | "time" | "url" | "week";
    value?: string;
    width?: number;
}
/**
 * Properties permitted on the `<ins>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/ins
 */
export interface JsxInsElementProps extends JsxHtmlGlobalProps {
    cite?: string;
    dateTime?: string;
}
/**
 * Properties permitted on the `<label>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/label
 */
export interface JsxLabelElementProps extends JsxHtmlGlobalProps {
    for?: string;
}
/**
 * Properties permitted on the `<li>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/li
 */
export interface JsxLiElementProps extends JsxHtmlGlobalProps {
    value?: number;
}
/**
 * Properties permitted on the `<link>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/link
 */
export interface JsxLinkElementProps extends JsxHtmlGlobalProps {
    as?: "audio" | "document" | "embed" | "fetch" | "font" | "image" | "object" | "script" | "style" | "track" | "video" | "worker";
    crossOrigin?: "anonymous" | "use-credentials";
    disabled?: boolean;
    href?: string;
    hreflang?: string;
    imageSizes?: string;
    imageSrcset?: string;
    media?: string;
    rel?: string;
    sizes?: string;
    type?: string;
}
/**
 * Properties permitted on the `<map>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/map
 */
export interface JsxMapElementProps extends JsxHtmlGlobalProps {
    name?: string;
}
/**
 * Properties permitted on the `<meta>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/meta
 */
export interface JsxMetaElementProps extends JsxHtmlGlobalProps {
    "http-equiv"?: "content-security-policy" | "content-type" | "default-style" | "x-ua-compatible" | "refresh";
    charset?: "utf-8";
    content?: string;
    name?: string;
}
/**
 * Properties permitted on the `<meter>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/meter
 */
export interface JsxMeterElementProps extends JsxHtmlGlobalProps {
    form?: string;
    high?: number;
    low?: number;
    max?: number;
    min?: number;
    optimum?: number;
    value?: number;
}
/**
 * Properties permitted on the `<object>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/object
 */
export interface JsxObjectElementProps extends JsxHtmlGlobalProps {
    data?: string;
    form?: string;
    height?: number;
    name?: string;
    type?: string;
    useMap?: string;
    width?: number;
}
/**
 * Properties permitted on the `<ol>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/ol
 */
export interface JsxOlElementProps extends JsxHtmlGlobalProps {
    reversed?: boolean;
    start?: number;
    type?: "a" | "A" | "i" | "I" | "1";
}
/**
 * Properties permitted on the `<optgroup>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/optgroup
 */
export interface JsxOptgroupElementProps extends JsxHtmlGlobalProps {
    disabled?: boolean;
    label: string;
}
/**
 * Properties permitted on the `<option>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/option
 */
export interface JsxOptionElementProps extends JsxHtmlGlobalProps {
    disabled?: boolean;
    label?: string;
    selected?: boolean;
    value?: string;
}
/**
 * Properties permitted on the `<output>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/output
 */
export interface JsxOutputElementProps extends JsxHtmlGlobalProps {
    for?: string;
    form?: string;
    name?: string;
}
/**
 * Properties permitted on the `<param>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/param
 */
export interface JsxParamElementProps extends JsxHtmlGlobalProps {
    name?: string;
    value?: string;
}
/**
 * Properties permitted on the `<portal>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/portal
 */
export interface JsxPortalElementProps extends JsxHtmlGlobalProps {
    referrerPolicy?: "no-referrer" | "no-referrer-when-downgrade" | "origin" | "origin-when-cross-origin" | "same-origin" | "strict-origin" | "strict-origin-when-cross-origin" | "unsafe-url";
    src: string;
}
/**
 * Properties permitted on the `<progress>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/progress
 */
export interface JsxProgressElementProps extends JsxHtmlGlobalProps {
    max?: number;
    value?: number;
}
/**
 * Properties permitted on the `<q>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/q
 */
export interface JsxQElementProps extends JsxHtmlGlobalProps {
    cite?: string;
}
/**
 * Properties permitted on the `<script>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/script
 */
export interface JsxScriptElementProps extends JsxHtmlGlobalProps {
    async?: boolean;
    crossOrigin?: "anonymous" | "use-credentials";
    defer?: boolean;
    integrity?: string;
    noModule?: boolean;
    referrerPolicy?: "no-referrer" | "no-referrer-when-downgrade" | "origin" | "origin-when-cross-origin" | "same-origin" | "strict-origin" | "strict-origin-when-cross-origin" | "unsafe-url";
    src?: string;
    type?: "module" | (string & {});
}
/**
 * Properties permitted on the `<select>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/select
 */
export interface JsxSelectElementProps extends JsxHtmlGlobalProps {
    autocomplete?: string;
    autofocus?: boolean;
    disabled?: boolean;
    form?: string;
    multiple?: boolean;
    name?: string;
    required?: boolean;
    size?: number;
}
/**
 * Properties permitted on the `<slot>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/slot
 */
export interface JsxSlotElementProps extends JsxHtmlGlobalProps {
    name?: string;
}
/**
 * Properties permitted on the `<source>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/source
 */
export interface JsxSourceElementProps extends JsxHtmlGlobalProps {
    media?: string;
    sizes?: string;
    src?: string;
    srcset?: string;
    type?: string;
}
/**
 * Properties permitted on the `<style>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/style
 */
export interface JsxStyleElementProps extends JsxHtmlGlobalProps {
    media?: string;
    type?: string;
    nonce?: string;
}
/**
 * Properties permitted on the `<td>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/td
 */
export interface JsxTdElementProps extends JsxHtmlGlobalProps {
    colSpan?: number;
    headers?: string;
    rowSpan?: number;
}
/**
 * Properties permitted on the `<textarea>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/textarea
 */
export interface JsxTextareaElementProps extends JsxHtmlGlobalProps {
    autocomplete?: string;
    autofocus?: boolean;
    cols?: number;
    disabled?: boolean;
    maxLength?: number;
    minLength?: number;
    name?: string;
    placeholder?: string;
    readOnly?: boolean;
    required?: boolean;
    rows?: number;
    wrap?: "hard" | "soft";
}
/**
 * Properties permitted on the `<th>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/th
 */
export interface JsxThElementProps extends JsxHtmlGlobalProps {
    abbr?: string;
    colSpan?: number;
    headers?: string;
    rowSpan?: number;
    scope?: "row" | "col" | "rowgroup" | "colgroup";
}
/**
 * Properties permitted on the `<time>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/time
 */
export interface JsxTimeElementProps extends JsxHtmlGlobalProps {
    dateTime?: string;
}
/**
 * Properties permitted on the `<track>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/track
 */
export interface JsxTrackElementProps extends JsxHtmlGlobalProps {
    default?: boolean;
    kind?: "subtitles" | "captions" | "descriptions" | "chapters" | "metadata";
    label?: string;
    src?: string;
    srclang?: string;
}
/**
 * Properties permitted on the `<video>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/video
 */
export interface JsxVideoElementProps extends JsxHtmlGlobalProps {
    autoplay?: boolean;
    controls?: boolean;
    crossOrigin?: "anonymous" | "use-credentials";
    height?: number;
    loop?: boolean;
    muted?: boolean;
    playsInline?: boolean;
    poster?: string;
    preload?: string;
    src?: string;
    width?: number;
}
export interface JsxSvgCoreProps {
    id?: string;
    lang?: string;
    tabindex?: string;
    "xml:base"?: string;
    xmlns?: string;
}
export interface JsxSvgStyleProps {
    class?: string;
    style?: string;
}
export interface JsxSvgConditionalProcessingProps {
    systemLanguage?: string;
}
export interface JsxSvgPresentationProps {
    "alignment-baseline"?: "baseline" | "text-bottom" | "text-before-edge" | "middle" | "central" | "text-after-edge" | "ideographic" | "alphabetic" | "hanging" | "mathematical" | "top" | "center" | "bottom";
    "baseline-shift"?: number | "sub" | "super";
    "clip-path"?: string;
    "clip-rule"?: "nonzero" | "evenodd" | "inherit";
    color?: string;
    "color-interpolation"?: "auto" | "sRGB" | "linearRGB";
    "color-interpolation-filters"?: "auto" | "sRGB" | "linearRGB";
    cursor?: string;
    direction?: "ltr" | "rtl";
    display?: string;
    "dominant-baseline"?: "auto" | "text-bottom" | "alphabetic" | "ideographic" | "middle" | "central" | "mathematical" | "hanging" | "text-top";
    fill?: string;
    "fill-opacity"?: number;
    "fill-rule"?: "nonzero" | "evenodd";
    filter?: string;
    "flood-color"?: string;
    "flood-opacity"?: number;
    "font-family"?: string;
    "font-size"?: string;
    "font-size-adjust"?: "none" | number;
    "font-stretch"?: string;
    "font-style"?: "normal" | "italic" | "oblique" | string;
    "font-variant"?: string;
    "font-weight"?: "normal" | "bold" | "bolder" | "lighter" | number | string;
    "image-rendering"?: "auto" | "optimizeSpeed" | "optimizeQuality";
    "letter-spacing"?: string;
    "lighting-color"?: string;
    "marker-end"?: string;
    "marker-mid"?: string;
    "marker-start"?: string;
    mask?: string;
    opacity?: number;
    overflow?: "visible" | "hidden" | "scroll" | "auto";
    "pointer-events"?: "bounding-box" | "visiblePainted" | "visibleFill" | "visibleStroke" | "visible" | "painted" | "fill" | "stroke" | "all" | "none";
    "shape-rendering"?: "auto" | "optimizeSpeed" | "crispEdges" | "geometricPrecision";
    "stop-color"?: string;
    "stop-opacity"?: string;
    stroke?: string;
    "stroke-dasharray"?: string;
    "stroke-dashoffset"?: string;
    "stroke-linecap"?: "butt" | "round" | "square";
    "stroke-linejoin"?: "arcs" | "bevel |miter" | "miter-clip" | "round";
    "stroke-miterlimit"?: number;
    "stroke-opacity"?: string | number;
    "stroke-width"?: string | number;
    "text-anchor"?: "start" | "middle" | "end";
    "text-decoration"?: string;
    "text-rendering"?: "auto" | "optimizeSpeed" | "optimizeLegibility" | "geometricPrecision";
    transform?: string;
    "transform-origin"?: string;
    "unicode-bidi"?: "normal" | "embed" | "isolate" | "bidi-override" | "isolate-override" | "plaintext";
    "vector-effect"?: "none" | "non-scaling-stroke" | "non-scaling-size" | "non-rotation" | "fixed-position";
    visibility?: "visible" | "hidden" | "collapse";
    "word-spacing"?: string;
    "writing-mode"?: "horizontal-tb" | "vertical-rl" | "vertical-lr";
}
/**
 * Properties permitted on the `<svg>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/SVG/Element/svg
 */
export interface JsxSvgElementProps extends JsxSvgCoreProps, JsxSvgStyleProps, JsxSvgPresentationProps {
    height?: string | number;
    preserveAspectRatio?: `${"none" | "xMinYMin" | "xMaxYMin" | "xMinYMid" | "xMaxYMid" | "xMinYMax" | "xMidYMax" | "xMaxYMax"}${"" | " meet" | " slice"}`;
    viewBox?: string;
    width?: string | number;
    x?: string | number;
    y?: string | number;
}
/**
 * Properties permitted on the `<g>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/SVG/Element/g
 */
export interface JsxGElementProps extends JsxSvgCoreProps, JsxSvgStyleProps, JsxSvgConditionalProcessingProps, JsxSvgPresentationProps {
}
/**
 * Properties permitted on the `<path>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/SVG/Element/path
 */
export interface JsxPathElementProps extends JsxSvgCoreProps, JsxSvgStyleProps, JsxSvgConditionalProcessingProps, JsxSvgPresentationProps {
    d: string;
    pathLength?: number;
}
/**
 * Properties permitted on the `<rect>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/SVG/Element/rect
 */
export interface JsxRectElementProps extends JsxSvgCoreProps, JsxSvgStyleProps, JsxSvgConditionalProcessingProps, JsxSvgPresentationProps {
    height?: string | number;
    pathLength?: number;
    rx?: string | number;
    ry?: string | number;
    width?: string | number;
    x?: string | number;
    y?: string | number;
}
/**
 * Properties permitted on the `<circle>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/SVG/Element/circle
 */
export interface JsxCircleElementProps extends JsxSvgCoreProps, JsxSvgStyleProps, JsxSvgConditionalProcessingProps, JsxSvgPresentationProps {
    cx?: string | number;
    cy?: string | number;
    r?: string | number;
    pathLength?: number;
}
/**
 * Properties permitted on the `<ellipse>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/SVG/Element/ellipse
 */
export interface JsxEllipseElementProps extends JsxSvgCoreProps, JsxSvgStyleProps, JsxSvgConditionalProcessingProps, JsxSvgPresentationProps {
    cx?: string | number;
    cy?: string | number;
    rx?: string | number;
    ry?: string | number;
    pathLength?: number;
}
/**
 * Properties permitted on the `<polygon>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/SVG/Element/polygon
 */
export interface JsxPolygonElementProps extends JsxSvgCoreProps, JsxSvgStyleProps, JsxSvgConditionalProcessingProps, JsxSvgPresentationProps {
    points?: string;
    pathLength?: number;
}
/** Properties permitted on the `<polyline>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/SVG/Element/polyline
 */
export interface JsxPolylineElementProps extends JsxSvgCoreProps, JsxSvgStyleProps, JsxSvgConditionalProcessingProps, JsxSvgPresentationProps {
    points?: string;
    pathLength?: number;
}
/** Properties permitted on the `<line>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/SVG/Element/line
 */
export interface JsxLineElementProps extends JsxSvgCoreProps, JsxSvgStyleProps, JsxSvgConditionalProcessingProps, JsxSvgPresentationProps {
    x1?: string | number;
    y1?: string | number;
    x2?: string | number;
    y2?: string | number;
    pathLength?: number;
}
/**
 * Properties permitted on the `<use>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/SVG/Element/use
 */
export interface JsxUseElementProps extends JsxSvgCoreProps, JsxSvgStyleProps, JsxSvgConditionalProcessingProps, JsxSvgPresentationProps {
    href: string;
    x?: string | number;
    y?: string | number;
    width?: string | number;
    height?: string | number;
}
/**
 * Properties permitted on the `<text>` element.
 *
 * Reference: https://developer.mozilla.org/en-US/docs/Web/SVG/Element/text
 */
export interface JsxTextElementProps extends JsxSvgCoreProps, JsxSvgStyleProps, JsxSvgPresentationProps {
    x?: string | number;
    y?: string | number;
}
