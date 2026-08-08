/**
 * Determines the name of the parameter/template/property from the tag content
 * when processing `@param x`
 */
export declare function extractTagName(text: string): {
    name: string;
    newText: string;
};
