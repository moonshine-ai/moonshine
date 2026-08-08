/**
 * Compresses a JSON-serializable object into a Base64-encoded deflate string.
 *
 * @param data - The JSON-serializable object to compress.
 * @returns A promise that resolves to a Base64-encoded string of the deflate-compressed data.
 */
export declare function compressJson(data: any): Promise<string>;
