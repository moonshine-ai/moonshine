import { deflate } from "zlib";
import { promisify } from "util";
const deflateP = promisify(deflate);
/**
 * Compresses a JSON-serializable object into a Base64-encoded deflate string.
 *
 * @param data - The JSON-serializable object to compress.
 * @returns A promise that resolves to a Base64-encoded string of the deflate-compressed data.
 */
export async function compressJson(data) {
    const gz = await deflateP(Buffer.from(JSON.stringify(data)));
    return gz.toString("base64");
}
