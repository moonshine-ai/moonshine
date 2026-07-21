import { binaryFindPartition } from "./array.js";
// I don't like this, but it's necessary so that the lineStarts property isn't
// visible in the `MinimalSourceFile` type. Even when private it causes compilation
// errors downstream.
const lineStarts = new WeakMap();
export class MinimalSourceFile {
    fileName;
    text;
    constructor(text, fileName) {
        this.fileName = fileName;
        // This is unfortunate, but the yaml library we use relies on the source
        // text using LF line endings https://github.com/eemeli/yaml/issues/127.
        // If we don't do this, in a simple document which includes a single key
        // like:
        // ---<CR><LF>
        // title: Windows line endings<CR><LF>
        // ---<CR><LF>
        // we'll end up with a parsed title of "Windows line endings\r"
        this.text = text.replaceAll("\r\n", "\n");
        lineStarts.set(this, [0]);
    }
    getLineAndCharacterOfPosition(pos) {
        if (pos < 0 || pos >= this.text.length) {
            throw new Error("pos must be within the range of the file.");
        }
        const starts = lineStarts.get(this);
        while (pos >= starts[starts.length - 1]) {
            const nextStart = this.text.indexOf("\n", starts[starts.length - 1]);
            if (nextStart === -1) {
                starts.push(Infinity);
            }
            else {
                starts.push(nextStart + 1);
            }
        }
        const line = binaryFindPartition(starts, (x) => x > pos) - 1;
        return {
            character: pos - starts[line],
            line,
        };
    }
}
