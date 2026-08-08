export type Infer<T extends Schema> = T extends Optional<infer U> ? Infer<U> : T extends Guard<infer U> ? U : T extends typeof String ? string : T extends typeof Number ? number : T extends typeof Boolean ? boolean : T extends readonly string[] ? T[number] : T extends readonly [typeof Array, Schema] ? Array<Infer<T[1]>> : {
    -readonly [K in OptionalKeys<T>]?: Infer<Extract<T[K & keyof T], Schema>>;
} & {
    -readonly [K in Exclude<keyof T, OptionalKeys<T> | typeof additionalProperties>]: Infer<Extract<T[K], Schema>>;
};
export type Optional<T extends Schema> = Record<typeof opt, T>;
export type Guard<T> = (x: unknown) => x is T;
type OptionalKeys<T> = keyof {
    [K in keyof T as T[K] extends Optional<any> ? K : never]: 1;
};
declare const opt: unique symbol;
/**
 * Symbol that may be placed on a schema object to define how additional properties are handled.
 * By default, additional properties are not checked.
 */
export declare const additionalProperties: unique symbol;
export type Schema = typeof String | typeof Number | typeof Boolean | readonly string[] | readonly [typeof Array, Schema] | {
    readonly [k: string]: Schema;
    [additionalProperties]?: boolean;
} | Guard<unknown> | Optional<typeof String> | Optional<typeof Number> | Optional<typeof Boolean> | Optional<readonly string[]> | Optional<readonly [typeof Array, Schema]> | Optional<{
    readonly [k: string]: Schema;
    [additionalProperties]?: boolean;
}> | Optional<Guard<unknown>>;
/**
 * Straightforward, fairly dumb, validation helper.
 * @param schema
 * @param obj
 */
export declare function validate<T extends Schema>(schema: T, obj: unknown): obj is Infer<T>;
export declare function optional<T extends Schema>(x: T): Optional<T>;
export declare function isTagString(x: unknown): x is `@${string}`;
export {};
