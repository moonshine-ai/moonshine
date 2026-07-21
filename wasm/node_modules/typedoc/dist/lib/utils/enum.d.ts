export declare function getEnumFlags<T extends number>(flags: T): T[];
export declare function removeFlag<T extends number>(flag: T, remove: T & {}): T;
export declare function hasAllFlags(flags: number, check: number): boolean;
export declare function hasAnyFlag(flags: number, check: number): boolean;
export declare function debugFlags(Enum: object, flags: number): string[];
export declare function getEnumKeys(Enum: object): string[];
export type EnumKeys<E extends object> = {
    [K in keyof E]: number extends E[K] ? K : never;
}[keyof E] & {};
