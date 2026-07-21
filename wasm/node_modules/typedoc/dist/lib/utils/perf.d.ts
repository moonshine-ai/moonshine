export declare function bench<T extends (..._: any) => any>(fn: T, name?: string): T;
export declare function Bench<T extends (..._: any) => any>(value: T, context: ClassMethodDecoratorContext): T;
export declare function measure<T>(cb: () => T): T;
