import * as shiki from "@gerrit0/mini-shiki";
export declare function loadHighlighter(lightTheme: shiki.BundledTheme, darkTheme: shiki.BundledTheme, langs: shiki.BundledLanguage[], ignoredLangs: string[] | undefined): Promise<void>;
export declare function isSupportedLanguage(lang: string): boolean;
export declare function getSupportedLanguages(): string[];
export declare function getSupportedThemes(): string[];
export declare function isLoadedLanguage(lang: string): boolean;
export declare function highlight(code: string, lang: string): string;
export declare function getStyles(): string;
