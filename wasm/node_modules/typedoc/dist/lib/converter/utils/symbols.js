import ts from "typescript";
export function resolveAliasedSymbol(symbol, checker) {
    const seen = new Set();
    while (ts.SymbolFlags.Alias & symbol.flags) {
        symbol = checker.getAliasedSymbol(symbol);
        // #2438, with declaration files, we might have an aliased symbol which eventually points to itself.
        if (seen.has(symbol))
            return symbol;
        seen.add(symbol);
    }
    return symbol;
}
