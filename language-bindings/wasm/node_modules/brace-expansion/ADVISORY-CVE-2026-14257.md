# DoS via unbounded expansion length — out-of-memory process crash

- **CVE:** CVE-2026-14257
- **Package:** brace-expansion (npm)
- **Reporter:** @bnbdr
- **Severity (proposed):** High — `CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H` (7.5)
- **Weakness:** CWE-770 (Allocation of Resources Without Limits or Throttling) / CWE-400 (Uncontrolled Resource Consumption)
- **Affected:** all versions up to and including `5.0.7` (the `1.x`, `2.x`, `3.x` and `4.x` lines share the same combine logic and are expected to be affected)

### Summary

`expand()` bounds the *number* of results it produces (the `max` option,
`100_000` by default) but not their *length*. By chaining many brace groups,
an attacker keeps the result count under `max` while making every result grow
with the number of groups. Building `max` long results — plus the intermediate
arrays combined at each brace group — exhausts memory and crashes the Node
process with an **uncatchable** out-of-memory error. `try/catch` around
`expand()` does not help: the fatal error terminates the process.

A ~7.5 KB input (`'{a,b}'.repeat(1500)`) is enough to crash a default Node
process.

### Details

For `N` chained brace groups such as `'{a,b}'.repeat(N)`:

- the result count is `2^N`, immediately capped at `max` (`100_000`), so the
  `max` protection appears to hold, but
- each result is `N` characters long, so the total output size is
  `max × N` characters, which grows without bound in `N`.

`expand_` combines each brace set with the fully-expanded tail:

```js
const post = m.post.length ? expand_(m.post, max, false) : ['']
...
for (let j = 0; j < N.length; j++) {
  for (let k = 0; k < post.length && expansions.length < max; k++) {
    const expansion = pre + N[j] + post[k]   // grows one group longer per level
    ...
    expansions.push(expansion)
  }
}
```

The loop guard `expansions.length < max` limits how many strings are built, but
nothing limits how long they get. Each recursion level materializes another
array of up to `max` strings, one character longer than the level below, and —
because V8 represents `pre + N[j] + post[k]` as a cons-string (rope) that
references `post[k]` — those intermediate strings stay reachable through the
whole chain. Memory therefore scales with `max × N`.

Measured on `5.0.7` (`'{a,b}'.repeat(N)`, default `max`):

| groups (N) | input bytes | result count | peak RSS |
|---|---|---|---|
| 20 | 100 | 100,000 | ~80 MB |
| 50 | 250 | 100,000 | ~214 MB |
| 100 | 500 | 100,000 | ~409 MB |
| 300 | 1,500 | 100,000 | ~1,148 MB |
| 1500 | 7,500 | — | **OOM crash** |

### Proof of concept

```js
const { expand } = require('brace-expansion')

// ~7.5 KB input — crashes the process with a fatal, uncatchable OOM:
//   FATAL ERROR: ... JavaScript heap out of memory
try {
  expand('{a,b}'.repeat(1500))
} catch (e) {
  // never reached — the process is already dead
}
```

### Impact

Any application that passes attacker-influenced strings to
`brace-expansion.expand()` — directly, or transitively via `minimatch` / `glob`
brace patterns — can be crashed by a small request. Because the failure is a
fatal V8 out-of-memory error rather than a thrown exception, it cannot be caught
and it takes down the whole worker/process, denying service.

### Remediation

Upgrade to a patched release. The fix bounds the total number of characters a
single `expand()` call may accumulate (`EXPANSION_MAX_LENGTH`, default
`4_000_000`, configurable via a new `maxLength` option), applied inside the
output-building loops so intermediate arrays are bounded too. Once the limit is
reached, output is truncated — consistent with how `max` already truncates —
instead of growing without bound. The limit sits well above any realistic
expansion (100,000 results hitting `max` measure ~1M characters), so legitimate
input is unaffected.

After the fix, `'{a,b}'.repeat(1500)` returns a bounded, truncated result in
~0.7 s using ~340 MB and never crashes, including under a constrained 512 MB
heap.

The fix bounds memory but the algorithm still rebuilds intermediate arrays at
each level (roughly `O(N × maxLength)` work on this input class). A streaming
rewrite that produces output in `O(total output size)` can be a non-urgent
follow-up.

If immediate upgrade isn't possible, avoid passing untrusted input to
`expand()` / glob brace patterns, or pass a small explicit `max` **and**
`maxLength`.
