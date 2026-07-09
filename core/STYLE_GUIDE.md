# Moonshine Core C++ Style Guide

This document describes the coding policies for the first-party C++ in `core/`.
The goals are reliability and readability. Every rule here is backed by an
automated check so the policies do not drift over time.

Third-party and vendored code (`core/third-party/` and `core/cpp-annote/`) is
**out of scope**: we do not modify it, format it, or lint it.

## Language and philosophy

- **C++20 or later.** The standard is enforced in CMake
  (`CMAKE_CXX_STANDARD 20`, `CXX_STANDARD_REQUIRED ON`, `CXX_EXTENSIONS OFF`).
  The public C++ wrapper (`moonshine-cpp-test`) is additionally built as C++11
  to guarantee the header stays compatible for downstream consumers.
- **Treat C++ as "a better C".** Prefer plain classes, free functions,
  standard containers, and internal exceptions. Lean on the standard library.
- **Keep it readable.** Avoid heavy template metaprogramming, deep class
  hierarchies, and clever abstractions. If a reader has to work to understand
  the control flow, simplify it. A little repetition beats a lot of machinery.

## Memory safety

- **RAII over manual memory management.** Use `std::vector`, `std::string`,
  `std::unique_ptr`, and other owning containers. Ownership should be expressed
  by the type, not by comments.
- **No owning raw pointers, and no `new` / `delete`.** Existing occurrences are
  recorded in `core/.banned-constructs-allowlist` and are being migrated to
  RAII. New code must not add them outside that baseline.
- **Prefer bounds-checked access** for anything that is not performance
  critical: `.at()`, range-`for`, `std::span`, and iterators over raw indexing
  and pointer arithmetic. In a genuinely hot loop, unchecked indexing is
  acceptable — add a brief comment noting that it is deliberate.
- **`reinterpret_cast` is banned** in new code. Where a byte-level view is
  unavoidable (e.g. the C ABI), it must stay on the baseline allow-list.
- **Unsafe C string functions are banned everywhere** (`strcpy`, `strcat`,
  `sprintf`, `vsprintf`, `strncpy`, `strncat`, `gets`). Use `std::string`,
  `snprintf`, or bounded alternatives.

## The C ABI boundary

`moonshine-c-api.*` is the exception firewall for foreign-language bindings:

- Internal exceptions must be caught here and translated to error codes; they
  must never propagate across the C ABI.
- This is the **only** place `malloc` / `free` are allowed, because the ABI
  contract hands ownership of allocated buffers to the caller. These spots are
  on the baseline allow-list and should be documented at the call site.

## Formatting

- Formatting is **Google style**, defined by the repo's `.clang-format`.
- Run `scripts/format-core.sh` to reformat first-party `core/` code in place.
  It runs anywhere `clang-format` is installed (including macOS) and never
  touches vendored trees.
- `scripts/format-core.sh --check` fails if anything is unformatted; it is part
  of the reliability run.

## Automated enforcement

| Policy | Enforced by |
|--------|-------------|
| Formatting | `scripts/format-core.sh --check` (`clang-format`) |
| Banned constructs | `scripts/check-banned-constructs.sh` (also the `check-banned-constructs` ctest) |
| Memory / UB bugs | ASan + UBSan build via `-DMOONSHINE_RELIABILITY=ON` |
| Container bounds / preconditions | `-D_GLIBCXX_ASSERTIONS` (reliability build only) |
| Data races | ThreadSanitizer build (`-DMOONSHINE_SANITIZER=thread`) driving `transcriber-concurrency-test` (many parallel streams), run by `scripts/reliability.sh`. TSan runs with `MOONSHINE_ORT_SINGLE_THREAD=1` so onnxruntime stays on the calling thread (its uninstrumented pool otherwise deadlocks TSan); first-party locking is still exercised. |
| Per-module robustness | libFuzzer targets in `core/reliability/` |
| Reliability lints + static analysis | `core/.clang-tidy` (bounds, `reinterpret_cast`, bugprone, and the `clang-analyzer-*` path-sensitive engine) |

### The banned-constructs baseline

`scripts/check-banned-constructs.sh` enforces two tiers:

1. **Hard ban (zero tolerance):** the unsafe C string functions above.
2. **Baseline gate:** `new`/`delete`, C allocation calls, and `reinterpret_cast`
   are tolerated only in the files listed in
   `core/.banned-constructs-allowlist`. They are blocked in every other
   first-party file. As a module is cleaned up, regenerate the baseline with
   `scripts/check-banned-constructs.sh --update-baseline` to lock in the
   improvement so the construct can never come back.

The `core/reliability/` fuzz harnesses are test-only tooling and are exempt from
the banned-construct rules.

## Running the checks

- **Locally (fast, no build):** `scripts/format-core.sh --check` and
  `scripts/check-banned-constructs.sh`.
- **Full reliability sweep:** `scripts/reliability.sh` builds with sanitizers,
  runs the test suite and clang-tidy, and fuzzes each module on a Linux x86 box
  over SSH. See that script's header for configuration.

## Performance guarantee

None of this reaches production. `MOONSHINE_RELIABILITY` defaults to `OFF`, so
release builds contain no sanitizer instrumentation, no fuzzing code, and no
extra runtime dependencies. Refactors done for safety must not regress the
performance of hot paths; keep unchecked access where it is genuinely needed and
say so in a comment.
