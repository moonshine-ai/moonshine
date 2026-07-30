#!/usr/bin/env python3
"""Regenerate core/moonshine-model-file-metadata.generated.cpp.

This is the single source of truth for per-file integrity metadata (expected
size in bytes and CRC32C checksum) that the C API joins into the download
manifest returned by moonshine_get_stt_dependencies /
moonshine_get_embedding_dependencies.

The set of files is enumerated directly from the compiled catalog (via the
moonshine C API), so this never drifts from the catalog itself. For each file we
issue an HTTP HEAD to its CDN URL and read `content-length` and the
`x-goog-hash: crc32c=...` header that Google Cloud Storage returns for every
object (including composite uploads).

Usage:
    # Build libmoonshine first (scripts/test-core.sh or a plain cmake build) so
    # the Python binding can load core/build/libmoonshine.dylib, then:
    python3 scripts/generate-model-file-metadata.py

Requires network access to https://download.moonshine.ai.
"""

import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_SRC = REPO_ROOT / "python" / "src"
OUTPUT_PATH = REPO_ROOT / "core" / "moonshine-model-file-metadata.generated.cpp"

sys.path.insert(0, str(PYTHON_SRC))

from moonshine_voice.moonshine_api import (  # noqa: E402
    moonshine_get_embedding_catalog_string,
    moonshine_get_embedding_dependencies_string,
    moonshine_get_stt_catalog_string,
    moonshine_get_stt_dependencies_string,
)


def enumerate_urls() -> "set[str]":
    """Every downloadable file URL across the whole catalog (STT + embedding)."""
    urls: set[str] = set()

    stt_catalog = json.loads(moonshine_get_stt_catalog_string())
    for language in stt_catalog.get("languages", []):
        code = language["code"]
        for model in language.get("models", []):
            arch = model["model_arch"]
            # include_spelling=True picks up the English spelling group too, and
            # word_timestamps=True adds the optional *_with_attention.ort
            # decoders. Both are supersets of the default manifest, so asking
            # for them here keeps integrity metadata for every file a caller
            # could ever request, not just the default download.
            manifest = json.loads(
                moonshine_get_stt_dependencies_string(
                    code,
                    {
                        "model_arch": arch,
                        "include_spelling": True,
                        "word_timestamps": True,
                    },
                )
            )
            for group in manifest.get("groups", []):
                for file_info in group.get("files", []):
                    urls.add(file_info["url"])

    embedding_catalog = json.loads(moonshine_get_embedding_catalog_string())
    for model in embedding_catalog.get("models", []):
        name = model["name"]
        for variant in model.get("variants", []):
            manifest = json.loads(
                moonshine_get_embedding_dependencies_string(name, {"variant": variant})
            )
            for group in manifest.get("groups", []):
                for file_info in group.get("files", []):
                    urls.add(file_info["url"])

    return urls


def head(url: str) -> "tuple[str, int, str]":
    """Returns (url, size, crc32c_base64) via an HTTP HEAD; raises on failure."""
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=30) as response:
        size = int(response.headers.get("content-length", "-1"))
        crc32c = ""
        for value in response.headers.get_all("x-goog-hash") or []:
            for part in value.split(","):
                part = part.strip()
                if part.startswith("crc32c="):
                    crc32c = part[len("crc32c="):]
    if size < 0:
        raise RuntimeError(f"no content-length for {url}")
    return url, size, crc32c


def cpp_string_literal(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def render(entries: "list[tuple[str, int, str]]") -> str:
    entries = sorted(entries, key=lambda e: e[0])
    lines = [
        "// GENERATED FILE - DO NOT EDIT BY HAND.",
        "//",
        "// Regenerate with:",
        "//   python3 scripts/generate-model-file-metadata.py",
        "//",
        "// This maps every downloadable model file's full CDN URL to its expected size",
        "// (bytes) and CRC32C checksum (base64, as reported by Google Cloud Storage).",
        "// It is the single source of truth for per-file integrity metadata that",
        "// moonshine-model-catalog.cpp joins into the download manifest.",
        "",
        '#include "moonshine-model-file-metadata.h"',
        "",
        "#include <algorithm>",
        "#include <array>",
        "#include <string_view>",
        "",
        "namespace moonshine {",
        "namespace {",
        "",
        "struct Entry {",
        "  std::string_view url;",
        "  int64_t size;",
        "  std::string_view checksum;",
        "  std::string_view checksum_type;",
        "};",
        "",
        "// Sorted by `url` (ascending) so lookups can binary-search.",
        f"constexpr std::array<Entry, {len(entries)}> kEntries = {{{{",
    ]
    for url, size, crc32c in entries:
        checksum_type = "crc32c" if crc32c else ""
        lines.append(
            "    {%s, %d, %s, %s},"
            % (
                cpp_string_literal(url),
                size,
                cpp_string_literal(crc32c),
                cpp_string_literal(checksum_type),
            )
        )
    lines.extend(
        [
            "}};",
            "",
            "}  // namespace",
            "",
            "ModelFileMetadata find_model_file_metadata(const std::string& url) {",
            "  const std::string_view key(url);",
            "  const auto* begin = kEntries.data();",
            "  const auto* end = begin + kEntries.size();",
            "  const auto* it = std::lower_bound(",
            "      begin, end, key,",
            "      [](const Entry& entry, std::string_view value) {",
            "        return entry.url < value;",
            "      });",
            "  if (it != end && it->url == key) {",
            "    return ModelFileMetadata{it->size, std::string(it->checksum),",
            "                             std::string(it->checksum_type)};",
            "  }",
            "  return ModelFileMetadata{-1, \"\", \"\"};",
            "}",
            "",
            "}  // namespace moonshine",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    print("Enumerating catalog file URLs via the moonshine C API...", file=sys.stderr)
    urls = sorted(enumerate_urls())
    print(f"Found {len(urls)} unique files; fetching metadata...", file=sys.stderr)

    entries: list[tuple[str, int, str]] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(head, url): url for url in urls}
        for future in futures:
            url = futures[future]
            try:
                entries.append(future.result())
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{url}: {exc}")

    if errors:
        print("\nFailed to fetch metadata for some files:", file=sys.stderr)
        for err in sorted(errors):
            print(f"  {err}", file=sys.stderr)
        print(
            "\nRefusing to write a partial registry. Fix the CDN uploads and retry.",
            file=sys.stderr,
        )
        return 1

    OUTPUT_PATH.write_text(render(entries))
    print(f"Wrote {len(entries)} entries to {OUTPUT_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
