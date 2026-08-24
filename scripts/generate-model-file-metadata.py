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

Watch out for a stale library. _load_library() checks the copy sitting in
language-bindings/python/src/moonshine_voice/ *before* core/build, and that copy
is a gitignored build artifact that nothing refreshes automatically. If it is
older than your catalog change, this script enumerates the old catalog and
cheerfully writes a registry with the new models missing. It prints the library
it loaded for exactly this reason; check the date, and copy the fresh build over
that file if it is behind.

Requires network access to https://download.moonshine.ai.
"""

import base64
import json
import os
import re
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import google_crc32c

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_SRC = REPO_ROOT / "language-bindings" / "python" / "src"
OUTPUT_PATH = REPO_ROOT / "core" / "moonshine-model-file-metadata.generated.cpp"

sys.path.insert(0, str(PYTHON_SRC))

# Cloudflare sits in front of download.moonshine.ai and answers urllib's default
# "Python-urllib/3.x" with 403 on every object, including ones curl fetches
# fine. Without a real User-Agent this script fails on the entire catalog and
# reads as "the CDN upload is broken".
REQUEST_HEADERS = {"User-Agent": "moonshine-metadata-generator/1.0 (+curl-like)"}

from moonshine_voice.moonshine_api import (  # noqa: E402
    moonshine_get_diarization_dependencies_string,
    moonshine_get_embedding_catalog_string,
    moonshine_get_embedding_dependencies_string,
    moonshine_get_stt_catalog_string,
    moonshine_get_stt_dependencies_string,
)


def enumerate_urls() -> "set[str]":
    """Every downloadable file URL across the catalog (STT, embedding, diarization)."""
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

    diarization = json.loads(moonshine_get_diarization_dependencies_string())
    for group in diarization.get("groups", []):
        for file_info in group.get("files", []):
            urls.add(file_info["url"])

    return urls


def existing_entries() -> "dict[str, tuple[int, str]]":
    """url -> (size, crc32c) parsed back out of the generated file.

    Used to avoid re-downloading objects that have not changed. The CDN
    directories this reads are write-once and pinned by the catalog, so a URL
    still serving the same number of bytes is still serving the same bytes.
    """
    if not OUTPUT_PATH.exists():
        return {}
    text = OUTPUT_PATH.read_text()
    # Entries are rendered as a C++ string literal (which the formatter is free
    # to split across lines), then a size, then the checksum literal.
    pattern = re.compile(
        r'\{((?:\s*"(?:[^"\\]|\\.)*")+)\s*,\s*(\d+)\s*,\s*'
        r'"((?:[^"\\]|\\.)*)"\s*,\s*"((?:[^"\\]|\\.)*)"\s*\}',
        re.S)
    out = {}
    for match in pattern.finditer(text):
        url = "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', match.group(1)))
        out[url] = (int(match.group(2)), match.group(3))
    return out


def head(url: str) -> "tuple[str, int, str]":
    """Returns (url, size, crc32c_base64) via an HTTP HEAD; raises on failure."""
    request = urllib.request.Request(url, method="HEAD", headers=REQUEST_HEADERS)
    with urllib.request.urlopen(request, timeout=30) as response:
        size = int(response.headers.get("content-length", "-1"))
        crc32c = ""
        # download.moonshine.ai used to be Google Cloud Storage, which reported
        # crc32c on every object. It is Cloudflare R2 now, and R2 sends an MD5
        # ETag instead, so this header is absent in practice and the checksum has
        # to be computed from the bytes. Kept because it costs nothing and is
        # free when true.
        for value in response.headers.get_all("x-goog-hash") or []:
            for part in value.split(","):
                part = part.strip()
                if part.startswith("crc32c="):
                    crc32c = part[len("crc32c="):]
    if size < 0:
        raise RuntimeError(f"no content-length for {url}")
    return url, size, crc32c


def download_crc32c(url: str, size: int) -> str:
    """Streams an object and returns its base64 crc32c.

    Only called for objects this run cannot account for otherwise. Before R2 the
    checksum came free in a HEAD, and losing it silently would have been the
    worst outcome available: `checksum_type` goes empty, every client falls back
    to checking the size alone, and a corrupted-but-right-length model download
    stops being detectable. Paying the bandwidth is the cheaper mistake.
    """
    checksum = google_crc32c.Checksum()
    request = urllib.request.Request(url, headers=REQUEST_HEADERS)
    with urllib.request.urlopen(request, timeout=300) as response:
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            checksum.update(block)
    return base64.b64encode(checksum.digest()).decode("ascii")


def metadata_for(url: str, cache: "dict[str, tuple[int, str]]") -> "tuple[str, int, str]":
    url, size, crc32c = head(url)
    if crc32c:
        return url, size, crc32c
    cached = cache.get(url)
    if cached and cached[0] == size and cached[1]:
        return url, size, cached[1]
    print(f"  computing crc32c for {url} ({size:,d} B)", file=sys.stderr)
    return url, size, download_crc32c(url, size)


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
        "// (bytes) and CRC32C checksum (base64).",
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
    # Report which library got loaded. See the note above: a stale copy shadows
    # the fresh build and produces a registry missing whatever you just added.
    from moonshine_voice.moonshine_api import _MoonshineLib

    lib_path = Path(_MoonshineLib()._lib._name)
    print(f"libmoonshine: {lib_path} "
          f"(built {datetime.fromtimestamp(lib_path.stat().st_mtime):%Y-%m-%d %H:%M})",
          file=sys.stderr)

    print("Enumerating catalog file URLs via the moonshine C API...", file=sys.stderr)
    urls = sorted(enumerate_urls())
    print(f"Found {len(urls)} unique files; fetching metadata...", file=sys.stderr)

    cache = existing_entries()
    print(f"Reusing checksums for unchanged files from {len(cache)} existing "
          "entries.", file=sys.stderr)

    entries: list[tuple[str, int, str]] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(metadata_for, url, cache): url for url in urls}
        for future in futures:
            url = futures[future]
            try:
                entries.append(future.result())
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{url}: {exc}")

    # A missing checksum is not a partial write, so it would not trip the check
    # below, but it silently downgrades that file to a size-only integrity check.
    # Worth naming rather than discovering later.
    unchecked = [url for url, _, crc in entries if not crc]
    if unchecked:
        print(f"\n{len(unchecked)} files have no checksum:", file=sys.stderr)
        for url in sorted(unchecked)[:10]:
            print(f"  {url}", file=sys.stderr)
        return 1

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
    # Without this the file re-wraps every long URL differently from the
    # committed copy, and adding one model shows up as a whole-file rewrite.
    try:
        subprocess.run(["clang-format", "-i", str(OUTPUT_PATH)], check=True)
    except (OSError, subprocess.CalledProcessError) as err:
        print(
            f"Wrote the registry but could not clang-format it ({err}); "
            "format it before committing.",
            file=sys.stderr,
        )
    print(f"Wrote {len(entries)} entries to {OUTPUT_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
