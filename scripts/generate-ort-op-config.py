#!/usr/bin/env python3
"""Regenerates the operator config that the minimal ORT-wasm build is cut from.

The wasm runtime is built with ``--include_ops_by_config``, which drops every
kernel no shipped model needs. That is only safe if the config covers every
model a client can actually load, so this script enumerates them rather than
taking a hand-maintained list:

  * TTS assets are whatever ``core/moonshine-tts/data`` holds, since
    ``scripts/upload-tts-assets-to-gcs.sh`` mirrors that tree to the CDN.
  * STT, spelling and embedding models come from the native catalog (via the
    ``moonshine_voice`` bindings), covering every language, architecture and
    variant it can hand out. Those are fetched and cached under
    ``--cache-dir``; the download runs to several GB the first time.

Run it after adding a model, changing a model's ops, or converting one to a new
form, then commit the result. ``scripts/check-ort-op-config.sh`` fails CI when
the committed config no longer matches what this produces.

Usage:
  scripts/generate-ort-op-config.py [--cache-dir DIR] [--offline] [--check]
"""

import argparse
import json
import pathlib
import re
import sys
import tempfile
import urllib.request

REPO = pathlib.Path(__file__).resolve().parent.parent
TTS_DATA = REPO / "core" / "moonshine-tts" / "data"
OUTPUT = REPO / "core" / "third-party" / "onnxruntime" / (
    "moonshine-required-operators.config"
)
DEFAULT_CACHE = pathlib.Path.home() / ".cache" / "moonshine-ort-op-config"


def local_tts_models():
    """Every ORT model in the TTS tree, which is what the CDN mirrors."""
    return sorted(TTS_DATA.rglob("*.ort"))


# Sources holding a model as a C array. These are compiled straight into the
# library, so they never appear as a file or a download, but the runtime still
# has to have kernels for them.
EMBEDDED_SOURCES = [
    REPO / "core" / "silero-vad-model-data.h",
    REPO / "core" / "cpp-annote" / "src" / "community1_ort_embedded.cpp",
]

ARRAY_PATTERN = re.compile(
    r"unsigned\s+char\s+(\w+)\s*\[\]\s*=\s*\{(.*?)\}\s*;", re.DOTALL
)


def embedded_models(cache_dir):
    """Extracts every ORT model compiled in as a C array.

    Detected by the ORT file magic rather than by symbol name, so a model added
    to one of these generated sources is picked up without touching this list.
    """
    found = []
    out_dir = cache_dir / "embedded"
    out_dir.mkdir(parents=True, exist_ok=True)
    for source in EMBEDDED_SOURCES:
        if not source.is_file():
            continue
        text = source.read_text()
        for name, body in ARRAY_PATTERN.findall(text):
            digits = body.replace("0x", "").replace(",", "")
            data = bytes.fromhex("".join(digits.split()))
            # ORT flatbuffers carry "ORTM" as the file identifier at offset 4.
            if data[4:8] != b"ORTM":
                continue
            target = out_dir / f"{name}.ort"
            if not target.is_file() or target.read_bytes() != data:
                target.write_bytes(data)
            found.append((target, f"{source.relative_to(REPO)}:{name}"))
    return found


def catalog_model_urls():
    """Every ``.ort`` URL the native catalog can hand a client."""
    sys.path.insert(0, str(REPO / "python" / "src"))
    from moonshine_voice.download import (  # noqa: E402
        _embedding_catalog,
        _stt_catalog,
        get_embedding_model_variants,
    )
    from moonshine_voice.moonshine_api import (  # noqa: E402
        moonshine_get_embedding_dependencies_string,
        moonshine_get_stt_dependencies_string,
    )

    urls = set()

    def collect(raw):
        for group in json.loads(raw).get("groups", []):
            for entry in group.get("files", []):
                url = entry.get("url", "")
                if url.endswith(".ort"):
                    urls.add(url)

    for code, info in _stt_catalog().items():
        for model in info["models"]:
            # Spelling and word-timestamp decoders are optional downloads, so
            # ask for both: a client that enables them still has to load them.
            collect(
                moonshine_get_stt_dependencies_string(
                    code,
                    {
                        "model_arch": int(model["model_arch"]),
                        "include_spelling": True,
                        "word_timestamps": True,
                    },
                )
            )

    for name in _embedding_catalog():
        for variant in get_embedding_model_variants(name):
            collect(
                moonshine_get_embedding_dependencies_string(name, {"variant": variant})
            )

    return sorted(urls)


def fetch(url, cache_dir, offline=False):
    """Returns the cached copy of *url*, downloading it when absent.

    Returns None in offline mode when the model is not already cached.
    """
    # The URL path is unique per model, so it doubles as the cache layout.
    relative = url.split("://", 1)[1]
    target = cache_dir / relative
    if target.is_file() and target.stat().st_size > 0:
        return target
    if offline:
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetching {url}", flush=True)
    partial = target.with_suffix(target.suffix + ".partial")
    with urllib.request.urlopen(url) as response, open(partial, "wb") as out:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    partial.rename(target)
    return target


def build_config(models, destination, names):
    from onnxruntime.tools.ort_format_model import utils

    utils.create_config_from_models(models, destination, enable_type_reduction=False)
    return normalized(destination.read_text(), names)


def normalized(text, names):
    """Replaces the generator's absolute paths with stable names.

    The tool lists the models it read, using whatever path they happened to have
    on the machine that ran it. Those differ per checkout and per download
    cache, which would make the committed file look stale everywhere but here.
    """
    body = [line for line in text.splitlines() if not line.startswith("#")]
    header = [
        "# Operators required by every model Moonshine can load.",
        "# Generated by scripts/generate-ort-op-config.py; do not edit by hand.",
        "# Consumed by scripts/build-ort-wasm.sh via --include_ops_by_config.",
        "#",
        "# Models:",
    ]
    header += [f"# - {name}" for name in sorted(names)]
    return "\n".join(header + body) + "\n"


def operators(text):
    """The ``(domain, opset, operator)`` triples a config requires."""
    required = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ";" not in line:
            continue
        domain, _, rest = line.partition(";")
        opset, _, names = rest.partition(";")
        for name in names.split(","):
            if name.strip():
                required.add((domain.strip(), opset.strip(), name.strip()))
    return required


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=pathlib.Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use only the TTS tree and already-cached catalog models.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail instead of writing when the committed config is stale.",
    )
    args = parser.parse_args()

    models = local_tts_models()
    names = [str(path.relative_to(REPO)) for path in models]
    print(f"{len(models)} ORT models in {TTS_DATA.relative_to(REPO)}")

    embedded = embedded_models(args.cache_dir)
    for path, name in embedded:
        models.append(path)
        names.append(name)
    print(f"{len(embedded)} ORT models embedded in the library")

    urls = catalog_model_urls()
    print(f"{len(urls)} ORT models in the native catalog")
    missing = 0
    for url in urls:
        try:
            cached = fetch(url, args.cache_dir, offline=args.offline)
        except Exception as error:  # noqa: BLE001 - reported, not swallowed
            print(f"error: could not fetch {url}: {error}", file=sys.stderr)
            return 1
        if cached is None:
            missing += 1
            continue
        models.append(cached)
        names.append(url)
    if missing:
        print(f"warning: skipped {missing} uncached catalog models (--offline)")

    print(f"generating config from {len(models)} models")
    with tempfile.TemporaryDirectory() as work:
        generated = pathlib.Path(work) / "required_operators.config"
        text = build_config(models, generated, names)
    required = operators(text)

    if args.check:
        # Coverage, not equality: an offline run sees fewer models than the
        # committed config was built from, and a config listing an operator no
        # model needs only wastes a little binary size. What must never happen
        # is a model needing an operator the build left out, because that fails
        # at session creation in the browser.
        if not OUTPUT.is_file():
            print(f"error: {OUTPUT.relative_to(REPO)} is missing", file=sys.stderr)
            return 1
        uncovered = sorted(required - operators(OUTPUT.read_text()))
        if uncovered:
            print(
                f"error: {len(uncovered)} operator(s) needed by a model are "
                f"absent from {OUTPUT.relative_to(REPO)}:",
                file=sys.stderr,
            )
            for domain, opset, name in uncovered:
                print(f"  {domain} opset {opset}: {name}", file=sys.stderr)
            print(
                "Regenerate with scripts/generate-ort-op-config.py and rebuild "
                "the wasm archive with scripts/build-ort-wasm.sh force.",
                file=sys.stderr,
            )
            return 1
        print(
            f"{OUTPUT.relative_to(REPO)} covers all {len(required)} operators "
            f"required by {len(models)} models"
        )
        return 0

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(text)
    print(f"wrote {OUTPUT.relative_to(REPO)} ({len(required)} operators)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
