#!/usr/bin/env python3
"""List every CDN file the download manifests reference.

The set is enumerated from the compiled catalog through the C API rather than by
listing the CDN bucket, so it cannot drift from what the library actually
downloads. That distinction matters: the bucket also holds unconverted `.onnx`
sources, older copies of models the manifests no longer point at, and platform
artifacts like the Raspberry Pi image, none of which are shipped assets.

Every combination a caller could ask for is walked: each language and
architecture in the STT catalog (with `include_spelling` and `word_timestamps`
so the optional groups are covered), each embedding model and variant, the
diarization dependencies, and the TTS dependencies for each catalog language
paired with the voices available in that language.

This produced the inventory for the archival mirror at
https://huggingface.co/moonshine-ai/moonshine-voice-assets, so re-run it after
adding models to see what the mirror is missing.

Usage:
    # Build libmoonshine first (scripts/test-core.sh or a plain cmake build) so
    # the Python binding can load core/build/libmoonshine.dylib, then:
    python3 scripts/list-manifest-assets.py            # CDN-relative paths
    python3 scripts/list-manifest-assets.py --urls     # full download URLs
    python3 scripts/list-manifest-assets.py --by-model # grouped by what needs it
"""

import argparse
import json
import sys
import urllib.parse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_SRC = REPO_ROOT / "python" / "src"

sys.path.insert(0, str(PYTHON_SRC))

from moonshine_voice.download import (  # noqa: E402
    cdn_url_for_tts_asset_key,
    get_tts_voice_catalog,
    is_downloadable_tts_asset_key,
)
from moonshine_voice.moonshine_api import (  # noqa: E402
    moonshine_get_diarization_dependencies_string,
    moonshine_get_embedding_catalog_string,
    moonshine_get_embedding_dependencies_string,
    moonshine_get_stt_catalog_string,
    moonshine_get_stt_dependencies_string,
    moonshine_get_tts_dependencies_string,
)

CDN_PREFIX = "https://download.moonshine.ai/"


def manifest_urls(manifest_json):
    """Every file URL in a `{groups:[{files:[...]}]}` manifest."""
    urls = set()
    for group in json.loads(manifest_json).get("groups", []):
        for file_info in group.get("files", []):
            urls.add(file_info["url"])
    return urls


def collect():
    """Maps each referenced URL to the set of things that reference it."""
    refs = {}

    def add(url, reason):
        refs.setdefault(url, set()).add(reason)

    stt = json.loads(moonshine_get_stt_catalog_string())
    for language in stt.get("languages", []):
        code = language["code"]
        for model in language.get("models", []):
            arch = model["model_arch"]
            manifest = moonshine_get_stt_dependencies_string(
                code,
                {"model_arch": arch, "include_spelling": True, "word_timestamps": True},
            )
            for url in manifest_urls(manifest):
                add(url, f"stt {code} arch {arch}")

    embedding = json.loads(moonshine_get_embedding_catalog_string())
    for model in embedding.get("models", []):
        name = model["name"]
        for variant in model.get("variants", []):
            manifest = moonshine_get_embedding_dependencies_string(name, {"variant": variant})
            for url in manifest_urls(manifest):
                add(url, f"embedding {name} {variant}")

    for url in manifest_urls(moonshine_get_diarization_dependencies_string()):
        add(url, "diarization")

    # TTS returns flat canonical asset keys rather than a group manifest, and the
    # keys depend on the voice, so ask once per voice. Each language is asked only
    # about its own voices; the full cross product invents rejected combinations.
    for language, entries in sorted(get_tts_voice_catalog().items()):
        voices = sorted({getattr(e, "id", None) or e["id"] for e in entries})
        for voice in voices or [None]:
            options = {"voice": voice} if voice else None
            try:
                keys = json.loads(moonshine_get_tts_dependencies_string(language, options))
            except Exception as err:
                print(f"warning: tts {language}/{voice}: {err}", file=sys.stderr)
                continue
            for key in keys:
                if is_downloadable_tts_asset_key(key):
                    add(cdn_url_for_tts_asset_key(key), f"tts {language} {voice or 'default'}")

    return refs


def to_path(url):
    """CDN-relative path, with percent-escapes decoded to match object names."""
    if not url.startswith(CDN_PREFIX):
        return None
    return urllib.parse.unquote(url[len(CDN_PREFIX):])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urls", action="store_true", help="print full URLs instead of paths")
    parser.add_argument("--by-model", action="store_true", help="group by what references each file")
    args = parser.parse_args()

    refs = collect()

    external = sorted(u for u in refs if to_path(u) is None)
    for url in external:
        print(f"warning: referenced outside the CDN: {url}", file=sys.stderr)

    if args.by_model:
        by_reason = {}
        for url, reasons in refs.items():
            for reason in reasons:
                by_reason.setdefault(reason, []).append(url)
        for reason in sorted(by_reason):
            print(f"\n{reason}")
            for url in sorted(by_reason[reason]):
                print(f"  {url if args.urls else to_path(url)}")
        return 0

    for url in sorted(refs):
        value = url if args.urls else to_path(url)
        if value is not None:
            print(value)

    print(f"\n{len(refs)} files referenced", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
