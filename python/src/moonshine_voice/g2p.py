"""Grapheme-to-phoneme (IPA) via the Moonshine C API."""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Union

from moonshine_voice.download import (
    download_g2p_assets,
    is_downloadable_tts_asset_key,
    list_g2p_dependency_keys,
    normalize_moonshine_language_tag,
)
from moonshine_voice.errors import MoonshineError
from moonshine_voice.moonshine_api import (
    MOONSHINE_HEADER_VERSION,
    _MoonshineLib,
    moonshine_c_string_array,
    moonshine_options_array,
    moonshine_text_to_phonemes_string,
)


class GraphemeToPhonemizer:
    """
    G2P / IPA conversion using Moonshine. Assets are listed by ``moonshine_get_g2p_dependencies``
    and downloaded from the same ``https://download.moonshine.ai/tts/`` tree as TTS lexicons.
    """

    def __init__(
        self,
        language: str,
        *,
        options: Optional[Dict[str, Union[str, int, float, bool]]] = None,
        asset_root: Optional[Path] = None,
        download: bool = True,
    ):
        self._language = normalize_moonshine_language_tag(language)
        self._extra_options = dict(options) if options else {}

        if download:
            self._asset_root = download_g2p_assets(
                self._language,
                options=self._extra_options,
                cache_root=Path(asset_root) if asset_root is not None else None,
            )
        else:
            if asset_root is None:
                raise MoonshineError(
                    "When download=False, asset_root must point to a directory "
                    "already populated with G2P assets."
                )
            self._asset_root = Path(asset_root).resolve()

        self._lib = _MoonshineLib().lib
        create_opts = dict(self._extra_options)
        create_opts["g2p_root"] = str(self._asset_root)
        opt_arr, opt_n, _ok = moonshine_options_array(create_opts)

        keys = [
            k
            for k in list_g2p_dependency_keys(
                self._language, options=self._extra_options
            )
            if is_downloadable_tts_asset_key(k)
        ]
        fn_arr, fn_n, _fk = moonshine_c_string_array(keys)
        lang_b = self._language.encode("utf-8")
        handle = self._lib.moonshine_create_grapheme_to_phonemizer_from_files(
            lang_b,
            fn_arr,
            fn_n,
            opt_arr,
            opt_n,
            MOONSHINE_HEADER_VERSION,
        )
        if handle < 0:
            msg = self._lib.moonshine_error_to_string(handle)
            raise MoonshineError(
                msg.decode("utf-8") if msg else f"Failed to create G2P ({handle})"
            )
        self._handle = handle

    @property
    def language(self) -> str:
        return self._language

    @property
    def asset_root(self) -> Path:
        return self._asset_root

    def to_ipa(
        self,
        text: str,
        options: Optional[Dict[str, Union[str, int, float, bool]]] = None,
    ) -> str:
        """Return IPA for ``text`` (single string from the native layer)."""
        return moonshine_text_to_phonemes_string(self._handle, text, options)

    def close(self) -> None:
        if getattr(self, "_handle", None) is not None:
            self._lib.moonshine_free_grapheme_to_phonemizer(self._handle)
            self._handle = None

    def __enter__(self) -> "GraphemeToPhonemizer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print IPA for text using Moonshine G2P (downloads assets by default).",
    )
    parser.add_argument(
        "-l",
        "--language",
        required=True,
        help="Moonshine language tag (e.g. en_us, ar_msa, cmn_hans_cn)",
    )
    parser.add_argument(
        "--text",
        help="Input text to convert (quote if it contains spaces)",
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=None,
        help="Cache directory for G2P assets (default: Moonshine download cache)",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Use only existing files under --asset-root (no network)",
    )
    args = parser.parse_args()

    try:
        with GraphemeToPhonemizer(
            args.language,
            asset_root=args.asset_root,
            download=not args.no_download,
        ) as g2p:
            ipa = g2p.to_ipa(args.text)
    except MoonshineError as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(1) from e

    print(ipa)


if __name__ == "__main__":
    main()
