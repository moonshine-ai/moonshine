# Hindi — `hi`

## Contents

- **`dict.tsv`** — Devanagari surface → IPA (tab-separated). Used by C++ `HindiRuleG2p` **before** the rule-based syllable parser: on a full-token NFC match, the lexicon IPA is returned; otherwise the engine falls back to the same heuristics as `hindi_rule_g2p.py`.

No other data files are required for Hindi in C++ (digit expansion and rules are implemented in `hindi-numbers.cpp` / `hindi.cpp`).

## Provenance

| Field | Detail |
|--------|--------|
| **Primary copy** | Repository root `data/hi/dict.tsv` (canonical, in the parent monorepo when embedded). This tree mirrors it for layouts where `model_root` points at this checkout's `data/` directory. |
| **IPA source** | English Wiktionary, parsed HTML (`== Hindi ==` section, phonemic `/…/` in IPA spans). **License:** [CC BY-SA](https://en.wiktionary.org/wiki/Wiktionary:Copyrights). |
| **Word ranking** | [hermitdave/FrequencyWords](https://github.com/hermitdave/FrequencyWords) Hindi list (`content/2018/hi/hi_full.txt`, OpenSubtitles-derived). **License:** CC BY-SA 4.0. |
| **Build script** | `scripts/build_hi_lexicon_from_wiktionary.py` |

Comment headers inside `dict.tsv` repeat the URLs and licenses for the merged Wiktionary + frequency-list block.

## Path resolution (C++)

- **`MoonshineG2P` / factory:** `resolve_hindi_dict_path(model_root)` checks, in order:  
  `model_root/../data/hi/dict.tsv`, `model_root/../../data/hi/dict.tsv`, then `model_root/hi/dict.tsv`.
- **`hindi_text_to_ipa` / doctest parity:** `builtin_hindi_dict_path()` prefers repo `data/hi/dict.tsv`, else `data/hi/dict.tsv` (paths derived from `hindi.cpp`).
- **`hindi_rule_g2p` standalone tool:** same default as `builtin_hindi_dict_path()`; override with `--dict PATH`.

## Recreating / refreshing

From the repository root (network required; be polite with `--delay`):

```bash
# Example: top 1000 tokens from the default frequency URL → merge into the canonical lexicon
python scripts/build_hi_lexicon_from_wiktionary.py --freq-url DEFAULT --limit 1000 --merge data/hi/dict.tsv
```

Then refresh the C++ mirror:

```bash
cp data/hi/dict.tsv data/hi/dict.tsv
```

Category-wide crawl (no frequency list) remains:

```bash
python scripts/build_hi_lexicon_from_wiktionary.py --limit 500 --out data/hi/dict_wiki.tsv
```

Use `--out` for a new file or `--merge data/hi/dict.tsv` to append missing keys (see script docstring).

## Python parity

`hindi_rule_g2p.py` uses `_DEFAULT_HI_DICT` → `data/hi/dict.tsv`. After updating either copy, keep them aligned with the `cp` step above so C++ and Python stay consistent.
