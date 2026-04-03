# French — `fr`

## Contents

- **`dict.tsv`** — citation IPA lexicon (`word<TAB>ipa`).
- **`*.csv`** — part-of-speech word lists (`adj`, `adv`, `conj`, `det`, `noun`, `prep`, `pron`, `verb`, …) with a header containing **`form`**; used for liaison heuristics in C++ `FrenchRuleG2p` / Python `french_g2p`.

## Provenance

| Asset | Source |
|--------|--------|
| `dict.tsv` | [open-dict-data/ipa-dict](https://github.com/open-dict-data/ipa-dict) — `data/fr_FR.txt` (MIT). Fetched via `scripts/download_multilingual_ipa_lexicons.py`. |
| POS CSVs | Curated lists for liaison; the top-level [README.md](../../../README.md) acknowledges related lexicon work at [hbenbel/French-Dictionary](https://github.com/hbenbel/French-Dictionary). This repo does not ship an automated downloader for the CSV set—regeneration is “rebuild compatible CSVs” under the constraints of `french_g2p` / `FrenchRuleG2p` (see below). |

## Recreating

### Lexicon

```bash
python scripts/download_multilingual_ipa_lexicons.py --only fr
```

Produces `data/fr/dict.tsv` and `data/fr/source.txt`.

### POS CSVs

There is no single `scripts/download_*.py` for the French CSV bundle. To recreate or extend:

1. Each file must be UTF-8 CSV with a header line containing **`form`** (see `load_french_pos_dir` in `french_g2p.py` / `src/lang-specific/french.cpp`).
2. The filename stem (e.g. `noun` → category `NOUN`) is uppercased and used as the POS bucket key.
3. Align new lists with the liaison logic documented in `french_g2p.py` (determiners, pronouns, *h aspiré* blocks, etc.).

Copy updated files into `data/fr/` as needed.
