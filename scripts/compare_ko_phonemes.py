#!/usr/bin/env python3
"""
Compare Moonshine vs Piper Korean phonemes directly (no TTS/Whisper needed).

Extracts phonemes from both systems for a list of Korean words/phrases and
compares them character-by-character to find systematic differences.

Usage:
    python scripts/compare_ko_phonemes.py
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "python" / "src"))

from moonshine_voice.download import download_tts_assets, normalize_moonshine_language_tag
from moonshine_voice.g2p import GraphemeToPhonemizer


def get_piper_phonemes(text: str, piper_voice) -> str:
    """Get phonemes from Piper's eSpeak phonemizer."""
    try:
        from piper.phonemize_espeak import EspeakPhonemizer
        phonemizer = EspeakPhonemizer()
        clauses = phonemizer.phonemize("ko", text)
        parts = []
        for clause in clauses:
            parts.append("".join(clause))
        return " ".join(parts).strip()
    except Exception as e:
        return f"ERROR: {e}"


def levenshtein_alignment(s: str, t: str):
    """Return (distance, list of (op, s_char, t_char)) edit operations."""
    n, m = len(s), len(t)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    # Backtrace
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if s[i - 1] == t[j - 1] else 1):
            if s[i - 1] == t[j - 1]:
                ops.append(("match", s[i - 1], t[j - 1]))
            else:
                ops.append(("sub", s[i - 1], t[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", s[i - 1], ""))
            i -= 1
        else:
            ops.append(("ins", "", t[j - 1]))
            j -= 1
    ops.reverse()
    return dp[n][m], ops


def classify_char(ch):
    vowels = set("aeiouɐɛɯʌɔɪ")
    glides = set("wj")
    stress = set("ˈˌ")
    if ch in stress: return "stress"
    if ch in vowels: return "vowel"
    if ch in glides: return "glide"
    if ch in " ": return "space"
    if ch in "-.?,!": return "punct"
    return "consonant"


def main():
    # Korean conversational words - extracted from ko_conversational.txt
    text_path = _REPO_ROOT / "scripts" / "data" / "wiki-text" / "ko_conversational.txt"
    if text_path.is_file():
        sentences = [l.strip() for l in text_path.read_text("utf-8").splitlines() if l.strip()]
    else:
        sentences = [
            "오늘 날씨가 정말 좋아서 산책하기 딱 좋은 것 같아요",
            "어제 친구랑 같이 영화를 봤는데 생각보다 재미있었어요",
            "커피 한 잔 마시면서 이야기할까요",
        ]

    # Extract unique words
    words = []
    for s in sentences:
        for w in s.split():
            w = w.strip("?!.,")
            if w and any('\uAC00' <= c <= '\uD7A3' for c in w):
                words.append(w)
    unique_words = list(dict.fromkeys(words))  # preserve order, dedupe
    print(f"Extracted {len(unique_words)} unique Korean words from {len(sentences)} sentences\n")

    # Initialize Moonshine G2P
    lang = "ko"
    print("Downloading/loading Moonshine G2P assets...")
    asset_root = download_tts_assets(lang, voice="piper_ko_KR-melotts-medium", show_progress=True)

    # Overlay repo dict.tsv
    repo_dict = _REPO_ROOT / "core" / "moonshine-tts" / "data" / "ko" / "dict.tsv"
    if repo_dict.is_file():
        import shutil
        dst = asset_root / "ko" / "dict.tsv"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo_dict, dst)
        print(f"Overlayed {repo_dict} -> {dst}")

    g2p = GraphemeToPhonemizer(lang, asset_root=asset_root, download=False)

    # Initialize Piper phonemizer
    piper_voice = None
    try:
        from piper import PiperVoice
        onnx_matches = sorted(asset_root.resolve().rglob("ko_KR-melotts-medium.onnx"))
        if onnx_matches:
            piper_voice = PiperVoice.load(onnx_matches[0])
            # Patch to use NFC
            from piper.phonemize_espeak import EspeakPhonemizer
            original_phonemize = EspeakPhonemizer.phonemize

            def phonemize_nfc(self, voice, text):
                from piper import espeakbridge
                espeakbridge.set_voice(voice)
                all_phonemes = []
                sentence_phonemes = []
                clause_phonemes = espeakbridge.get_phonemes(text)
                for phonemes_str, terminator_str, end_of_sentence in clause_phonemes:
                    phonemes_str = re.sub(r"\([^)]+\)", "", phonemes_str)
                    phonemes_str += terminator_str
                    if terminator_str in (",", ":", ";"):
                        phonemes_str += " "
                    sentence_phonemes.extend(list(unicodedata.normalize("NFC", phonemes_str)))
                    if end_of_sentence:
                        all_phonemes.append(sentence_phonemes)
                        sentence_phonemes = []
                if sentence_phonemes:
                    all_phonemes.append(sentence_phonemes)
                return all_phonemes

            EspeakPhonemizer.phonemize = phonemize_nfc
            print("Piper phonemizer loaded successfully")
    except ImportError:
        print("WARNING: piper-tts not installed, skipping Piper comparison")

    # Compare phonemes word by word
    print(f"\n{'='*80}")
    print("WORD-BY-WORD PHONEME COMPARISON")
    print(f"{'='*80}\n")

    all_subs = Counter()
    all_dels = Counter()
    all_ins = Counter()
    results = []
    total_lev = 0

    for word in unique_words:
        moon_ipa = g2p.to_ipa(word).strip()

        if piper_voice is not None:
            try:
                from piper.phonemize_espeak import EspeakPhonemizer
                ph = EspeakPhonemizer()
                clauses = ph.phonemize("ko", word)
                piper_ipa = "".join("".join(c) for c in clauses).strip()
            except Exception:
                piper_ipa = ""
        else:
            piper_ipa = ""

        if not piper_ipa or not moon_ipa:
            continue

        lev, ops = levenshtein_alignment(moon_ipa, piper_ipa)
        total_lev += lev
        results.append((word, moon_ipa, piper_ipa, lev))

        for op, mc, pc in ops:
            if op == "sub":
                all_subs[(mc, pc)] += 1
            elif op == "del":
                all_dels[mc] += 1
            elif op == "ins":
                all_ins[pc] += 1

    # Sort by Levenshtein distance
    results.sort(key=lambda x: -x[3])

    # Print all words
    for word, moon, piper, lev in results:
        marker = " ***" if lev >= 3 else ""
        print(f"  {word:12s}  lev={lev:2d}  moon={moon:30s}  piper={piper}{marker}")

    g2p.close()

    # Analysis
    print(f"\n{'='*80}")
    print("SUBSTITUTION PATTERNS (Moonshine -> Piper)")
    print(f"{'='*80}")
    for (mc, pc), count in all_subs.most_common(25):
        print(f"  {mc!r:8s} -> {pc!r:8s}  count={count:3d}  ({classify_char(mc)}->{classify_char(pc)})")

    print(f"\n{'='*80}")
    print("DELETIONS (in Moonshine but not Piper)")
    print(f"{'='*80}")
    for ch, count in all_dels.most_common(15):
        print(f"  {ch!r:8s}  count={count:3d}  ({classify_char(ch)})")

    print(f"\n{'='*80}")
    print("INSERTIONS (in Piper but not Moonshine)")
    print(f"{'='*80}")
    for ch, count in all_ins.most_common(15):
        print(f"  {ch!r:8s}  count={count:3d}  ({classify_char(ch)})")

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    total_ops = sum(all_subs.values()) + sum(all_dels.values()) + sum(all_ins.values())
    print(f"  Total words compared: {len(results)}")
    print(f"  Total Levenshtein: {total_lev}")
    print(f"  Avg Levenshtein/word: {total_lev/len(results):.2f}" if results else "  N/A")
    print(f"  Total edit ops: {total_ops}")
    print(f"    Substitutions: {sum(all_subs.values())}")
    print(f"    Deletions: {sum(all_dels.values())}")
    print(f"    Insertions: {sum(all_ins.values())}")

    # Top actionable patterns
    print(f"\n  TOP ACTIONABLE PATTERNS:")
    for (mc, pc), count in all_subs.most_common(5):
        print(f"    {mc!r} -> {pc!r}: {count} occurrences")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
