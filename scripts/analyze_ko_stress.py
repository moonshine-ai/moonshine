#!/usr/bin/env python3
"""
Deep analysis of Korean stress placement: Moonshine vs Piper/eSpeak.

Generates a large word set, extracts phonemes from both systems, and analyzes
where primary (ˈ) and secondary (ˌ) stress markers are placed relative to
syllable structure.

Usage:
    python scripts/analyze_ko_stress.py
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

from moonshine_voice.download import download_tts_assets
from moonshine_voice.g2p import GraphemeToPhonemizer


# Common Korean words organized by syllable count for systematic testing
WORDS_BY_SYLLABLE_COUNT = {
    1: [
        "나", "너", "그", "이", "것", "집", "밤", "말", "눈", "손",
        "달", "산", "강", "바", "해", "꽃", "물", "길", "비", "불",
    ],
    2: [
        "사람", "학교", "친구", "오늘", "내일", "우리", "서울", "한국",
        "시간", "공부", "여름", "겨울", "아침", "저녁", "음식", "운동",
        "가족", "회사", "버스", "영화", "이름", "생각", "의자", "나무",
        "바다", "도시", "지금", "항상", "정말", "아직",
    ],
    3: [
        "한국어", "대학교", "도서관", "공원에", "서울에", "아이들",
        "컴퓨터", "선생님", "학생들", "시작해", "전화기", "여행을",
        "주말에", "일요일", "토요일", "월요일", "감사해", "미안해",
        "사랑해", "행복해", "건강해", "가능해", "필요해", "중요해",
        "공항에", "병원에", "식당에", "커피를", "음악을", "문제가",
    ],
    4: [
        "대한민국", "이야기해", "재미있어", "어려워요", "기다려요",
        "도서관에", "만족해요", "공부해요", "준비해요", "시작해요",
        "걱정해요", "노력해요", "생각해요", "부지런히", "행복해요",
        "감사해요", "미안해요", "사랑해요", "필요해요", "중요해요",
        "가능해요", "불가능해", "지하철이", "유치원에", "마시면서",
    ],
    5: [
        "재미있었어요", "어려웠어요", "공부하고있어", "이야기할까요",
        "도서관에서", "대학교에서", "기다리고있어", "감사합니다",
        "미안합니다", "필요합니다", "안녕하세요", "만나서반가워",
    ],
}

# Verb conjugation patterns (same stem, different endings)
VERB_CONJUGATIONS = {
    "가다": ["가다", "가요", "갔어요", "가고", "가면", "가는", "갈까요", "가세요", "갑니다"],
    "먹다": ["먹다", "먹어요", "먹었어요", "먹고", "먹으면", "먹는", "먹을까요", "먹으러"],
    "하다": ["하다", "해요", "했어요", "하고", "하면", "하는", "할까요", "하세요", "합니다"],
    "보다": ["보다", "봐요", "봤어요", "보고", "보면", "보는", "볼까요", "보세요"],
    "오다": ["오다", "와요", "왔어요", "오고", "오면", "오는", "올까요", "오세요"],
    "주다": ["주다", "줘요", "줬어요", "주고", "주면", "주는", "줄까요", "주세요"],
    "알다": ["알다", "알아요", "알았어요", "알고", "알면", "아는"],
    "좋다": ["좋다", "좋아요", "좋았어요", "좋고", "좋은", "좋겠어요"],
    "있다": ["있다", "있어요", "있었어요", "있고", "있으면", "있는"],
    "없다": ["없다", "없어요", "없었어요", "없고", "없으면", "없는"],
}


def count_hangul_syllables(word: str) -> int:
    """Count Hangul syllable characters in a word."""
    return sum(1 for c in word if '\uAC00' <= c <= '\uD7A3')


def extract_stress_positions(ipa: str):
    """
    Extract the character positions of stress markers relative to the IPA string.
    Returns list of (marker, position_in_string, preceding_consonants, following_vowel).
    """
    markers = []
    for i, ch in enumerate(ipa):
        if ch in ('ˈ', 'ˌ'):
            # What's before (consonant onset) and after (vowel nucleus)
            before = ipa[:i]
            after = ipa[i+1:i+4] if i+1 < len(ipa) else ""
            markers.append({
                "marker": ch,
                "pos": i,
                "total_len": len(ipa),
                "rel_pos": i / len(ipa) if len(ipa) > 0 else 0,
                "before": before[-3:],  # last 3 chars before stress
                "after": after,          # first 3 chars after stress
            })
    return markers


def extract_stress_pattern(ipa: str, num_syllables: int) -> str:
    """
    Create a simplified stress pattern string.
    e.g., "1-0-2-0" means: primary on syl 1, none on syl 2, secondary on syl 3, none on syl 4.
    """
    # Find stress marker positions as fraction of total string length
    primary_pos = ipa.find('ˈ')
    secondary_positions = []
    pos = 0
    while True:
        pos = ipa.find('ˌ', pos)
        if pos == -1:
            break
        secondary_positions.append(pos)
        pos += 1

    if primary_pos == -1:
        return "none"

    # Estimate which syllable each stress falls on based on position
    total = len(ipa)
    if total == 0:
        return "none"

    primary_syl = int(primary_pos / total * num_syllables)
    pattern = ["0"] * num_syllables
    if primary_syl < num_syllables:
        pattern[primary_syl] = "1"
    for sp in secondary_positions:
        sec_syl = int(sp / total * num_syllables)
        if sec_syl < num_syllables and pattern[sec_syl] == "0":
            pattern[sec_syl] = "2"

    return "-".join(pattern)


def analyze_stress_position_type(ipa: str) -> str:
    """
    Classify where primary stress ˈ appears relative to the word start.
    Returns: 'pos0' (very start), 'after_onset' (after 1-2 consonants), 'later', 'none'
    """
    pos = ipa.find('ˈ')
    if pos == -1:
        return "none"
    if pos == 0:
        return "pos0"
    # Check if everything before ˈ is consonantal onset
    prefix = ipa[:pos]
    vowels = set("aeiouɐɛɯʌɔɪ")
    if any(c in vowels for c in prefix):
        return "later"  # stress is after a vowel = not on first syllable
    return "after_onset"  # consonant(s) then ˈ = stress on first syllable after onset


def setup_piper_phonemizer():
    """Set up Piper's eSpeak phonemizer with NFC patch."""
    try:
        from piper.phonemize_espeak import EspeakPhonemizer

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
        return EspeakPhonemizer()
    except ImportError:
        return None


def get_piper_ipa(phonemizer, word: str) -> str:
    if phonemizer is None:
        return ""
    try:
        clauses = phonemizer.phonemize("ko", word)
        return "".join("".join(c) for c in clauses).strip()
    except Exception:
        return ""


def main():
    # Build complete word list
    all_words = []
    for n, words in sorted(WORDS_BY_SYLLABLE_COUNT.items()):
        all_words.extend(words)
    for stem, forms in VERB_CONJUGATIONS.items():
        all_words.extend(forms)

    # Deduplicate preserving order
    seen = set()
    unique_words = []
    for w in all_words:
        if w not in seen:
            seen.add(w)
            unique_words.append(w)

    print(f"Testing {len(unique_words)} unique Korean words\n")

    # Initialize systems
    print("Loading Moonshine G2P...")
    import shutil
    root = download_tts_assets("ko", voice="piper_ko_KR-melotts-medium", show_progress=False)
    repo_dict = _REPO_ROOT / "core" / "moonshine-tts" / "data" / "ko" / "dict.tsv"
    if repo_dict.is_file():
        shutil.copy2(repo_dict, root / "ko" / "dict.tsv")
    g2p = GraphemeToPhonemizer("ko", asset_root=root, download=False)

    print("Loading Piper phonemizer...")
    piper_ph = setup_piper_phonemizer()

    # Collect data
    results = []
    for word in unique_words:
        moon = g2p.to_ipa(word).strip()
        piper = get_piper_ipa(piper_ph, word)
        if not moon or not piper:
            continue
        nsyl = count_hangul_syllables(word)
        results.append({
            "word": word,
            "nsyl": nsyl,
            "moon": moon,
            "piper": piper,
            "moon_stress_type": analyze_stress_position_type(moon),
            "piper_stress_type": analyze_stress_position_type(piper),
            "moon_pattern": extract_stress_pattern(moon, nsyl),
            "piper_pattern": extract_stress_pattern(piper, nsyl),
            "moon_markers": extract_stress_positions(moon),
            "piper_markers": extract_stress_positions(piper),
        })

    g2p.close()

    # ==========================================
    # Analysis 1: Stress position type comparison
    # ==========================================
    print(f"\n{'='*80}")
    print("ANALYSIS 1: WHERE DOES PRIMARY STRESS FALL?")
    print(f"{'='*80}")

    moon_types = Counter(r["moon_stress_type"] for r in results)
    piper_types = Counter(r["piper_stress_type"] for r in results)

    print(f"\n  Position type      Moonshine   Piper")
    print(f"  ──────────────     ─────────   ─────")
    for t in ["pos0", "after_onset", "later", "none"]:
        print(f"  {t:17s}   {moon_types[t]:5d}       {piper_types[t]:5d}")

    # ==========================================
    # Analysis 2: Agreement rate by syllable count
    # ==========================================
    print(f"\n{'='*80}")
    print("ANALYSIS 2: STRESS PATTERN AGREEMENT BY SYLLABLE COUNT")
    print(f"{'='*80}")

    for nsyl in sorted(set(r["nsyl"] for r in results)):
        subset = [r for r in results if r["nsyl"] == nsyl]
        agree = sum(1 for r in subset if r["moon_pattern"] == r["piper_pattern"])
        print(f"\n  {nsyl}-syllable words: {len(subset)} total, {agree} agree ({100*agree/len(subset):.0f}%)")

        # Show the pattern distribution
        moon_patterns = Counter(r["moon_pattern"] for r in subset)
        piper_patterns = Counter(r["piper_pattern"] for r in subset)
        all_patterns = sorted(set(list(moon_patterns.keys()) + list(piper_patterns.keys())))
        if len(all_patterns) <= 10:
            print(f"    {'Pattern':15s}  Moon  Piper")
            for p in all_patterns:
                print(f"    {p:15s}  {moon_patterns[p]:4d}  {piper_patterns[p]:4d}")

    # ==========================================
    # Analysis 3: Specific disagreements
    # ==========================================
    print(f"\n{'='*80}")
    print("ANALYSIS 3: STRESS PLACEMENT DISAGREEMENTS (sample)")
    print(f"{'='*80}")

    disagree = [r for r in results if r["moon_pattern"] != r["piper_pattern"]]
    print(f"\n  Total disagreements: {len(disagree)} of {len(results)} ({100*len(disagree)/len(results):.0f}%)")

    # Group by syllable count
    for nsyl in sorted(set(r["nsyl"] for r in disagree)):
        subset = [r for r in disagree if r["nsyl"] == nsyl]
        print(f"\n  --- {nsyl}-syllable disagreements ({len(subset)} words) ---")
        for r in subset[:8]:
            print(f"    {r['word']:12s}  moon={r['moon']:30s}  piper={r['piper']}")
            print(f"    {'':12s}  pattern: {r['moon_pattern']:10s}  vs  {r['piper_pattern']}")

    # ==========================================
    # Analysis 4: Primary stress position (character index)
    # ==========================================
    print(f"\n{'='*80}")
    print("ANALYSIS 4: PRIMARY STRESS RELATIVE POSITION")
    print(f"{'='*80}")

    for nsyl in sorted(set(r["nsyl"] for r in results)):
        subset = [r for r in results if r["nsyl"] == nsyl]
        moon_positions = []
        piper_positions = []
        for r in subset:
            for m in r["moon_markers"]:
                if m["marker"] == "ˈ":
                    moon_positions.append(m["rel_pos"])
            for m in r["piper_markers"]:
                if m["marker"] == "ˈ":
                    piper_positions.append(m["rel_pos"])

        if moon_positions and piper_positions:
            moon_avg = sum(moon_positions) / len(moon_positions)
            piper_avg = sum(piper_positions) / len(piper_positions)
            print(f"\n  {nsyl}-syl: Moonshine avg primary stress at {moon_avg:.2f}, Piper at {piper_avg:.2f}")

    # ==========================================
    # Analysis 5: Secondary stress patterns
    # ==========================================
    print(f"\n{'='*80}")
    print("ANALYSIS 5: SECONDARY STRESS PRESENCE")
    print(f"{'='*80}")

    for nsyl in sorted(set(r["nsyl"] for r in results)):
        subset = [r for r in results if r["nsyl"] == nsyl]
        moon_has_sec = sum(1 for r in subset if 'ˌ' in r["moon"])
        piper_has_sec = sum(1 for r in subset if 'ˌ' in r["piper"])
        print(f"  {nsyl}-syl: Moonshine {moon_has_sec}/{len(subset)} have ˌ, Piper {piper_has_sec}/{len(subset)}")

    # ==========================================
    # Analysis 6: Verb conjugation stress patterns
    # ==========================================
    print(f"\n{'='*80}")
    print("ANALYSIS 6: VERB CONJUGATION STRESS PATTERNS")
    print(f"{'='*80}")

    for stem, forms in VERB_CONJUGATIONS.items():
        print(f"\n  {stem}:")
        for r in results:
            if r["word"] in forms:
                match = "✓" if r["moon_pattern"] == r["piper_pattern"] else "✗"
                print(f"    {match} {r['word']:12s} ({r['nsyl']}syl)  moon={r['moon']:25s}  piper={r['piper']}")

    # ==========================================
    # Analysis 7: What comes right after ˈ in Piper
    # ==========================================
    print(f"\n{'='*80}")
    print("ANALYSIS 7: WHAT FOLLOWS PRIMARY STRESS IN PIPER")
    print(f"{'='*80}")

    piper_after_stress = Counter()
    moon_after_stress = Counter()
    for r in results:
        for m in r["piper_markers"]:
            if m["marker"] == "ˈ" and m["after"]:
                piper_after_stress[m["after"][0]] += 1
        for m in r["moon_markers"]:
            if m["marker"] == "ˈ" and m["after"]:
                moon_after_stress[m["after"][0]] += 1

    print(f"\n  Char after ˈ     Moonshine   Piper")
    print(f"  ─────────────    ─────────   ─────")
    all_chars = sorted(set(list(moon_after_stress.keys()) + list(piper_after_stress.keys())),
                       key=lambda c: -(moon_after_stress[c] + piper_after_stress[c]))
    for ch in all_chars[:15]:
        print(f"  {ch!r:14s}    {moon_after_stress[ch]:5d}       {piper_after_stress[ch]:5d}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
