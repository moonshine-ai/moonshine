"""Stdlib-only manifest helpers: these must work without the ``[lora]`` extra."""

import json

from moonshine_voice.lora.manifest import (
    apply_text_mode,
    choose_text_mode,
    load_manifest,
)


def test_jsonl_manifest(tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"fake")
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        json.dumps({"audio": "a.wav", "text": "hello there"}) + "\n"
        + json.dumps({"path": str(wav), "text": "second clip"}) + "\n"
    )
    rows = load_manifest(str(manifest))
    assert len(rows) == 2
    assert rows[0].text == "hello there"
    assert rows[0].audio.endswith("a.wav")
    assert rows[1].text == "second clip"


def test_json_utterances_wrapper(tmp_path):
    manifest = tmp_path / "train.json"
    manifest.write_text(
        json.dumps({"utterances": [{"audio": "x.wav", "text": "one", "duration": 1.5}]})
    )
    rows = load_manifest(str(manifest), data_root=str(tmp_path))
    assert rows[0].seconds == 1.5
    assert rows[0].audio.endswith("x.wav")


def test_tsv_manifest(tmp_path):
    manifest = tmp_path / "train.tsv"
    manifest.write_text("clip.wav\tlufthansa four six five two\n")
    rows = load_manifest(str(manifest))
    assert rows[0].text == "lufthansa four six five two"
    assert rows[0].audio.endswith("clip.wav")


def test_auto_text_mode_lowercases_uppercase_corpora():
    assert choose_text_mode(["HELLO WORLD", "GOOD MORNING"]) == "lower"
    assert choose_text_mode(["Hello world.", "Good morning."]) == "none"
    assert choose_text_mode(["HELLO"], requested="none") == "none"
    assert apply_text_mode("HELLO", "lower") == "hello"
    assert apply_text_mode("Hello.", "none") == "Hello."


def test_empty_text_is_rejected(tmp_path):
    manifest = tmp_path / "bad.jsonl"
    manifest.write_text(json.dumps({"audio": "a.wav", "text": "  "}) + "\n")
    try:
        load_manifest(str(manifest))
    except ValueError as error:
        assert "text" in str(error)
    else:
        raise AssertionError("expected ValueError")
