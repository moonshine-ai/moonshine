"""Runs the ``__main__`` sections of the most significant modules.

Each module in moonshine_voice doubles as a command-line demo through its
``if __name__ == "__main__"`` block, so invoking ``python -m`` on it
exercises the module end to end: model loading, the native library, and the
public API. These tests run the headless-safe ones for real and check their
output, and smoke-test the microphone-only ones by starting them and failing
only on an argument parsing error (the same trick as test_docs.py).

The suite is meant to stay under roughly three minutes with a warm model
cache, so it uses the short bundled audio clips and the quantized embedding
model.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

FULL_RUN_TIMEOUT_SECONDS = 300
# Long enough for a warm-cache model load to reach the microphone banner, short
# enough that the two mic modules (which then run until killed) don't dominate
# the suite.
SMOKE_TIMEOUT_SECONDS = 20

ARGPARSE_USAGE_ERROR_EXIT_CODE = 2


def run_module(module, *args, cwd=None, timeout=FULL_RUN_TIMEOUT_SECONDS):
    return subprocess.run(
        [sys.executable, "-m", f"moonshine_voice.{module}", *args],
        cwd=cwd,
        timeout=timeout,
        capture_output=True,
        text=True,
        # Modules emit UTF-8 (e.g. g2p prints IPA like the schwa 'ə'). Decode
        # their output as UTF-8 explicitly so capture doesn't blow up on
        # Windows, where subprocess text mode otherwise defaults to the cp1252
        # console code page and chokes on those bytes. errors="replace" keeps a
        # genuinely mangled byte from masking the real assertion.
        encoding="utf-8",
        errors="replace",
    )


def describe(result):
    return (
        f"exit code {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )


def assets_path():
    moonshine_voice = pytest.importorskip("moonshine_voice")
    return Path(moonshine_voice.get_assets_path())


def test_transcriber_transcribes_bundled_audio():
    wav_path = assets_path() / "beckett.wav"
    result = run_module("transcriber", "--wav-path", str(wav_path), "--quiet")
    assert result.returncode == 0, describe(result)
    # The clip says "Ever tried, ever failed. No matter. Try again.
    # Fail again. Fail better."
    combined = (result.stdout + result.stderr).lower()
    assert "fail" in combined, describe(result)


def test_diarization_finds_two_speakers_on_endgame_clip():
    """Synthetic Nagg/Nell ZipVoice dialogue should yield two speaker IDs."""
    moonshine_voice = pytest.importorskip("moonshine_voice")
    from moonshine_voice import Transcriber
    from moonshine_voice.moonshine_api import ModelArch
    from moonshine_voice.utils import load_wav_file

    wav_path = assets_path() / "endgame_nagg_nell.wav"
    if not wav_path.exists():
        pytest.skip("endgame_nagg_nell.wav not bundled")

    model_path = REPO_ROOT / "test-assets" / "tiny-en"
    if not (model_path / "decoder_with_attention.ort").exists():
        pytest.skip("test-assets/tiny-en missing decoder_with_attention.ort")

    # The diarization models are a download, so use the checked-in copies
    # rather than reaching for the network in a unit test.
    diarization_path = REPO_ROOT / "test-assets" / "diarization"
    if not (diarization_path / "segmentation.ort").exists():
        pytest.skip("test-assets/diarization missing segmentation.ort")

    audio, sample_rate = load_wav_file(wav_path)
    duration = len(audio) / float(sample_rate)
    assert 20.0 <= duration <= 35.0

    transcriber = Transcriber(
        str(model_path),
        model_arch=ModelArch.TINY,
        options={
            "identify_speakers": "true",
            "diarization_model_dir": str(diarization_path),
        },
    )
    transcript = transcriber.transcribe_without_streaming(audio, sample_rate)
    assert transcript.lines

    speaker_ids = set()
    total_span_duration = 0.0
    lines_with_spans = 0
    for line in transcript.lines:
        assert line.words
        if not line.speaker_spans:
            continue
        lines_with_spans += 1
        for span in line.speaker_spans:
            speaker_ids.add(span.speaker_id)
            total_span_duration += span.duration
            if span.end_char > span.start_char:
                assert span.end_char <= len(line.text.encode("utf-8"))

    assert lines_with_spans > 0
    assert len(speaker_ids) >= 2
    assert total_span_duration >= duration * 0.35
    assert total_span_duration <= duration * 1.25


def test_tts_synthesizes_wav(tmp_path):
    out_path = tmp_path / "out.wav"
    result = run_module(
        "tts",
        "--language", "en_us",
        "--text", "Hello world",
        "--out", str(out_path),
        cwd=tmp_path,
    )
    assert result.returncode == 0, describe(result)
    assert out_path.exists(), describe(result)
    # A WAV header alone is 44 bytes; real speech is far larger.
    assert out_path.stat().st_size > 10000, describe(result)


def test_g2p_prints_ipa():
    result = run_module("g2p", "--language", "en_us", "--text", "Hello world")
    assert result.returncode == 0, describe(result)
    assert result.stdout.strip(), describe(result)


def test_embedding_backend_matches_transcribed_command():
    """The embedding model is internal (AgentFlow owns one), so this drives
    the module directly rather than through a CLI demo. It scores phrases the
    way AgentFlow does, through calculate_embedding and distance."""
    moonshine_voice = pytest.importorskip("moonshine_voice")
    from moonshine_voice.embedding_model import EmbeddingModel

    wav_path = REPO_ROOT / "test-assets" / "intent.wav"
    audio, sample_rate = moonshine_voice.load_wav_file(str(wav_path))
    model_path, model_arch = moonshine_voice.get_model_for_language("en")
    transcriber = moonshine_voice.Transcriber(model_path, model_arch)
    try:
        transcript = transcriber.transcribe_without_streaming(audio, sample_rate)
    finally:
        transcriber.close()
    utterance = " ".join(line.text for line in transcript.lines).strip()
    assert utterance, "expected the clip to transcribe to something"

    unrelated = "bake a chocolate cake"
    phrases = ("move forward", "move backward", "turn left", "turn right")

    embedding_path, embedding_arch = moonshine_voice.get_embedding_model(variant="q4")
    embedder = EmbeddingModel(
        model_path=embedding_path, model_arch=embedding_arch, model_variant="q4"
    )
    try:
        spoken = embedder.calculate_embedding(utterance)
        scores = {
            phrase: embedder.distance(
                spoken, embedder.calculate_embedding(phrase)
            )
            for phrase in (*phrases, unrelated)
        }
    finally:
        embedder.close()

    best = max(scores, key=scores.get)
    assert best != unrelated, f"{utterance!r} scored closest to {unrelated!r}: {scores}"


def test_download_g2p_assets():
    result = run_module("download", "--g2p", "--language", "en_us")
    assert result.returncode == 0, describe(result)
    root = result.stdout.strip().splitlines()[-1]
    assert Path(root).exists(), describe(result)


def test_download_tts_assets_zipvoice_fetches_model_files():
    """The bare ``zipvoice`` engine selector (used for voice cloning) must
    download the shared ZipVoice model files, not be dropped as an unknown
    voice id. Regression test for a missing ``zipvoice/tokens.txt`` when
    cloning a voice.
    """
    result = run_module(
        "download", "--tts", "--language", "en_us", "--voice", "zipvoice"
    )
    assert result.returncode == 0, describe(result)
    root = Path(result.stdout.strip().splitlines()[-1])
    assert root.exists(), describe(result)
    assert (root / "zipvoice" / "tokens.txt").exists(), describe(result)


def test_agent_flow_lists_output_devices():
    result = run_module("agent_flow", "--list-output-devices")
    assert result.returncode == 0, describe(result)


@pytest.mark.parametrize(
    "module,args,banner",
    [
        ("mic_transcriber", ["--language", "en"], "Listening to the microphone"),
        (
            "alphanumeric_listener",
            ["--language", "en"],
            "speak letters, digits, or symbols",
        ),
    ],
)
def test_mic_modules_get_as_far_as_opening_the_microphone(module, args, banner):
    """Microphone modules can't run headlessly, so start them and check they
    reach the banner they print just before opening the capture device.

    Getting that far means the arguments parsed and the model loaded, which is
    everything we can check without hardware. Looking for the banner rather
    than the exit code means this still catches a crash on a machine with no
    microphone at all, where the process exits either way.
    """
    try:
        result = run_module(module, *args, timeout=SMOKE_TIMEOUT_SECONDS)
        output = result.stdout + result.stderr
        assert result.returncode != ARGPARSE_USAGE_ERROR_EXIT_CODE, describe(result)
    except subprocess.TimeoutExpired as expired:
        # Still running, which is the healthy case on a machine with a mic.
        output = _partial_output(expired)

    assert banner in output, f"{module} never reached the microphone:\n{output}"


def _partial_output(expired):
    """Text captured before a timed-out module was killed."""
    chunks = []
    for stream in (expired.stdout, expired.stderr):
        if stream is None:
            continue
        chunks.append(
            stream if isinstance(stream, str) else stream.decode("utf-8", "replace")
        )
    return "".join(chunks)
