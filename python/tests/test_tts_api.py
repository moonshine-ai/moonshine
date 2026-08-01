"""Tests for the TextToSpeech builder API.

These drive the public surface against stand-ins for the native synthesizer and
the asset downloader, so they need neither a model download nor an audio
device. The behaviour they pin down is what the documented examples depend on:
configure with chainable setters, call load(), then say() or clone_from().
"""

import pytest


@pytest.fixture
def tts_module():
    from moonshine_voice import tts

    return tts


class FakeLib:
    """Stands in for the loaded shared library."""

    def __init__(self):
        self.created_from_files = []
        self.created_from_memory = []
        self.freed = []
        self._next_handle = 1

    def _handle(self):
        handle = self._next_handle
        self._next_handle += 1
        return handle

    def moonshine_create_tts_synthesizer_from_files(
        self, language, filenames, count, options, options_count, version
    ):
        self.created_from_files.append(
            (language.decode("utf-8"), _options_dict(options, options_count))
        )
        return self._handle()

    def moonshine_create_tts_synthesizer_from_memory(
        self, language, filenames, count, memory, sizes, options, options_count, version
    ):
        self.created_from_memory.append(
            (language.decode("utf-8"), _options_dict(options, options_count))
        )
        return self._handle()

    def moonshine_free_tts_synthesizer(self, handle):
        self.freed.append(handle)

    def moonshine_error_to_string(self, code):
        return b"fake failure"


def _options_dict(options, count):
    return {
        options[i].name.decode("utf-8"): options[i].value.decode("utf-8")
        for i in range(count)
    }


@pytest.fixture
def fake_native(tts_module, monkeypatch, tmp_path):
    """Swaps out the native library and every download path."""
    lib = FakeLib()
    downloads = []

    monkeypatch.setattr(
        tts_module, "_MoonshineLib", lambda: type("L", (), {"lib": lib})()
    )
    monkeypatch.setattr(
        tts_module,
        "validate_tts_language",
        lambda language, **kwargs: language.replace("-", "_"),
    )
    monkeypatch.setattr(
        tts_module, "validate_tts_voice_known", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        tts_module, "ensure_tts_voice_downloaded", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(tts_module, "tts_asset_cache_path", lambda root: tmp_path)

    def fake_download(language, *, voice=None, options=None, cache_root=None, **kwargs):
        downloads.append(
            {
                "language": language,
                "voice": voice,
                "cache_root": cache_root,
                "on_progress": kwargs.get("on_progress"),
            }
        )
        return tmp_path

    monkeypatch.setattr(tts_module, "download_tts_assets", fake_download)
    lib.downloads = downloads
    return lib


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_the_old_constructor_arguments_name_their_replacement(tts_module):
    with pytest.raises(TypeError) as excinfo:
        tts_module.TextToSpeech("en_us", voice="kokoro_af_heart")

    message = str(excinfo.value)
    assert ".language()" in message
    assert "load()" in message
    assert "clone_from" in message


def test_positional_arguments_are_refused_too(tts_module):
    with pytest.raises(TypeError):
        tts_module.TextToSpeech("en_us")


def test_constructing_one_touches_nothing(tts_module, fake_native):
    """The constructor cannot fail, so nothing is opened until load()."""
    tts_module.TextToSpeech()

    assert fake_native.created_from_files == []
    assert fake_native.downloads == []


def test_setters_are_chainable(tts_module):
    tts = tts_module.TextToSpeech()

    result = (
        tts.language("en_us")
        .voice("kokoro_af_heart")
        .models_from("unused")
        .cloning(False)
        .options({"speed": "1.1"})
        .output_device(3)
        .volume(0.5)
        .debug(False)
        .on_progress(lambda fraction, name: None)
    )

    assert result is tts


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_load_passes_configuration_through(tts_module, fake_native, tmp_path):
    tts = (
        tts_module.TextToSpeech()
        .language("en-us")
        .voice("kokoro_af_heart")
        .options({"speed": "1.1"})
    )

    assert tts.load() is tts

    language, options = fake_native.created_from_files[0]
    assert language == "en_us"
    assert options["voice"] == "kokoro_af_heart"
    assert options["speed"] == "1.1"
    assert options["g2p_root"] == str(tmp_path)
    assert tts.language_tag == "en_us"
    assert tts.asset_root == tmp_path


def test_load_is_idempotent(tts_module, fake_native):
    tts = tts_module.TextToSpeech()
    tts.load()

    tts.load()

    assert len(fake_native.created_from_files) == 1


def test_the_progress_handler_reaches_the_downloader(tts_module, fake_native):
    def handler(fraction, name):
        pass

    tts_module.TextToSpeech().on_progress(handler).load()

    assert fake_native.downloads[0]["on_progress"] is handler


def test_models_from_skips_the_download(tts_module, fake_native, tmp_path):
    local = tmp_path / "already-here"
    local.mkdir()

    tts = tts_module.TextToSpeech().models_from(local)
    tts.load()

    assert fake_native.downloads == []
    assert tts.asset_root == local.resolve()


def test_models_from_can_be_a_cache_root_instead(tts_module, fake_native, tmp_path):
    cache = tmp_path / "cache"

    tts_module.TextToSpeech().models_from(cache, download=True).load()

    assert fake_native.downloads[0]["cache_root"] == cache


def test_a_voice_named_through_options_is_still_a_voice(tts_module, fake_native):
    """AgentFlow passes its voice down as an option rather than a setter."""
    tts_module.TextToSpeech().options({"voice": "kokoro_af_heart"}).load()

    assert fake_native.downloads[0]["voice"] == "kokoro_af_heart"
    _, options = fake_native.created_from_files[0]
    assert options["voice"] == "kokoro_af_heart"


def test_a_failure_to_create_reports_the_native_message(
    tts_module, fake_native, monkeypatch
):
    monkeypatch.setattr(
        fake_native,
        "moonshine_create_tts_synthesizer_from_files",
        lambda *args: -3,
    )

    from moonshine_voice.errors import MoonshineError

    with pytest.raises(MoonshineError, match="fake failure"):
        tts_module.TextToSpeech().load()


# ---------------------------------------------------------------------------
# Using it before it is ready
# ---------------------------------------------------------------------------


def test_saying_something_before_load_says_what_to_do(tts_module, fake_native):
    from moonshine_voice.errors import MoonshineError

    with pytest.raises(MoonshineError, match=r"load\(\)"):
        tts_module.TextToSpeech().say("hello")


def test_synthesizing_before_load_says_what_to_do(tts_module, fake_native):
    from moonshine_voice.errors import MoonshineError

    with pytest.raises(MoonshineError, match=r"load\(\)"):
        tts_module.TextToSpeech().synthesize("hello")


def test_asset_root_before_load_says_what_to_do(tts_module, fake_native):
    from moonshine_voice.errors import MoonshineError

    with pytest.raises(MoonshineError, match=r"load\(\)"):
        tts_module.TextToSpeech().asset_root


# ---------------------------------------------------------------------------
# Cloning
# ---------------------------------------------------------------------------


def test_cloning_fetches_the_engine_but_leaves_the_voice_open(tts_module, fake_native):
    tts = tts_module.TextToSpeech().cloning()

    tts.load()

    # There is nothing to clone yet, so the assets come down without a
    # synthesizer being built on top of them.
    assert fake_native.downloads[0]["voice"] == "zipvoice"
    assert fake_native.created_from_files == []
    assert not tts.is_cloned


def test_using_a_cloning_synthesizer_early_points_at_clone_from(
    tts_module, fake_native
):
    from moonshine_voice.errors import MoonshineError

    tts = tts_module.TextToSpeech().cloning()
    tts.load()

    with pytest.raises(MoonshineError, match="clone_from"):
        tts.synthesize("hello")


def test_clone_from_samples_builds_from_memory(tts_module, fake_native):
    tts = tts_module.TextToSpeech()
    tts.load()

    tts.clone_from(([0.1, 0.2, 0.3], 16000), transcript="hello there")

    assert tts.is_cloned
    language, options = fake_native.created_from_memory[0]
    assert options["voice"] == "zipvoice"
    assert options["zipvoice_clone_transcript"] == "hello there"
    assert options["zipvoice_clone_sample_rate"] == "16000"


def test_cloning_replaces_the_earlier_synthesizer(tts_module, fake_native):
    tts = tts_module.TextToSpeech().voice("kokoro_af_heart")
    tts.load()
    first = tts._handle

    tts.clone_from(([0.1, 0.2], 16000), transcript="hello")

    assert fake_native.freed == [first]


def test_clone_from_a_voice_clone_uses_its_audio(tts_module, fake_native):
    from moonshine_voice.voice_clone import VoiceClone

    clone = VoiceClone()
    clone._clip = [0.1, 0.2, 0.3]

    tts = tts_module.TextToSpeech()
    tts.load()
    tts.clone_from(clone, transcript="hello")

    _, options = fake_native.created_from_memory[0]
    assert options["zipvoice_clone_sample_rate"] == str(VoiceClone.CLIP_SAMPLE_RATE)


def test_clone_from_an_unfinished_voice_clone_says_to_wait(tts_module, fake_native):
    from moonshine_voice.errors import MoonshineError
    from moonshine_voice.voice_clone import VoiceClone

    tts = tts_module.TextToSpeech()
    tts.load()

    with pytest.raises(MoonshineError, match="on_ready"):
        tts.clone_from(VoiceClone())


def test_start_cloning_passes_its_thresholds_on(tts_module, fake_native):
    clone = tts_module.TextToSpeech().start_cloning(
        clip_duration_seconds=6, minimum_speech_seconds=3
    )

    assert clone._clip_duration_seconds == 6
    assert clone._minimum_speech_seconds == 3


# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------


def test_close_releases_the_synthesizer(tts_module, fake_native):
    tts = tts_module.TextToSpeech()
    tts.load()
    handle = tts._handle

    tts.close()

    assert fake_native.freed == [handle]


def test_it_works_as_a_context_manager(tts_module, fake_native):
    with tts_module.TextToSpeech() as tts:
        tts.load()
        handle = tts._handle

    assert fake_native.freed == [handle]


def test_closing_one_that_never_loaded_does_not_explode(tts_module, fake_native):
    """Tidying up after a failed load should not raise on top of it."""
    tts_module.TextToSpeech().close()

    assert fake_native.freed == []
