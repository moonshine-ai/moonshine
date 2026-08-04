"""Tests for the MicTranscriber builder API.

These drive the public surface against stand-ins for the native transcriber
and stream, so they need neither a model nor a microphone. The behaviour they
pin down is what the documented example depends on: configure with chainable
setters, register handlers before anything is loaded, then load() and start().
"""

from types import SimpleNamespace
from pathlib import Path

import pytest

from moonshine_voice.transcriber import (
    Error,
    LineCompleted,
    LineStarted,
    LineTextChanged,
    ModelArch,
    TranscriptEventListener,
)


@pytest.fixture
def mic_module():
    pytest.importorskip("sounddevice")
    from moonshine_voice import mic_transcriber

    return mic_transcriber


class FakeStream:
    def __init__(self, update_interval=None, transcribe_flags=0):
        self.update_interval = update_interval
        self.transcribe_flags = transcribe_flags
        self.listeners = []
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True

    def set_transcribe_flags(self, flags):
        self.transcribe_flags = int(flags)

    def add_listener(self, listener):
        self.listeners.append(listener)

    def remove_listener(self, listener):
        if listener in self.listeners:
            self.listeners.remove(listener)

    def remove_all_listeners(self):
        self.listeners.clear()

    def emit(self, event):
        for listener in list(self.listeners):
            listener(event)


class FakeTranscriber:
    def __init__(self, model_path, model_arch=None, options=None):
        self.model_path = model_path
        self.model_arch = model_arch
        self.options = options
        self.closed = False
        self.stream = None

    def create_stream(self, update_interval=None, transcribe_flags=0):
        self.stream = FakeStream(update_interval, transcribe_flags)
        return self.stream

    def close(self):
        self.closed = True


@pytest.fixture
def fake_native(mic_module, monkeypatch):
    """Swaps in the fake transcriber and returns the instances it creates."""
    created = []

    def factory(*args, **kwargs):
        transcriber = FakeTranscriber(*args, **kwargs)
        created.append(transcriber)
        return transcriber

    monkeypatch.setattr(mic_module, "Transcriber", factory)
    # Speaker identification would otherwise fetch the diarization models from
    # the CDN, which these tests have no business doing.
    monkeypatch.setattr(
        mic_module, "get_diarization_model", lambda: "downloaded/diarization"
    )
    return created


def line(text="hello"):
    return SimpleNamespace(text=text)


def loaded(mic_module, **_):
    """A MicTranscriber pointed at a local directory, so load() downloads nothing."""
    return mic_module.MicTranscriber().models_from("unused")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_the_old_constructor_arguments_name_their_replacement(mic_module):
    with pytest.raises(TypeError) as excinfo:
        mic_module.MicTranscriber(model_path="somewhere", model_arch=ModelArch.TINY)

    message = str(excinfo.value)
    assert "models_from" in message
    assert "load()" in message


def test_positional_arguments_are_refused_too(mic_module):
    with pytest.raises(TypeError):
        mic_module.MicTranscriber("somewhere")


def test_constructing_one_touches_nothing(mic_module, fake_native):
    """The constructor cannot fail, so nothing is opened until load()."""
    mic = mic_module.MicTranscriber()

    assert mic.transcriber is None
    assert mic.mic_stream is None
    assert fake_native == []


def test_setters_are_chainable(mic_module):
    mic = mic_module.MicTranscriber()

    result = (
        mic.language("en")
        .model_arch(ModelArch.TINY)
        .models_from("unused")
        .update_interval(0.25)
        .options({"identify_speakers": "true"})
        .device(3)
        .samplerate(44100)
        .channels(2)
        .blocksize(2048)
        .transcribe_flags(1)
        .on_text(lambda text: None)
        .on_line(lambda line: None)
        .on_error(lambda error: None)
        .on_progress(lambda fraction, name: None)
    )

    assert result is mic


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_load_passes_configuration_through(mic_module, fake_native):
    mic = (
        mic_module.MicTranscriber()
        .models_from("some/model/dir")
        .model_arch(ModelArch.TINY)
        .update_interval(0.25)
        .options({"identify_speakers": "true"})
        .transcribe_flags(4)
    )

    assert mic.load() is mic

    transcriber = fake_native[0]
    assert Path(transcriber.model_path) == Path("some/model/dir")
    assert transcriber.model_arch == ModelArch.TINY
    assert transcriber.options["identify_speakers"] == "true"
    # Asking for speaker IDs fetches the diarization models and points the
    # transcriber at them; they are no longer part of the library.
    assert Path(transcriber.options["diarization_model_dir"]) == Path(
        "downloaded/diarization"
    )
    assert mic.mic_stream.update_interval == 0.25
    assert mic.mic_stream.transcribe_flags == 4


def test_load_is_idempotent(mic_module, fake_native):
    mic = loaded(mic_module)
    mic.load()
    stream = mic.mic_stream

    mic.load()

    assert mic.mic_stream is stream
    assert len(fake_native) == 1


def test_load_downloads_by_language_when_no_directory_is_given(
    mic_module, fake_native, monkeypatch
):
    seen = {}

    def fake_get_model(language, arch, on_progress=None):
        seen["language"] = language
        seen["arch"] = arch
        seen["on_progress"] = on_progress
        return "downloaded/dir", ModelArch.MEDIUM_STREAMING

    monkeypatch.setattr(mic_module, "get_model_for_language", fake_get_model)
    monkeypatch.setattr(mic_module, "get_spelling_model_path", lambda language: None)
    handler = lambda fraction, name: None  # noqa: E731

    mic_module.MicTranscriber().language("es").on_progress(handler).load()

    assert seen["language"] == "es"
    # None means "the catalog's default for this language". Only English
    # publishes a medium streaming model, so naming one here would break
    # every other language.
    assert seen["arch"] is None
    assert seen["on_progress"] is handler


def test_the_spelling_model_is_found_automatically(mic_module, fake_native, monkeypatch):
    monkeypatch.setattr(
        mic_module,
        "get_model_for_language",
        lambda language, arch, on_progress=None: ("dir", ModelArch.TINY),
    )
    monkeypatch.setattr(
        mic_module, "get_spelling_model_path", lambda language: "spelling.ort"
    )

    mic_module.MicTranscriber().load()

    assert fake_native[0].options["spelling_model_path"] == "spelling.ort"


def test_passing_no_spelling_model_stops_the_lookup(mic_module, fake_native, monkeypatch):
    def fail(language):
        raise AssertionError("should not look for a spelling model")

    monkeypatch.setattr(
        mic_module,
        "get_model_for_language",
        lambda language, arch, on_progress=None: ("dir", ModelArch.TINY),
    )
    monkeypatch.setattr(mic_module, "get_spelling_model_path", fail)

    mic = mic_module.MicTranscriber().spelling_model(None)
    mic.load()

    assert not (fake_native[0].options or {}).get("spelling_model_path")


def test_a_missing_spelling_model_is_not_fatal(mic_module, fake_native, monkeypatch):
    """It is not published for every language, and only costs accuracy inside
    spelling mode."""

    def boom(language):
        raise RuntimeError("no spelling model published")

    monkeypatch.setattr(
        mic_module,
        "get_model_for_language",
        lambda language, arch, on_progress=None: ("dir", ModelArch.TINY),
    )
    monkeypatch.setattr(mic_module, "get_spelling_model_path", boom)

    mic_module.MicTranscriber().load()

    assert fake_native[0].model_path == "dir"


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def test_handlers_registered_before_load_still_fire(mic_module, fake_native):
    """The documented example configures handlers before the stream exists."""
    partial, final, failures = [], [], []
    mic = (
        mic_module.MicTranscriber()
        .models_from("unused")
        .on_text(partial.append)
        .on_line(lambda line: final.append(line.text))
        .on_error(failures.append)
    )

    mic.load()
    mic.mic_stream.emit(LineTextChanged(line=line("par"), stream_handle=1))
    mic.mic_stream.emit(LineCompleted(line=line("partial"), stream_handle=1))
    failure = RuntimeError("audio died")
    mic.mic_stream.emit(Error(error=failure, stream_handle=1))

    assert partial == ["par"]
    assert final == ["partial"]
    assert failures == [failure]


def test_on_text_ignores_other_events(mic_module, fake_native):
    seen = []
    mic = loaded(mic_module).on_text(seen.append)
    mic.load()

    mic.mic_stream.emit(LineStarted(line=line("x"), stream_handle=1))
    mic.mic_stream.emit(LineCompleted(line=line("x"), stream_handle=1))

    assert seen == []


def test_listener_objects_still_work(mic_module, fake_native):
    """add_listener stays the escape hatch for line ids and speaker spans."""
    seen = []

    class Recorder(TranscriptEventListener):
        def on_line_completed(self, event):
            seen.append(event.line.text)

    recorder = Recorder()
    mic = loaded(mic_module)
    mic.add_listener(recorder)
    mic.load()

    assert recorder in mic.mic_stream.listeners


def test_a_listener_can_be_removed_before_load(mic_module, fake_native):
    def listener(event):
        pass

    mic = loaded(mic_module)
    mic.add_listener(listener)
    mic.remove_listener(listener)
    mic.load()

    assert mic.mic_stream.listeners == []


def test_push_listener_before_load_says_what_to_do(mic_module):
    mic = mic_module.MicTranscriber()

    with pytest.raises(RuntimeError, match="load()"):
        mic.push_listener(lambda event: None)


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def test_start_before_load_says_what_to_do(mic_module):
    mic = mic_module.MicTranscriber()

    with pytest.raises(RuntimeError, match="load()"):
        mic.start()


def test_transcribe_flags_apply_live_once_loaded(mic_module, fake_native):
    mic = loaded(mic_module)
    mic.load()

    mic.transcribe_flags(8)

    assert mic.mic_stream.transcribe_flags == 8


def test_set_transcribe_flags_is_the_same_thing(mic_module, fake_native):
    """AgentFlow calls this one to flip spelling mode mid-conversation."""
    mic = loaded(mic_module)
    mic.load()

    mic.set_transcribe_flags(2)

    assert mic.mic_stream.transcribe_flags == 2


def test_close_releases_a_transcriber_we_opened(mic_module, fake_native):
    mic = loaded(mic_module)
    mic.load()
    stream = mic.mic_stream

    mic.close()

    assert stream.closed
    assert fake_native[0].closed
    assert mic.mic_stream is None


def test_close_leaves_a_borrowed_transcriber_alone(mic_module, fake_native):
    """use_transcriber() means the caller keeps ownership."""
    borrowed = FakeTranscriber("elsewhere")
    mic = mic_module.MicTranscriber().use_transcriber(borrowed)
    mic.load()

    mic.close()

    assert not borrowed.closed


def test_it_works_as_a_context_manager(mic_module, fake_native):
    with loaded(mic_module).load() as mic:
        stream = mic.mic_stream

    assert stream.closed


def test_stop_before_load_does_not_explode(mic_module):
    """Tidying up after a failed load should not raise on top of it."""
    mic_module.MicTranscriber().stop()
    mic_module.MicTranscriber().close()
