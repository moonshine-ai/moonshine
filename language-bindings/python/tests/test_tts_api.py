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


def test_split_say_utterances_breaks_on_punct_plus_space(tts_module):
    split = tts_module.split_say_utterances
    assert split("") == []
    assert split("  ") == []
    assert split("Hello") == ["Hello"]
    assert split("Hello.") == ["Hello."]
    assert split("Hello. World") == ["Hello.", "World"]
    assert split("Hello! World? Yes.") == ["Hello!", "World?", "Yes."]
    assert split("3.14 is pi.") == ["3.14 is pi."]
    assert split("Hello.  World") == ["Hello.", "World"]
    assert split("Warning: the core is hot.") == [
        "Warning:",
        "the core is hot.",
    ]


def test_split_say_utterances_keeps_abbreviations_whole(tts_module):
    split = tts_module.split_say_utterances
    assert split("Dr. Smith is here.") == ["Dr. Smith is here."]
    assert split("J. R. R. Tolkien wrote it.") == ["J. R. R. Tolkien wrote it."]
    assert split("Bring milk, eggs, etc. Then come home.") == [
        "Bring milk, eggs, etc. Then come home.",
    ]
    assert split("It is 3 p.m. now.") == ["It is 3 p.m. now."]


def test_split_say_utterances_handles_other_scripts(tts_module):
    split = tts_module.split_say_utterances
    assert split("やめて。そこまでだ。") == ["やめて。", "そこまでだ。"]
    assert split("नमस्ते। आप कैसे हैं।") == ["नमस्ते।", "आप कैसे हैं।"]
    assert split("مرحبا؟ كيف حالك؟") == ["مرحبا؟", "كيف حالك؟"]


def test_split_say_utterances_uses_the_language_abbreviations(tts_module):
    split = tts_module.split_say_utterances
    # "z.B." only ends a sentence in languages that don't use it as "e.g.".
    assert split("Nimm z.B. den Zug.", "de") == ["Nimm z.B. den Zug."]


def test_synthesizing_before_load_says_what_to_do(tts_module, fake_native):
    from moonshine_voice.errors import MoonshineError

    with pytest.raises(MoonshineError, match=r"load\(\)"):
        tts_module.TextToSpeech().synthesize("hello")


def test_asset_root_before_load_says_what_to_do(tts_module, fake_native):
    from moonshine_voice.errors import MoonshineError

    with pytest.raises(MoonshineError, match=r"load\(\)"):
        tts_module.TextToSpeech().asset_root


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_stream(tts_module, monkeypatch):
    """Stands in for the native streaming calls with a one-chunk-per-unit engine."""

    class FakeStream:
        NEED_TEXT = 1
        END_OF_STREAM = 2
        CANCELLED = 3

        def __init__(self):
            self.buffer = ""
            self.units = []
            self.ended = False
            self.utterance_id = 0
            self.streaming = False
            self.cancelled = False

        def push_text(self, _handle, text):
            self.streaming = True
            self.buffer += text
            # Anything up to and including a full stop is a complete unit.
            while "." in self.buffer:
                head, _, self.buffer = self.buffer.partition(".")
                self.units.append(head.strip() + ".")

        def flush(self, _handle):
            if self.buffer.strip():
                self.units.append(self.buffer.strip())
            self.buffer = ""

        def end_input(self, handle):
            self.flush(handle)
            self.ended = True

        def cancel(self, _handle):
            self.buffer = ""
            self.units.clear()
            self.ended = False
            self.streaming = False
            self.cancelled = True

        def is_streaming(self, _handle):
            return self.streaming

        def next_chunk(self, _handle):
            if self.cancelled:
                # Reported once and then forgotten, per MOONSHINE_TTS_CANCELLED: a consumer
                # pulling chunks has no other way to tell a reply that was abandoned from
                # one that is merely waiting for more text.
                self.cancelled = False
                return self.CANCELLED, None
            if self.units:
                text = self.units.pop(0)
                self.utterance_id += 1
                chunk = ([0.5] * len(text), 24000, text, self.utterance_id, True)
                return 0, chunk
            if self.ended:
                self.streaming = False
                return self.END_OF_STREAM, None
            return self.NEED_TEXT, None

    fake = FakeStream()
    monkeypatch.setattr(tts_module, "moonshine_tts_push_text", fake.push_text)
    monkeypatch.setattr(tts_module, "moonshine_tts_flush", fake.flush)
    monkeypatch.setattr(tts_module, "moonshine_tts_end_input", fake.end_input)
    monkeypatch.setattr(tts_module, "moonshine_tts_cancel", fake.cancel)
    monkeypatch.setattr(tts_module, "moonshine_tts_is_streaming", fake.is_streaming)
    monkeypatch.setattr(tts_module, "moonshine_tts_next_chunk", fake.next_chunk)
    return fake


def test_streaming_before_load_says_what_to_do(tts_module, fake_native):
    from moonshine_voice.errors import MoonshineError

    with pytest.raises(MoonshineError, match=r"load\(\)"):
        tts_module.TextToSpeech().push_text("Hello.")


def test_pushed_text_is_held_until_a_sentence_is_complete(
    tts_module, fake_native, fake_stream
):
    tts = tts_module.TextToSpeech()
    tts.load()

    tts.push_text("Hello ")
    assert tts.next_chunk() is None

    tts.push_text("there. And then")
    chunk = tts.next_chunk()
    assert chunk.text == "Hello there."
    assert chunk.is_final
    assert chunk.sample_rate == 24000
    assert chunk.utterance_id == 1

    # The tail is still incomplete, so nothing more comes out yet.
    assert tts.next_chunk() is None
    tts.flush()
    assert tts.next_chunk().text == "And then"


def test_streaming_ends_once_input_ends_and_the_queue_drains(
    tts_module, fake_native, fake_stream
):
    tts = tts_module.TextToSpeech()
    tts.load()

    tts.push_text("One. Two.")
    tts.end_input()
    assert [chunk.text for chunk in tts.stream()] == ["One.", "Two."]
    assert not tts.is_streaming


def test_stream_can_be_given_the_whole_reply(tts_module, fake_native, fake_stream):
    tts = tts_module.TextToSpeech()
    tts.load()

    assert [chunk.text for chunk in tts.stream("One. Two.")] == ["One.", "Two."]


def test_stream_consumes_an_iterable_of_pieces(tts_module, fake_native, fake_stream):
    tts = tts_module.TextToSpeech()
    tts.load()

    tokens = iter(["One", ". ", "Two", "."])
    assert [chunk.text for chunk in tts.stream(tokens)] == ["One.", "Two."]


def test_cancel_drops_what_was_queued(tts_module, fake_native, fake_stream):
    tts = tts_module.TextToSpeech()
    tts.load()

    tts.push_text("Forget this. And this.")
    assert tts.is_streaming
    tts.cancel_stream()
    assert not tts.is_streaming
    assert tts.next_chunk() is None


# ---------------------------------------------------------------------------
# Cloning
# ---------------------------------------------------------------------------


def test_cloning_builds_zipvoice_so_start_cloning_works(tts_module, fake_native):
    tts = tts_module.TextToSpeech().cloning()

    tts.load()

    assert fake_native.downloads[0]["voice"] == "zipvoice_american_female"
    assert len(fake_native.created_from_files) == 1
    assert not tts.is_cloned
    clone = tts.start_cloning()
    assert clone is not None
    assert clone._tts_handle == tts._handle


def test_voice_and_cloning_are_mutually_exclusive(tts_module, fake_native):
    tts = tts_module.TextToSpeech().voice("kokoro_af_heart").cloning()
    assert tts._voice is None
    assert tts._cloning_wanted

    tts.voice("kokoro_af_heart")
    assert tts._cloning_wanted is False


def test_clone_from_requires_cloning_mode(tts_module, fake_native):
    from moonshine_voice.errors import MoonshineError

    tts = tts_module.TextToSpeech()
    tts.load()

    with pytest.raises(MoonshineError, match="cloning"):
        tts.clone_from(([0.1, 0.2, 0.3], 16000), transcript="hello there")


def test_clone_from_samples_builds_from_memory(tts_module, fake_native):
    tts = tts_module.TextToSpeech().cloning()
    tts.load()

    tts.clone_from(([0.1, 0.2, 0.3], 16000), transcript="hello there")

    assert tts.is_cloned
    language, options = fake_native.created_from_memory[0]
    assert options["voice"] == "zipvoice"
    assert options["zipvoice_clone_transcript"] == "hello there"
    assert options["zipvoice_clone_sample_rate"] == "16000"


def test_cloning_replaces_the_earlier_synthesizer(tts_module, fake_native):
    tts = tts_module.TextToSpeech().cloning()
    tts.load()
    first = tts._handle

    tts.clone_from(([0.1, 0.2], 16000), transcript="hello")

    assert fake_native.freed == [first]


def test_clone_from_a_voice_clone_uses_its_audio(tts_module, fake_native):
    from moonshine_voice.voice_clone import VoiceClone

    clone = VoiceClone(tts_handle=0)
    clone._clip = [0.1, 0.2, 0.3]

    tts = tts_module.TextToSpeech().cloning()
    tts.load()
    tts.clone_from(clone, transcript="hello")

    _, options = fake_native.created_from_memory[0]
    assert options["zipvoice_clone_sample_rate"] == str(VoiceClone.CLIP_SAMPLE_RATE)


def test_clone_from_an_unfinished_voice_clone_says_to_wait(tts_module, fake_native):
    from moonshine_voice.errors import MoonshineError
    from moonshine_voice.voice_clone import VoiceClone

    tts = tts_module.TextToSpeech().cloning()
    tts.load()

    with pytest.raises(MoonshineError, match="on_ready"):
        tts.clone_from(VoiceClone(tts_handle=0))


def test_start_cloning_passes_its_thresholds_on(tts_module, fake_native):
    tts = tts_module.TextToSpeech().cloning()
    tts.load()
    clone = tts.start_cloning(clip_duration_seconds=6, minimum_speech_seconds=3)

    assert clone._clip_duration_seconds == 6
    assert clone._minimum_speech_seconds == 3
    assert clone._tts_handle == tts._handle


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


def _play_with_fake_device(tts_module, monkeypatch, tts, data, sample_rate=24000):
    """Run one utterance through `_play_one` against a stand-in output stream."""
    import numpy as np

    written = []

    class FakeStream:
        time = 0.0
        latency = 0.01

        def write(self, chunk):
            written.append(np.array(chunk, copy=True))

    spec_key = tts_module._say_device_spec_key(None)
    tts._say_device_cache = (spec_key, 0)
    tts._say_settings_ok = ((spec_key, sample_rate), sample_rate)
    tts._announced_resolved_device = True
    monkeypatch.setattr(tts, "_acquire_output_stream",
                        lambda sd, device, sr: FakeStream())
    item = tts_module._PlayItem(data=data, sample_rate=sample_rate, device=None)
    tts._play_one(item, object(), np)
    return written


def test_playback_writes_the_samples_unchanged_in_short_blocks(tts_module, monkeypatch):
    """Blocking writes are split so stop() can land between them, without changing the audio."""
    import numpy as np

    tts = tts_module.TextToSpeech()
    rng = np.random.default_rng(0)
    data = rng.normal(0, 0.2, 24000).astype(np.float32)
    written = _play_with_fake_device(tts_module, monkeypatch, tts, data)

    assert len(written) > 1, "the utterance was written as one blocking call"
    assert np.array_equal(np.concatenate(written), data)


def test_is_talking_covers_an_utterance_being_synthesized(tts_module):
    """Off the say queue, not yet on the play queue: the synthesis window."""
    tts = tts_module.TextToSpeech()

    tts._say_queue.put(object())
    assert tts.is_talking() is True

    tts._say_queue.get()
    assert tts._say_queue.empty(), "precondition: the queue really is empty now"
    assert tts.is_talking() is True, "silence reported while an utterance was in flight"

    tts._say_queue.task_done()
    assert tts.is_talking() is False


def test_is_talking_covers_an_utterance_being_played(tts_module):
    """Off the play queue and not yet counted in the tail: most of the utterance."""
    tts = tts_module.TextToSpeech()

    tts._play_queue.put(object())
    tts._play_queue.get()
    assert tts._play_queue.empty()
    assert tts.is_talking() is True, "silence reported while audio was being written"

    tts._play_queue.task_done()
    assert tts.is_talking() is False


def test_stop_discards_queued_utterances(tts_module):
    tts = tts_module.TextToSpeech()
    tts._say_queue.put(object())
    tts._play_queue.put(object())
    assert tts.is_talking() is True

    tts.stop()

    assert tts.is_talking() is False


def test_stop_leaves_nothing_owed_even_if_an_utterance_arrives_mid_stop(tts_module,
                                                                       monkeypatch):
    """Otherwise a caller polling `is_talking` waits for work nothing will ever do.

    Reachable, and not by much: the synthesis worker checks the stop flag before handing an
    utterance over, but the handover blocks while the play queue is full, so stop()'s own
    drain is what releases it -- by which point the playback worker has exited and will
    never retire it. Draining cannot fix that, because the arrival is *after* the drain.
    `_release_output_stream` is called in exactly that window, so it stands in for the
    worker here.
    """
    tts = tts_module.TextToSpeech()
    real_release = tts._release_output_stream

    def hand_over_late(*args, **kwargs):
        tts._play_queue.put(object())
        return real_release(*args, **kwargs)

    monkeypatch.setattr(tts, "_release_output_stream", hand_over_late)

    tts.stop()

    assert tts._say_queue.unfinished_tasks == 0
    assert tts._play_queue.unfinished_tasks == 0
    assert tts.is_talking() is False, "stop() left the synthesizer looking busy forever"


def _start_workers_over_a_fake_device(tts_module, monkeypatch, tts, stream,
                                      sample_rate=24000):
    """Run the real worker pair, with `stream` standing in for the device."""
    import numpy as np

    monkeypatch.setattr(tts_module, "_import_say_audio_deps", lambda: (np, object()))
    spec_key = tts_module._say_device_spec_key(None)
    tts._say_device_cache = (spec_key, 0)
    tts._say_settings_ok = ((spec_key, sample_rate), sample_rate)
    tts._announced_resolved_device = True

    def acquire(sd, device, sr):
        # The real one publishes the stream on the instance, which is what makes it
        # reachable from another thread's `_release_output_stream`.
        tts._output_stream = stream
        tts._output_stream_key = (spec_key, sr)
        return stream

    monkeypatch.setattr(tts, "_acquire_output_stream", acquire)
    tts._ensure_say_workers()


def _live_playback_workers(tts):
    import threading

    return [t for t in threading.enumerate()
            if getattr(t, "_target", None) == tts._play_worker and t.is_alive()]


def test_stop_does_not_close_the_device_under_the_playback_worker(tts_module, monkeypatch):
    """Closing the stream from stop()'s thread frees a buffer the worker is writing into."""
    import numpy as np
    import threading
    import time

    tts = tts_module.TextToSpeech()
    writing = threading.Event()
    wrote_into_a_closed_stream = []

    class FakeStream:
        time = 0.0
        latency = 0.01

        def __init__(self):
            self.open = True

        def write(self, chunk):
            writing.set()
            time.sleep(0.05)
            if not self.open:
                wrote_into_a_closed_stream.append(True)

        def abort(self, ignore_errors=True):
            self.open = False

        def stop(self, ignore_errors=True):
            self.open = False

        def close(self, ignore_errors=True):
            self.open = False

    stream = FakeStream()
    _start_workers_over_a_fake_device(tts_module, monkeypatch, tts, stream)
    tts._play_queue.put(tts_module._PlayItem(
        data=np.ones(24000 * 2, dtype=np.float32), sample_rate=24000, device=None))

    assert writing.wait(timeout=5.0), "the playback worker never reached the device"
    tts.stop()

    assert not wrote_into_a_closed_stream, "stop() closed the stream mid-write"
    assert not stream.open, "playback was not actually stopped"


def test_stop_never_leaves_two_playback_workers_on_one_stream(tts_module, monkeypatch):
    """A second worker on one stream corrupts it, so a live one must never be forgotten."""
    import numpy as np
    import threading

    tts = tts_module.TextToSpeech()
    monkeypatch.setattr(tts_module, "_WORKER_JOIN_SECONDS", 0.1)
    writing = threading.Event()
    release = threading.Event()

    class WedgedStream:
        time = 0.0
        latency = 0.01

        def write(self, chunk):
            writing.set()
            release.wait(timeout=5.0)

        def abort(self, ignore_errors=True):
            pass

        def stop(self, ignore_errors=True):
            pass

        def close(self, ignore_errors=True):
            pass

    _start_workers_over_a_fake_device(tts_module, monkeypatch, tts, WedgedStream())
    original = tts._play_thread
    tts._play_queue.put(tts_module._PlayItem(
        data=np.ones(24000, dtype=np.float32), sample_rate=24000, device=None))
    assert writing.wait(timeout=5.0)

    tts.stop()
    assert tts._play_thread is original, "stop() forgot a worker that was still running"

    tts._ensure_say_workers()
    assert tts._play_thread is original, "a second worker was started over a live one"
    assert len(_live_playback_workers(tts)) == 1

    release.set()
    tts.close()


def test_saying_again_after_stop_still_queues(tts_module, monkeypatch):
    """stop() replaces the queues, so whatever say() uses must be the live pair."""
    tts = tts_module.TextToSpeech()
    tts.stop()

    monkeypatch.setattr(tts, "_require_loaded", lambda _name: 1)
    monkeypatch.setattr(tts_module, "_import_say_audio_deps", lambda: (None, None))
    monkeypatch.setattr(tts, "_ensure_say_workers", lambda: None)

    tts.say("Speaking again after a stop.")

    assert tts._say_queue.unfinished_tasks == 1
    assert tts.is_talking() is True


def _recording_synthesizer(tts_module, monkeypatch):
    """A loaded synthesizer that records what reached playback instead of playing it."""
    tts = tts_module.TextToSpeech()
    tts.load()
    played = []
    monkeypatch.setattr(
        tts, "_play_one", lambda item, sd, np: played.append(len(item.data))
    )
    return tts, played


def _wait_until(predicate, seconds=2.0):
    import time

    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_a_streamed_reply_after_a_cancelled_one_is_spoken_in_full(
    tts_module, fake_native, fake_stream, monkeypatch
):
    """A cancel must not cost the reply that follows it.

    The cancel is reported once, to whoever next asks for a chunk. If that is the
    following reply, it stops after its first sentence.
    """
    tts, played = _recording_synthesizer(tts_module, monkeypatch)

    interrupted = tts.say_stream()
    tts.push_text("The capital of Germany is Berlin. It is a large city. ")
    assert _wait_until(lambda: played), "the first reply never reached playback"
    tts.stop()
    interrupted.stop()
    played.clear()

    answer = tts.say_stream()
    tts.push_text("That would be London. It is a fantastic city. Full of history. ")
    tts.end_input()
    assert answer.wait(timeout=5.0), "the reply after a cancel never finished"

    assert len(played) == 3, f"only {len(played)} of 3 sentences were spoken"
    tts.close()


def test_stop_leaves_the_synthesizer_ready_for_another_streamed_reply(
    tts_module, fake_native, fake_stream, monkeypatch
):
    """`stop()` on its own must not leave the streaming side mid-reply."""
    tts, played = _recording_synthesizer(tts_module, monkeypatch)

    speech = tts.say_stream()
    tts.push_text("A reply that gets cut off. ")
    assert _wait_until(lambda: played), "the reply never reached playback"
    tts.stop()

    assert not tts.is_streaming, "stop() left a generation open inside the synthesizer"
    assert _wait_until(lambda: speech.finished), \
        "the pump was left polling a reply that had been abandoned"
    speech.stop()
