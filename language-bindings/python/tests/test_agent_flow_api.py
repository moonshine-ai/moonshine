"""Tests for the AgentFlow catch-all handler.

These drive the public surface with no microphone, no synthesizer and no
embedding model, so utterances route through substring matching and nothing
has to be downloaded. `speak_with` records what would have been said.
"""

import pytest


@pytest.fixture
def agent():
    moonshine_voice = pytest.importorskip("moonshine_voice")

    return (
        moonshine_voice.AgentFlow()
        .microphone(False)
        .speech(False)
        .use_embeddings(False)
    )


def test_otherwise_receives_speech_that_matched_nothing(agent):
    leftovers = []
    agent.otherwise(leftovers.append)

    assert agent.handle_utterance("the weather is nice today") is True
    assert leftovers == ["the weather is nice today"]


def test_otherwise_does_not_see_triggers_or_answers(agent):
    spoken = []
    leftovers = []
    agent.speak_with(spoken.append)
    agent.otherwise(leftovers.append)

    def setup(d):
        yield d.ask("Name?")

    agent.listen_for("start setup", setup)

    # A trigger phrase belongs to the flow it starts, and the answer that
    # follows belongs to the prompt waiting for it.
    agent.handle_utterance("start setup")
    agent.handle_utterance("Alice")

    assert leftovers == []
    assert "Name?" in spoken


def test_the_built_in_cancel_stops_the_active_flow(agent):
    agent.speak_with([].append)
    finished = []

    def setup(d):
        yield d.ask("Name?")
        finished.append("done")

    agent.listen_for("start setup", setup)

    agent.handle_utterance("start setup")
    assert agent.is_active is True
    agent.handle_utterance("cancel")

    assert finished == [], "cancel should abandon the flow"
    assert agent.is_active is False


def test_the_built_in_start_over_restarts_the_active_flow(agent):
    agent.speak_with([].append)
    starts = []

    def setup(d):
        starts.append("start")
        yield d.ask("Name?")

    agent.listen_for("begin", setup)

    agent.handle_utterance("begin")
    agent.handle_utterance("start over")

    assert starts == ["start", "start"]
    assert agent.is_active is True


def test_the_built_in_globals_do_not_claim_speech_outside_a_flow(agent):
    """The built-ins only apply to a flow in progress.

    With nothing active there is no flow for either phrase to act on, so
    claiming them here would lose a line of dictation.
    """
    leftovers = []
    agent.otherwise(leftovers.append)

    agent.handle_utterance("cancel")
    agent.handle_utterance("start over")
    agent.handle_utterance("cancel my subscription tomorrow")

    assert leftovers == ["cancel", "start over", "cancel my subscription tomorrow"]


def test_registering_a_built_in_phrase_with_always_makes_it_live(agent):
    leftovers = []
    cancels = []
    agent.otherwise(leftovers.append)
    agent.always("cancel", lambda d: cancels.append("cancel"))

    assert agent.handle_utterance("cancel") is True

    assert cancels == ["cancel"]
    assert leftovers == []


def test_otherwise_silences_the_didnt_get_that_cue(agent):
    beeps = []
    agent._error_beep_fn = lambda: beeps.append("error")

    agent.handle_utterance("nothing matches this")
    assert beeps == ["error"], "without a handler the cue still fires"

    agent.otherwise(lambda text: None)
    agent.handle_utterance("nothing matches this either")
    assert beeps == ["error"], "a registered handler makes the cue wrong"


def test_otherwise_returns_the_agent_for_chaining(agent):
    assert agent.otherwise(lambda text: None) is agent


def test_a_failing_otherwise_handler_is_reported_not_raised(agent):
    errors = []
    agent.on_error(errors.append)
    agent.otherwise(lambda text: 1 / 0)

    assert agent.handle_utterance("boom") is True
    assert len(errors) == 1
    assert "ZeroDivisionError" in str(errors[0])


def test_say_stream_speaks_the_pieces_a_flow_yields(agent):
    spoken = []
    agent.speak_with(spoken.append)

    def answer(d):
        yield d.say_stream(iter(["The answer ", "is forty two."]))

    agent.listen_for("ask the oracle", answer)
    agent.handle_utterance("ask the oracle")

    assert spoken == ["The answer is forty two."]


def test_a_failing_say_stream_source_is_reported_not_raised(agent):
    errors = []
    agent.on_error(errors.append)
    agent.speak_with([].append)

    def half_written():
        yield "This much arrived"
        raise RuntimeError("the model hung up")

    def answer(d):
        yield d.say_stream(half_written())

    agent.listen_for("ask the oracle", answer)
    agent.handle_utterance("ask the oracle")

    assert len(errors) == 1
    assert "the model hung up" in str(errors[0])


def test_say_stream_outside_a_flow_speaks_as_one_utterance(agent):
    spoken = []
    agent.speak_with(spoken.append)

    with agent.say_stream() as push:
        push("Downloading ")
        push("the model.")

    assert spoken == ["Downloading the model."]


class _FakeStreamHandle:
    """Stand-in for :class:`moonshine_voice.tts.SpeechInProgress`.

    As narrow as the real thing on purpose: you can wait for the reply or stop it,
    and that is all. A fake that also accepted text would have let the bug these
    tests guard against straight through.
    """

    def __init__(self, events):
        self._events = events

    def wait(self, timeout=None):
        self._events.append("wait")
        return True

    def stop(self):
        self._events.append("stop")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()


class _FakeSynthesizer:
    """The slice of :class:`moonshine_voice.TextToSpeech` that streamed speech uses."""

    def __init__(self):
        self.events = []
        self.pushed = []

    def say_stream(self):
        self.events.append("say_stream")
        return _FakeStreamHandle(self.events)

    def push_text(self, text):
        self.events.append("push_text")
        self.pushed.append(text)

    def end_input(self):
        self.events.append("end_input")


def _agent_with(tts):
    moonshine_voice = pytest.importorskip("moonshine_voice")
    return (
        moonshine_voice.AgentFlow()
        .microphone(False)
        .use_embeddings(False)
        .use_text_to_speech(tts)
    )


def test_say_stream_pushes_text_into_the_synthesizer_not_the_handle():
    """A streamed reply has to reach a real synthesizer, which takes text on itself.

    Every other ``say_stream`` test above configures ``speech(False)``, so they run the
    buffering fallback and say nothing about the branch that runs once audio is
    attached. That is how this path came to call three methods the handle does not
    have, raising ``AttributeError`` on the first token of the first reply.
    """
    tts = _FakeSynthesizer()

    with _agent_with(tts).say_stream() as push:
        push("The answer ")
        push("")
        push("is forty two.")

    assert tts.pushed == ["The answer ", "is forty two."]
    assert tts.events == [
        "say_stream", "push_text", "push_text", "end_input", "wait", "stop",
    ]


def test_a_streamed_reply_is_stopped_even_when_the_producer_raises():
    """The handle is released on the way out, or the next reply talks over this one."""
    tts = _FakeSynthesizer()

    with pytest.raises(RuntimeError):
        with _agent_with(tts).say_stream() as push:
            push("This much arrived")
            raise RuntimeError("the model hung up")

    assert tts.events == ["say_stream", "push_text", "stop"]


def test_the_streaming_handle_takes_no_text():
    """Pins the contract the fake above imitates.

    If :class:`SpeechInProgress` ever grows a ``push_text``, the fake stops standing for
    the real thing and the regression test above quietly loses its teeth.
    """
    pytest.importorskip("moonshine_voice")
    from moonshine_voice.tts import SpeechInProgress

    public = {name for name in vars(SpeechInProgress) if not name.startswith("_")}
    assert public == {"wait", "stop", "finished"}
