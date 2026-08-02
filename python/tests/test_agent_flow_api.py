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
