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
