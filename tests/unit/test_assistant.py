"""
tests/unit/test_assistant.py
=============================
Tests for the multi-turn maintenance assistant.

The assistant's failure mode differs from the report's. A report confabulates
a *cause*; a conversation confabulates *continuity* — it agrees with a premise
the user smuggled into a follow-up, because agreeing is conversational. These
tests assert the structural defences against that: facts re-sent every turn,
history actually reaching the prompt, and a bounded transcript.

As with the report tests, they check what we control (prompt construction,
session state, error handling) rather than what the model writes.
"""

import pytest
from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, HumanMessage

from src.genai import MaintenanceAssistant
from src.utils.exceptions import LLMConnectionError, ReportGenerationError


@pytest.fixture
def record():
    """An explain_machine() record with 3 hours of trend."""
    return {
        "machine_id": 51,
        "datetime": "2024-10-30 20:00:00",
        "failure_probability": 0.8731,
        "risk_level": "critical",
        "will_fail": True,
        "threshold": 0.6678,
        "context": {
            "age_years": 17,
            "errors_last_24h": 3,
            "hours_since_maintenance": {"comp1": 412, "comp3": 88},
            "sensors": {
                "vibration": {
                    "current": 62.27,
                    "baseline_24h": 46.36,
                    "change_24h": 24.04,
                    "volatility_24h": 8.99,
                    "deviation_sigma": 1.77,
                    "unit": "mm/s",
                    "direction": "above",
                    "is_concerning": True,
                    "typical_cause": "components loosening or bearings worn",
                }
            },
            "most_deviant_sensors": ["vibration"],
            "recent_readings": [
                {"datetime": "2024-10-30 18:00:00", "vibration": 51.1},
                {"datetime": "2024-10-30 19:00:00", "vibration": 57.4},
                {"datetime": "2024-10-30 20:00:00", "vibration": 62.27},
            ],
        },
    }


class CapturingLLM(FakeListChatModel):
    """Records the messages each invocation received."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if not hasattr(self, "_seen"):
            object.__setattr__(self, "_seen", [])
        self._seen.append(list(messages))
        return super()._generate(messages, stop, run_manager, **kwargs)


def assistant_with(responses, **kwargs):
    return MaintenanceAssistant(llm=FakeListChatModel(responses=responses), **kwargs)


class TestSessionLifecycle:
    def test_asking_before_starting_a_session_is_an_error(self):
        with pytest.raises(ReportGenerationError, match="No session started"):
            assistant_with(["x"]).ask("Why is it at risk?")

    def test_incomplete_record_is_rejected(self):
        assistant = assistant_with(["x"])
        with pytest.raises(ReportGenerationError, match="missing"):
            assistant.start_session({"machine_id": 1})

    def test_session_pins_the_machine(self, record):
        assistant = assistant_with(["a"])
        assert assistant.machine_id is None

        assistant.start_session(record)
        assert assistant.machine_id == 51
        assert assistant.turn_count == 0

    def test_reset_clears_the_conversation_but_keeps_the_machine(self, record):
        assistant = assistant_with(["a", "b"])
        assistant.start_session(record)
        assistant.ask("first")
        assert assistant.turn_count == 1

        assistant.reset()
        assert assistant.turn_count == 0
        assert assistant.machine_id == 51  # still usable

    def test_end_session_drops_everything(self, record):
        assistant = assistant_with(["a"])
        assistant.start_session(record)
        assistant.end_session()

        assert assistant.machine_id is None
        with pytest.raises(ReportGenerationError, match="No session started"):
            assistant.ask("anything")

    def test_empty_question_is_rejected(self, record):
        assistant = assistant_with(["a"])
        assistant.start_session(record)
        with pytest.raises(ReportGenerationError, match="empty"):
            assistant.ask("   ")


class TestConversationMemory:
    def test_history_accumulates_in_order(self, record):
        assistant = assistant_with(["first answer", "second answer"])
        assistant.start_session(record)
        assistant.ask("first question")
        assistant.ask("second question")

        history = assistant.history
        assert [type(m) for m in history] == [
            HumanMessage,
            AIMessage,
            HumanMessage,
            AIMessage,
        ]
        assert history[0].content == "first question"
        assert history[1].content == "first answer"
        assert assistant.turn_count == 2

    def test_the_second_turn_can_see_the_first(self, record):
        """
        The whole point of a session.

        "And is it getting worse?" is meaningless unless the model can see
        what "it" referred to.
        """
        llm = CapturingLLM(responses=["Vibration is 1.77 sigma high.", "Yes, rising."])
        assistant = MaintenanceAssistant(llm=llm)
        assistant.start_session(record)

        assistant.ask("Why is this machine at risk?")
        assistant.ask("And is it getting worse?")

        second_call = llm._seen[1]
        rendered = "\n".join(str(m.content) for m in second_call)
        assert "Why is this machine at risk?" in rendered
        assert "Vibration is 1.77 sigma high." in rendered
        assert "And is it getting worse?" in rendered

    def test_facts_are_resent_every_turn(self, record):
        """
        The DATA block is repeated, never summarised into history.

        A model asked to recall a number from its own earlier prose will
        eventually recall it wrong. Re-sending means every turn is answered
        against the source.
        """
        llm = CapturingLLM(responses=["a", "b", "c"])
        assistant = MaintenanceAssistant(llm=llm)
        assistant.start_session(record)

        for question in ("one", "two", "three"):
            assistant.ask(question)

        for call in llm._seen:
            rendered = "\n".join(str(m.content) for m in call)
            assert "62.27" in rendered, "sensor reading missing from a later turn"
            assert "0.8731" in rendered, "probability missing from a later turn"

    def test_history_is_capped(self, record):
        """
        An unbounded transcript eventually crowds out the DATA block.

        At that point the model is improvising from its own earlier answers,
        which is exactly the failure this class exists to prevent.
        """
        assistant = assistant_with(["ans"] * 12, max_turns=3)
        assistant.start_session(record)
        for i in range(6):
            assistant.ask(f"question {i}")

        assert assistant.turn_count == 3
        assert len(assistant.history) == 6
        # The oldest questions are gone; the newest survive.
        contents = [m.content for m in assistant.history]
        assert "question 0" not in contents
        assert "question 5" in contents

    def test_history_is_returned_as_a_copy(self, record):
        """A caller mutating the returned list must not corrupt the session."""
        assistant = assistant_with(["a"])
        assistant.start_session(record)
        assistant.ask("q")

        assistant.history.clear()
        assert assistant.turn_count == 1


class TestGrounding:
    def test_trend_readings_reach_the_prompt(self, record):
        """
        "Has it been getting worse?" needs a trend, not a snapshot.

        Without these rows the question has no answer in the data, and the
        model is left to invent one.
        """
        llm = CapturingLLM(responses=["rising"])
        assistant = MaintenanceAssistant(llm=llm)
        assistant.start_session(record)
        assistant.ask("Has vibration been climbing?")

        rendered = "\n".join(str(m.content) for m in llm._seen[0])
        assert "RECENT HOURLY READINGS" in rendered
        for value in ("51.1", "57.4", "62.27"):
            assert value in rendered

    def test_prompt_tells_the_model_that_declining_is_acceptable(self, record):
        """
        Without this the model guesses rather than refuses.

        "The sensors do not measure temperature" must be presented as a
        complete answer, or a plausible fabrication takes its place.
        """
        llm = CapturingLLM(responses=["not measured"])
        assistant = MaintenanceAssistant(llm=llm)
        assistant.start_session(record)
        assistant.ask("What is the bearing temperature?")

        rendered = "\n".join(str(m.content) for m in llm._seen[0])
        assert "cannot answer" in rendered or "cannot tell you" in rendered
        assert "only voltage, rotation, pressure and vibration" in rendered.lower()

    def test_prompt_instructs_the_model_to_check_question_premises(self, record):
        """
        The Day 8 regression, pinned.

        Every other rule governs what the assistant *introduces*. A premise
        smuggled into the question — "since comp2 was replaced yesterday..." —
        is not something the assistant introduced, so nothing covered it, and
        a live model duly built its whole answer on the invented repair.
        """
        llm = CapturingLLM(responses=["no record of that"])
        assistant = MaintenanceAssistant(llm=llm)
        assistant.start_session(record)
        assistant.ask("Since comp2 was replaced yesterday, can we rule out a seal?")

        rendered = "\n".join(str(m.content) for m in llm._seen[0])
        assert "CHECK THE PREMISE" in rendered
        assert "unsupported premise" in rendered

    def test_sensor_verdicts_reach_the_prompt(self, record):
        """The Day 7 grounding fix must hold in conversation too."""
        llm = CapturingLLM(responses=["ok"])
        assistant = MaintenanceAssistant(llm=llm)
        assistant.start_session(record)
        assistant.ask("What is wrong?")

        rendered = "\n".join(str(m.content) for m in llm._seen[0])
        assert "ABNORMAL in the concerning direction" in rendered

    def test_session_without_trend_says_so(self, record, caplog):
        """A record built without history_hours cannot answer trend questions."""
        thin = dict(record)
        thin["context"] = dict(record["context"], recent_readings=[])

        assistant = assistant_with(["a"])
        assistant.start_session(thin)  # must not raise

        llm = CapturingLLM(responses=["a"])
        assistant2 = MaintenanceAssistant(llm=llm)
        assistant2.start_session(thin)
        assistant2.ask("q")
        rendered = "\n".join(str(m.content) for m in llm._seen[0])
        assert "RECENT HOURLY READINGS" not in rendered


class TestFailureHandling:
    class ExplodingLLM(FakeMessagesListChatModel):
        error: str = "boom"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            raise RuntimeError(self.error)

    def _assistant(self, record, error):
        llm = self.ExplodingLLM(responses=[])
        llm.error = error
        assistant = MaintenanceAssistant(llm=llm)
        assistant.start_session(record)
        return assistant

    def test_connectivity_failure_is_classified(self, record):
        assistant = self._assistant(record, "Connection refused")
        with pytest.raises(LLMConnectionError) as exc:
            assistant.ask("why?")
        assert "prediction itself is unaffected" in str(exc.value)

    def test_other_failures_are_generation_errors(self, record):
        assistant = self._assistant(record, "template variable missing")
        with pytest.raises(ReportGenerationError):
            assistant.ask("why?")

    def test_a_failed_turn_does_not_enter_the_history(self, record):
        """
        A question that errored must not pollute the transcript.

        Otherwise the next turn carries a question with no answer, and the
        model tries to make sense of the gap.
        """
        assistant = self._assistant(record, "Connection refused")
        with pytest.raises(LLMConnectionError):
            assistant.ask("why?")

        assert assistant.turn_count == 0
        assert assistant.history == []

    def test_empty_answer_is_an_error(self, record):
        assistant = assistant_with(["   "])
        assistant.start_session(record)
        with pytest.raises(ReportGenerationError, match="empty answer"):
            assistant.ask("why?")
