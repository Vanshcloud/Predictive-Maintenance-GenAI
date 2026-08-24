"""
tests/unit/test_genai.py
========================
Tests for prompt construction, provider selection, and report generation.

These test what we control — the prompt, the grounding, the error handling —
and deliberately not what the LLM writes. Asserting on model output would be
testing OpenAI's weights, which are not ours, not deterministic, and not the
thing that breaks. What breaks is a fact silently dropped from the prompt, or
an outage taking a prediction down with it.

Every test runs against a fake chat model: no API key, no network, no cost.
"""

import pytest
from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
    FakeMessagesListChatModel,
)

from src.genai import ReportGenerator, format_machine_facts, get_llm
from src.utils.exceptions import LLMConnectionError, ReportGenerationError


@pytest.fixture
def record():
    """A full explain_machine() record for an alerting machine."""
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
            "hours_since_maintenance": {"comp1": 412, "comp2": 9999, "comp3": 88},
            "sensors": {
                # Both genuinely abnormal in the concerning direction.
                "vibration": {
                    "current": 61.2,
                    "baseline_24h": 42.8,
                    "change_24h": 18.4,
                    "volatility_24h": 9.1,
                    "deviation_sigma": 2.02,
                    "unit": "mm/s",
                    "direction": "above",
                    "is_concerning": True,
                    "typical_cause": "components loosening or bearings worn",
                },
                "pressure": {
                    "current": 78.5,
                    "baseline_24h": 99.1,
                    "change_24h": -20.6,
                    "volatility_24h": 11.3,
                    "deviation_sigma": -1.82,
                    "unit": "PSI",
                    "direction": "below",
                    "is_concerning": True,
                    "typical_cause": "a leak or a failing seal",
                },
                # Deviating, but in the harmless direction — rotation ABOVE
                # baseline is not "bearing drag". Its causal hint must be
                # withheld; the Day 7 live-model run showed a model happily
                # restating it as an observation otherwise.
                "rotation": {
                    "current": 470.0,
                    "baseline_24h": 448.0,
                    "change_24h": 22.0,
                    "volatility_24h": 30.0,
                    "deviation_sigma": 0.73,
                    "unit": "RPM",
                    "direction": "above",
                    "is_concerning": False,
                    "typical_cause": None,
                },
            },
            "most_deviant_sensors": ["vibration", "pressure", "rotation"],
        },
    }


@pytest.fixture
def healthy_record(record):
    r = dict(record)
    r.update(
        machine_id=12, failure_probability=0.0004, risk_level="low", will_fail=False
    )
    return r


def fake_llm(response="ASSESSMENT\nMachine 51 is at critical risk."):
    """A chat model that always returns the same text."""
    return FakeListChatModel(responses=[response])


class TestFactFormatting:
    """
    The prompt is the product here.

    A number missing from the formatted facts is a number the LLM cannot cite
    and will be tempted to invent, so these assert that every measurement in
    the record survives into the prompt.
    """

    def test_every_measurement_reaches_the_prompt(self, record):
        facts = format_machine_facts(record)

        for value in ("61.2", "42.8", "18.4", "9.1", "2.02"):  # vibration
            assert value in facts, f"vibration figure {value} missing from prompt"
        for value in ("78.5", "99.1", "20.6", "11.3"):  # pressure
            assert value in facts, f"pressure figure {value} missing from prompt"

        assert "0.8731" in facts and "0.6678" in facts
        assert "CRITICAL" in facts
        assert "17 years" in facts
        assert "Errors logged in last 24h: 3" in facts

    def test_units_travel_with_their_numbers(self, record):
        """A bare number is unusable; the model must be able to quote units."""
        facts = format_machine_facts(record)
        assert "mm/s" in facts
        assert "PSI" in facts

    def test_maintenance_sentinel_is_rendered_as_words(self, record):
        """
        9999 is the pipeline's "never maintained" sentinel, not a duration.

        Passing it through as a number invites the model to write "maintained
        9999 hours ago", which is both wrong and absurd.
        """
        facts = format_machine_facts(record)
        assert "no record" in facts
        assert "9999" not in facts
        assert "412 h" in facts

    def test_alerting_state_is_explicit(self, record, healthy_record):
        """The model should not have to compare floats to infer the decision."""
        assert "ABOVE — alerting" in format_machine_facts(record)
        assert "below — not alerting" in format_machine_facts(healthy_record)

    def test_most_deviant_sensor_is_listed_first(self, record):
        """Ordering is a hint about what matters; vibration is the 2.02-sigma one."""
        facts = format_machine_facts(record)
        assert facts.index("vibration:") < facts.index("pressure:")

    def test_unmeasured_quantities_are_ruled_out(self, record):
        """
        The prompt states which sensors exist.

        Without this the model happily discusses temperature and lubrication,
        neither of which this equipment measures.
        """
        facts = format_machine_facts(record)
        assert "only voltage, rotation, pressure and vibration" in facts

    def test_causal_hint_is_withheld_when_the_reading_is_not_concerning(self, record):
        """
        The Day 7 regression, pinned.

        A live model was given "pressure 106.8 PSI, 0.66 sigma ABOVE baseline"
        alongside the unconditional hint "concerning when it drops; typically
        indicates a leak" — and wrote "Pressure drop suggests a leak". The hint
        read as an observation. Causal explanations are now attached only when
        the deviation is in the direction that actually matters.
        """
        facts = format_machine_facts(record)

        # rotation is 0.73 sigma ABOVE baseline: high rotation is not drag.
        rotation_block = facts[facts.index("rotation:") :]
        rotation_line = rotation_block.split("\n")[1]
        assert "within normal variation" in rotation_line
        assert "bearing drag" not in rotation_block.split("vibration")[0]

        # vibration IS abnormal in the concerning direction, so it keeps its hint.
        vibration_block = facts[facts.index("vibration:") :]
        assert "ABNORMAL in the concerning direction" in vibration_block.split("\n")[1]
        assert "components loosening" in vibration_block

    def test_report_prompt_forbids_actions_that_contradict_the_risk_level(self):
        """
        The other half of the Day 7 regression.

        The same run reported "risk LOW, no threshold breached" and then
        recommended inspecting bearings and seals within 24 hours. The template
        now states that the action must match the risk level.
        """
        from src.genai.prompts import MAINTENANCE_EXPERT_SYSTEM, REPORT_TEMPLATE

        template_text = str(REPORT_TEMPLATE)
        assert "MUST match the" in template_text
        assert "routine monitoring" in template_text
        assert "ABNORMAL" in MAINTENANCE_EXPERT_SYSTEM

    def test_missing_context_still_formats(self):
        """A bare predict_machine() record must not crash the formatter."""
        facts = format_machine_facts(
            {
                "machine_id": 3,
                "datetime": "2024-01-01 00:00:00",
                "failure_probability": 0.1,
                "risk_level": "low",
                "will_fail": False,
                "threshold": 0.6678,
            }
        )
        assert "Machine ID: 3" in facts


class TestProviderSelection:
    def test_unknown_provider_is_rejected(self):
        with pytest.raises(LLMConnectionError, match="Unknown LLM provider"):
            get_llm(provider="anthropic-but-typo")

    def test_missing_openai_key_names_the_keyless_alternative(self, monkeypatch):
        """
        The error must be actionable.

        "OPENAI_API_KEY is not set" alone leaves a reader stuck; pointing at
        Ollama tells them how to run the project with no key at all.
        """
        from config import settings as settings_module

        settings_module.get_settings.cache_clear()
        monkeypatch.setenv("OPENAI_API_KEY", "")

        with pytest.raises(LLMConnectionError) as exc:
            get_llm(provider="openai")
        assert "ollama" in str(exc.value).lower()
        settings_module.get_settings.cache_clear()

    def test_google_provider_reports_the_missing_package(self, monkeypatch):
        """langchain-google-genai is an optional extra, not a hard dependency."""
        from config import settings as settings_module

        settings_module.get_settings.cache_clear()
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key-not-real")

        with pytest.raises(LLMConnectionError) as exc:
            get_llm(provider="google")
        message = str(exc.value).lower()
        assert "langchain-google-genai" in message or "google" in message
        settings_module.get_settings.cache_clear()


class TestReportGeneration:
    def test_generates_a_report(self, record):
        generator = ReportGenerator(llm=fake_llm())
        report = generator.generate_report(record)

        assert report.startswith("ASSESSMENT")
        assert "51" in report

    def test_the_prompt_actually_carries_the_facts(self, record):
        """
        End-to-end grounding check.

        Captures what the chain sent to the model and asserts the readings are
        in it. If a refactor drops the context block, the report would still
        generate — fluently, and about nothing.
        """
        captured = {}

        class CapturingLLM(FakeListChatModel):
            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                captured["prompt"] = "\n".join(m.content for m in messages)
                return super()._generate(messages, stop, run_manager, **kwargs)

        generator = ReportGenerator(llm=CapturingLLM(responses=["ASSESSMENT ok"]))
        generator.generate_report(record)

        assert "61.2" in captured["prompt"]
        assert "mm/s" in captured["prompt"]
        assert "0.8731" in captured["prompt"]
        assert "senior reliability engineer" in captured["prompt"]

    def test_incomplete_record_is_rejected(self):
        generator = ReportGenerator(llm=fake_llm())
        with pytest.raises(ReportGenerationError, match="missing required field"):
            generator.generate_report({"machine_id": 1})

    def test_empty_model_output_is_an_error(self, record):
        """Silence is a failure, not a report."""
        generator = ReportGenerator(llm=fake_llm(response="   "))
        with pytest.raises(ReportGenerationError, match="empty report"):
            generator.generate_report(record)


class TestGracefulDegradation:
    """
    An LLM failure must never take the prediction with it.

    The prediction decides whether a technician is dispatched; the report is a
    convenience over it. These assert the failure is classified correctly so
    the API can return the prediction and note the report is unavailable.
    """

    class ExplodingLLM(FakeMessagesListChatModel):
        error: str = "boom"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            raise RuntimeError(self.error)

    def _generator(self, error):
        llm = self.ExplodingLLM(responses=[])
        llm.error = error
        return ReportGenerator(llm=llm)

    @pytest.mark.parametrize(
        "error",
        [
            "Connection refused",
            "Request timed out",
            "Rate limit exceeded",
            "Error code: 429",
            "Incorrect API key provided",
        ],
    )
    def test_connectivity_failures_raise_llm_connection_error(self, record, error):
        generator = self._generator(error)
        with pytest.raises(LLMConnectionError) as exc:
            generator.generate_report(record)
        assert "prediction itself is unaffected" in str(exc.value)

    def test_other_failures_raise_report_generation_error(self, record):
        """A malformed-input failure is not an outage and must not look like one."""
        generator = self._generator("template variable 'foo' was not provided")
        with pytest.raises(ReportGenerationError):
            generator.generate_report(record)

    def test_both_error_types_are_catchable_as_one_base(self, record):
        """The API layer should be able to catch the whole family at once."""
        from src.utils.exceptions import PredMaintenanceError

        generator = self._generator("Connection refused")
        with pytest.raises(PredMaintenanceError):
            generator.generate_report(record)


class TestQuestionAnswering:
    def test_answers_a_question(self, record):
        generator = ReportGenerator(llm=fake_llm(response="Vibration is 2 sigma high."))
        answer = generator.answer_question(record, "Why is this machine at risk?")
        assert "Vibration" in answer

    def test_empty_question_is_rejected(self, record):
        generator = ReportGenerator(llm=fake_llm())
        with pytest.raises(ReportGenerationError, match="empty"):
            generator.answer_question(record, "   ")


class TestFleetSummary:
    def test_summarises_only_the_alerting_machines(self, record, healthy_record):
        """A summary reciting 100 healthy machines is one nobody reads."""
        captured = {}

        class CapturingLLM(FakeListChatModel):
            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                captured["prompt"] = "\n".join(m.content for m in messages)
                return super()._generate(messages, stop, run_manager, **kwargs)

        generator = ReportGenerator(llm=CapturingLLM(responses=["Machine 51 first."]))
        generator.generate_fleet_summary([record, healthy_record, healthy_record])

        assert "Fleet size assessed: 3 machines" in captured["prompt"]
        assert "at or above the alert threshold: 1" in captured["prompt"]
        assert "Machine ID: 51" in captured["prompt"]
        assert "Machine ID: 12" not in captured["prompt"]

    def test_healthy_fleet_still_summarises(self, healthy_record):
        """With nothing alerting, describe the top-ranked machines anyway."""
        generator = ReportGenerator(llm=fake_llm(response="Fleet is healthy."))
        summary = generator.generate_fleet_summary([healthy_record, healthy_record])
        assert "healthy" in summary.lower()

    def test_empty_fleet_is_rejected(self):
        generator = ReportGenerator(llm=fake_llm())
        with pytest.raises(ReportGenerationError, match="No predictions"):
            generator.generate_fleet_summary([])
