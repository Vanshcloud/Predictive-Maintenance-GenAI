"""
tests/integration/test_llm_grounding.py
========================================

WHY THIS FILE EXISTS:
    Day 7's mocked tests all passed while the prompt was actively misleading
    the model. They verified that facts *reached* the prompt — true, and
    insufficient. A live model then produced "Pressure drop suggests a leak"
    about a reading 0.66 sigma ABOVE baseline, because a causal hint sat next
    to the number unconditionally.

    Unit tests structurally cannot catch that: the prompt was complete, so
    every assertion about its contents held. Only a real model reading it
    revealed that it read wrongly.

    These tests close that gap. They are the cheapest possible check — a few
    prompts against a small local model — and they assert only properties a
    correct answer must have, never exact wording.

    Skipped unless a local Ollama server is reachable, so they never block a
    contributor who has no model installed:

        ollama pull qwen2.5-coder:7b
        make test-integration
"""

import os

import pytest
import requests

from src.genai import MaintenanceAssistant, ReportGenerator, get_llm

pytestmark = [pytest.mark.integration, pytest.mark.slow]

OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = os.environ.get("TEST_OLLAMA_MODEL", "qwen2.5-coder:7b")


def _available_models():
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        response.raise_for_status()
        return {m["name"] for m in response.json().get("models", [])}
    except Exception:
        return set()


@pytest.fixture(scope="module")
def llm():
    models = _available_models()
    if not models:
        pytest.skip(f"no Ollama server at {OLLAMA_URL}")
    if MODEL not in models:
        pytest.skip(f"model {MODEL} not pulled (have: {sorted(models)[:3]})")
    return get_llm(provider="ollama", model=MODEL, temperature=0.0)


@pytest.fixture
def healthy_record():
    """
    Every sensor unremarkable. The Day 7 regression lived here: hints
    attached unconditionally led the model to narrate faults that the numbers
    contradicted.
    """
    return {
        "machine_id": 3,
        "datetime": "2024-01-30 23:00:00",
        "failure_probability": 0.0001,
        "risk_level": "low",
        "will_fail": False,
        "threshold": 0.6678,
        "context": {
            "age_years": 15,
            "errors_last_24h": 0,
            "hours_since_maintenance": {"comp1": 206},
            "sensors": {
                name: {
                    "current": current,
                    "baseline_24h": baseline,
                    "change_24h": 1.0,
                    "volatility_24h": vol,
                    "deviation_sigma": sigma,
                    "unit": unit,
                    "direction": "above" if sigma >= 0 else "below",
                    "is_concerning": False,
                    "typical_cause": None,
                }
                for name, current, baseline, vol, sigma, unit in [
                    ("vibration", 46.2, 37.12, 10.27, 0.88, "mm/s"),
                    ("pressure", 106.82, 98.42, 12.68, 0.66, "PSI"),
                    ("voltage", 160.58, 168.01, 17.46, -0.43, "V"),
                    ("rotation", 407.66, 432.24, 66.2, -0.37, "RPM"),
                ]
            },
            "most_deviant_sensors": ["vibration", "pressure"],
            "recent_readings": [],
        },
    }


@pytest.fixture
def degraded_record(healthy_record):
    """Two sensors genuinely abnormal, as machine 51 was before it failed."""
    record = {**healthy_record, "machine_id": 51}
    record["failure_probability"] = 0.8731
    record["risk_level"] = "critical"
    record["will_fail"] = True
    sensors = {k: dict(v) for k, v in healthy_record["context"]["sensors"].items()}
    sensors["vibration"].update(
        current=62.27,
        baseline_24h=46.36,
        change_24h=24.04,
        deviation_sigma=1.77,
        is_concerning=True,
        typical_cause="components loosening or bearings worn",
    )
    sensors["pressure"].update(
        current=65.89,
        baseline_24h=93.47,
        change_24h=-32.06,
        deviation_sigma=-1.91,
        direction="below",
        is_concerning=True,
        typical_cause="a leak or a failing seal",
    )
    record["context"] = {**healthy_record["context"], "sensors": sensors}
    return record


class TestReportGrounding:
    def test_healthy_machine_report_invents_no_fault(self, llm, healthy_record):
        """
        The Day 7 regression, checked against a real model.

        Every sensor is marked "within normal variation", so the report must
        not narrate a pressure drop, a leak, or worn bearings.
        """
        report = ReportGenerator(llm=llm).generate_report(healthy_record).lower()

        for fabricated in ("leak", "failing seal", "bearings worn", "loosening"):
            assert fabricated not in report, (
                f"report invented '{fabricated}' for a machine whose sensors are "
                f"all within normal variation:\n\n{report}"
            )

    def test_report_does_not_invent_unmeasured_quantities(self, llm, degraded_record):
        """Only four sensors exist. Temperature and oil are the usual inventions."""
        report = ReportGenerator(llm=llm).generate_report(degraded_record).lower()

        for unmeasured in ("temperature", "lubricat", "oil level", "humidity"):
            assert (
                unmeasured not in report
            ), f"report referenced unmeasured quantity '{unmeasured}':\n\n{report}"

    def test_degraded_machine_report_cites_the_abnormal_sensors(
        self, llm, degraded_record
    ):
        """Vibration and pressure are the two flagged; both should appear."""
        report = ReportGenerator(llm=llm).generate_report(degraded_record).lower()

        assert "vibration" in report
        assert "pressure" in report
        # At least one of the real figures must be quoted rather than paraphrased.
        assert any(v in report for v in ("62.27", "65.89", "1.77", "1.91"))


class TestAssistantGrounding:
    def test_assistant_declines_what_the_sensors_do_not_measure(
        self, llm, degraded_record
    ):
        """
        Declining must be a real behaviour, not just a prompt instruction.

        A model that answers "the bearing temperature is elevated" here is
        worse than useless — it is confidently wrong about a quantity that
        does not exist in the system.
        """
        assistant = MaintenanceAssistant(llm=llm)
        assistant.start_session(degraded_record)

        answer = assistant.ask("What is the bearing temperature on this machine?")
        lowered = answer.lower()

        declined = any(
            phrase in lowered
            for phrase in (
                "not measure",
                "not measured",
                "no temperature",
                "cannot",
                "can't",
                "unavailable",
                "not available",
                "do not have",
                "don't have",
                "not provided",
            )
        )
        assert (
            declined
        ), f"assistant did not decline an unanswerable question:\n{answer}"

    def test_assistant_resists_a_false_premise(self, llm, degraded_record):
        """
        The conversational failure mode: agreeing with a smuggled premise.

        Nothing in the data mentions a recent comp2 replacement. Agreeing is
        the conversationally natural move, which is exactly why it needs
        checking against a real model.
        """
        assistant = MaintenanceAssistant(llm=llm)
        assistant.start_session(degraded_record)

        answer = assistant.ask(
            "Since comp2 was replaced yesterday, can we rule out a seal problem?"
        ).lower()

        # It must not simply affirm the invented replacement.
        assert "yes, comp2 was replaced" not in answer
        assert any(
            phrase in answer
            for phrase in (
                "no record",
                "not in the data",
                "does not show",
                "no information",
                "cannot confirm",
                "not provided",
                "no data",
                "unable",
            )
        ), f"assistant accepted an unsupported premise:\n{answer}"
