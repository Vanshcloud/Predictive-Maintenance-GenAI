"""
src/genai/chains.py — LLM Chains and Provider Selection
=======================================================

WHY THIS FILE EXISTS:
    Turns a grounded prediction record into readable prose, behind an
    interface that does not care which LLM is answering. The provider is a
    configuration value — OpenAI, Google, or a local Ollama model — because
    tying a portfolio project to one vendor's API key means it stops working
    for anyone who does not have one.

HOW IT WORKS:
    `get_llm()` builds a chat model from settings, importing the provider
    package lazily so a missing optional dependency produces a clear message
    rather than an ImportError at module load. `ReportGenerator` composes
    prompt -> model -> string parser and adds the thing that matters
    operationally: **a failure to generate a report must never take the
    prediction down with it.**

    That asymmetry is deliberate. The prediction is the safety-critical
    output — it is what decides whether a technician is dispatched. The
    report is a convenience layer over it. An LLM outage, a rate limit, or an
    expired key should degrade the system to "prediction available, narrative
    unavailable", never to "no answer". Every path here either returns a
    report or raises `LLMConnectionError` / `ReportGenerationError`, both of
    which callers above can catch precisely.
"""

from typing import Any, Dict, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from pydantic import SecretStr

from config.settings import get_settings
from src.genai.prompts import QA_TEMPLATE, REPORT_TEMPLATE, format_machine_facts
from src.utils.exceptions import LLMConnectionError, ReportGenerationError
from src.utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_PROVIDERS = ("openai", "google", "ollama")


def get_llm(
    provider: Optional[str] = None,
    temperature: float = 0.2,
    **kwargs: Any,
) -> BaseChatModel:
    """
    Build a chat model for the configured provider.

    Provider packages are imported lazily. `langchain-google-genai` and
    `langchain-ollama` are optional extras, and importing them at module load
    would make this whole module unimportable for anyone who installed only
    the base requirements.

    Args:
        provider: "openai", "google", or "ollama". Defaults to whichever has
            credentials configured, preferring OpenAI, then Google, then the
            keyless local Ollama.
        temperature: Low by default. This is a maintenance report, not
            creative writing — variation between runs on identical data would
            undermine trust in it.

    Raises:
        LLMConnectionError: unknown provider, missing package, or missing key.
    """
    settings = get_settings()
    provider = (provider or _default_provider(settings)).lower()

    if provider not in SUPPORTED_PROVIDERS:
        raise LLMConnectionError(
            f"Unknown LLM provider {provider!r}. "
            f"Supported: {', '.join(SUPPORTED_PROVIDERS)}."
        )

    try:
        if provider == "openai":
            if not settings.OPENAI_API_KEY:
                raise LLMConnectionError(
                    "OPENAI_API_KEY is not set. Add it to .env, or use "
                    "provider='ollama' to run a local model with no key."
                )
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=settings.OPENAI_MODEL,
                # SecretStr is what the client expects, and it keeps the key
                # from appearing in a repr() or a traceback.
                api_key=SecretStr(settings.OPENAI_API_KEY),
                temperature=temperature,
                **kwargs,
            )

        if provider == "google":
            if not settings.GOOGLE_API_KEY:
                raise LLMConnectionError("GOOGLE_API_KEY is not set.")
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
            except ImportError as e:
                raise LLMConnectionError(
                    "Google provider requires `pip install langchain-google-genai`."
                ) from e

            google_llm: BaseChatModel = ChatGoogleGenerativeAI(
                model=settings.GOOGLE_MODEL,
                google_api_key=settings.GOOGLE_API_KEY,
                temperature=temperature,
                **kwargs,
            )
            return google_llm

        # Ollama: local, no key. The zero-cost path, and the reason this
        # project does not require anyone to hold an API key.
        #
        # langchain-ollama is the maintained home for this class; the
        # langchain-community copy is deprecated and slated for removal in
        # 1.0. Prefer the new one, fall back so the base requirements still
        # work without the optional extra.
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            from langchain_community.chat_models import ChatOllama

        ollama_llm: BaseChatModel = ChatOllama(
            model=kwargs.pop("model", None) or settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=temperature,
            **kwargs,
        )
        return ollama_llm

    except LLMConnectionError:
        raise
    except Exception as e:
        raise LLMConnectionError(
            f"Could not initialise the {provider} provider: {e}"
        ) from e


def _default_provider(settings: Any) -> str:
    """Pick a provider from whatever credentials exist."""
    if settings.OPENAI_API_KEY:
        return "openai"
    if settings.GOOGLE_API_KEY:
        return "google"
    # Keyless fallback. It may not be running, which surfaces at call time
    # as LLMConnectionError — the correct place for a connectivity failure.
    return "ollama"


class ReportGenerator:
    """Generates maintenance reports and answers questions from predictions."""

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        provider: Optional[str] = None,
        temperature: float = 0.2,
        **llm_kwargs: Any,
    ) -> None:
        """
        Args:
            llm: An explicit chat model. Injectable so tests can pass a fake
                one — this class must be testable without a network call or
                an API key, or it will simply not be tested.
            provider: Ignored when `llm` is given.
        """
        self.llm = (
            llm if llm is not None else get_llm(provider, temperature, **llm_kwargs)
        )
        self.report_chain = REPORT_TEMPLATE | self.llm | StrOutputParser()
        self.qa_chain = QA_TEMPLATE | self.llm | StrOutputParser()

    def generate_report(self, record: Dict[str, Any]) -> str:
        """
        Write a maintenance report for one prediction.

        Args:
            record: A `Predictor.explain_machine()` record. A plain
                `predict_machine()` record also works but produces a thinner
                report, because there is no evidence to cite.

        Raises:
            ReportGenerationError: if the record is unusable or the model
                returns nothing.
            LLMConnectionError: if the provider cannot be reached.
        """
        self._require_fields(record)
        facts = format_machine_facts(record)

        try:
            report: str = self.report_chain.invoke({"machine_facts": facts})
        except Exception as e:
            raise self._wrap(e, f"report generation for machine {record['machine_id']}")

        if not report or not report.strip():
            raise ReportGenerationError(
                "The model returned an empty report for machine "
                f"{record['machine_id']}."
            )

        logger.info(
            f"Generated report for machine {record['machine_id']} "
            f"({len(report)} chars, risk={record['risk_level']})"
        )
        return report.strip()

    def answer_question(self, record: Dict[str, Any], question: str) -> str:
        """Answer a free-form question about one machine, grounded in its data."""
        self._require_fields(record)
        if not question or not question.strip():
            raise ReportGenerationError("Question is empty.")

        facts = format_machine_facts(record)
        try:
            answer: str = self.qa_chain.invoke(
                {"machine_facts": facts, "question": question.strip()}
            )
        except Exception as e:
            raise self._wrap(
                e, f"question answering for machine {record['machine_id']}"
            )

        if not answer or not answer.strip():
            raise ReportGenerationError("The model returned an empty answer.")
        return answer.strip()

    def generate_fleet_summary(self, records: list, limit: int = 10) -> str:
        """
        Summarise the highest-risk machines in one narrative.

        Only alerting machines are described individually — a summary that
        recites 100 healthy machines is one nobody reads.
        """
        if not records:
            raise ReportGenerationError("No predictions supplied to summarise.")

        alerting = [r for r in records if r.get("will_fail")]
        ranked = sorted(
            alerting or records,
            key=lambda r: r["failure_probability"],
            reverse=True,
        )[:limit]

        blocks = [format_machine_facts(r) for r in ranked]
        facts = (
            f"Fleet size assessed: {len(records)} machines\n"
            f"Machines at or above the alert threshold: {len(alerting)}\n\n"
            + "\n\n---\n\n".join(blocks)
        )

        try:
            summary: str = self.qa_chain.invoke(
                {
                    "machine_facts": facts,
                    "question": (
                        "Summarise the state of this fleet for a maintenance "
                        "supervisor planning the next shift. Say which machines "
                        "need attention first and why, citing the readings given. "
                        "If no machine is alerting, say the fleet is healthy."
                    ),
                }
            )
        except Exception as e:
            raise self._wrap(e, "fleet summary generation")

        if not summary or not summary.strip():
            raise ReportGenerationError("The model returned an empty fleet summary.")
        return summary.strip()

    # ------------------------------------------------------------------

    @staticmethod
    def _require_fields(record: Dict[str, Any]) -> None:
        required = ("machine_id", "failure_probability", "risk_level", "threshold")
        missing = [f for f in required if f not in record]
        if missing:
            raise ReportGenerationError(
                f"Prediction record is missing required field(s): {missing}. "
                "Pass a record from Predictor.explain_machine() or predict_machine()."
            )

    @staticmethod
    def _wrap(error: Exception, what: str) -> Exception:
        """
        Classify a provider failure as connectivity or generation.

        Callers above need the distinction: a connectivity failure is
        retryable and worth reporting as "the LLM is down", while a
        generation failure usually means the input was wrong.
        """
        text = str(error).lower()
        connectivity = (
            "connection",
            "timeout",
            "timed out",
            "unreachable",
            "rate limit",
            "429",
            "503",
            "authentication",
            "api key",
            "unauthorized",
            # A model the server does not have is a provider-availability
            # problem, not a bad prompt — the caller should degrade to
            # "prediction available, report unavailable" rather than treat it
            # as a malformed request.
            "404",
            "model is not found",
            "model not found",
        )
        if any(marker in text for marker in connectivity):
            logger.error(f"LLM unreachable during {what}: {error}")
            return LLMConnectionError(
                f"LLM provider unavailable during {what}: {error}. "
                "The prediction itself is unaffected."
            )
        logger.error(f"Failed during {what}: {error}")
        return ReportGenerationError(f"Failed during {what}: {error}")
