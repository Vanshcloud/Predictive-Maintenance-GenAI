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

import math
import time
from typing import Any, Dict, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from pydantic import SecretStr

from config.settings import get_settings
from src.genai.prompts import QA_TEMPLATE, REPORT_TEMPLATE, format_machine_facts
from src.utils.exceptions import LLMConnectionError, ReportGenerationError
from src.utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_PROVIDERS = ("openai", "google", "ollama")


def _deadline_client_kwargs(deadline: float) -> Dict[str, Any]:
    """
    httpx client settings that end a request `deadline` s after it starts.

    An httpx timeout limits each socket operation, not the whole request, so
    a provider sending a byte every few seconds never trips it: a 1 s timeout
    let a trickling server hold the call for 5 s, and a server trickling its
    response *headers* held a 0.2 s call for 1.4 s. This gives every read and
    write on the connection only the time left before the deadline, so the
    headers, the body, and a silent provider are all cut off at the deadline.
    It reaches the connection through httpx's `trace` request extension
    rather than a custom transport, because passing `transport=` makes httpx
    ignore proxy environment variables.

    Not covered: DNS resolution, which has no timeout of its own, and the
    TCP connect and TLS handshake, which are each bounded by the ordinary
    per-operation `timeout` (itself `deadline`) rather than by the time left.

    Keep-alive is off because the clamp belongs to the connection: a reused
    connection would carry an earlier request's deadline into the next one.
    """
    import httpcore
    import httpx

    def clamp(op: Any, stop_at: float) -> Any:
        def bounded(*args: Any, timeout: Optional[float] = None) -> Any:
            left = stop_at - time.monotonic()
            if left > 0:
                try:
                    return op(
                        *args, timeout=left if timeout is None else min(timeout, left)
                    )
                except httpcore.TimeoutException:
                    if time.monotonic() < stop_at:
                        raise
            raise httpcore.ReadTimeout(
                f"Request timed out: still running after the {deadline:g}s "
                f"total deadline"
            )

        return bounded

    def start(request: Any) -> None:
        stop_at = time.monotonic() + deadline

        def trace(event: str, info: Dict[str, Any]) -> None:
            # The stream returned by the TCP connect, and its replacement
            # after a TLS upgrade: every later read and write goes through
            # one of them.
            if event.endswith((".connect_tcp.complete", ".start_tls.complete")):
                stream = info["return_value"]
                stream.read = clamp(stream.read, stop_at)
                stream.write = clamp(stream.write, stop_at)

        request.extensions["trace"] = trace

    return {
        "timeout": deadline,
        "event_hooks": {"request": [start]},
        "limits": httpx.Limits(max_keepalive_connections=0),
    }


def get_llm(
    provider: Optional[str] = None,
    temperature: float = 0.2,
    timeout: Optional[float] = None,
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
        timeout: The provider client's connect and read-inactivity timeout,
            with retries disabled so one attempt is the whole budget. For
            the httpx-based clients (OpenAI, langchain-ollama) it is also a
            total deadline on every socket read and write, headers included
            (see `_deadline_client_kwargs`). It bounds the local call only —
            a hosted provider may still finish (and bill) the generation
            after the connection is dropped.

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

            if timeout is not None:
                import httpx

                kwargs.update(
                    timeout=timeout,
                    max_retries=0,
                    http_client=httpx.Client(**_deadline_client_kwargs(timeout)),
                )
            return ChatOpenAI(
                model=kwargs.pop("model", None) or settings.OPENAI_MODEL,
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

            if timeout is not None:
                kwargs.update(timeout=timeout, max_retries=0)
            google_llm: BaseChatModel = ChatGoogleGenerativeAI(
                model=kwargs.pop("model", None) or settings.GOOGLE_MODEL,
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

        if timeout is not None:
            # langchain-ollama configures its httpx client via client_kwargs;
            # the deprecated community class takes `timeout` directly.
            if "client_kwargs" in ChatOllama.model_fields:
                kwargs["client_kwargs"] = _deadline_client_kwargs(timeout)
            else:  # an int field: round up, never down to "no time"
                kwargs["timeout"] = math.ceil(timeout)
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
        timeout: Optional[float] = None,
        **llm_kwargs: Any,
    ) -> None:
        """
        Args:
            llm: An explicit chat model. Injectable so tests can pass a fake
                one — this class must be testable without a network call or
                an API key, or it will simply not be tested.
            provider: Ignored when `llm` is given.
            timeout: Total seconds one generation may run locally, also
                passed to the provider client (see `get_llm`).
        """
        self.timeout = timeout
        self.llm = (
            llm
            if llm is not None
            else get_llm(provider, temperature, timeout=timeout, **llm_kwargs)
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
            report = self._run(self.report_chain, {"machine_facts": facts})
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
            answer = self._run(
                self.qa_chain, {"machine_facts": facts, "question": question.strip()}
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
            summary = self._run(
                self.qa_chain,
                {
                    "machine_facts": facts,
                    "question": (
                        "Summarise the state of this fleet for a maintenance "
                        "supervisor planning the next shift. Say which machines "
                        "need attention first and why, citing the readings given. "
                        "If no machine is alerting, say the fleet is healthy."
                    ),
                },
            )
        except Exception as e:
            raise self._wrap(e, "fleet summary generation")

        if not summary or not summary.strip():
            raise ReportGenerationError("The model returned an empty fleet summary.")
        return summary.strip()

    # ------------------------------------------------------------------

    def _run(self, chain: Any, inputs: Dict[str, Any]) -> str:
        """
        Run a chain within `self.timeout` seconds in total.

        A provider timeout limits each read, so a model that is still
        producing tokens never trips it. Streaming lets the total be checked
        between chunks, and closing the stream drops the connection. Python
        cannot safely kill a thread, so stopping between chunks is the only
        place this can happen. The httpx-based clients also cut the socket
        at the deadline (see `_deadline_client_kwargs`). For Google and the
        deprecated community Ollama client this check is the only total
        limit: a provider that goes silent is stopped by its per-read
        timeout instead, so one call can run past `timeout`, and one that
        keeps sending bytes that never form a token is not stopped at all.

        The parser is applied once to the merged message rather than
        streamed: LangChain drains the rest of the upstream stream when a
        sequence ending in a parser is closed, which kept the connection
        reading for as long as the provider kept sending.
        """
        if self.timeout is None:
            result: str = chain.invoke(inputs)
            return result
        stop_at = time.monotonic() + self.timeout
        merged = None
        head: Any = RunnableSequence(*chain.steps[:-1])
        stream = head.stream(inputs)
        try:
            for part in stream:
                merged = part if merged is None else merged + part
                if time.monotonic() > stop_at:
                    raise TimeoutError(
                        f"Generation timed out: still running after the "
                        f"{self.timeout:g}s total deadline"
                    )
        finally:
            stream.close()
        if merged is None:
            return ""
        parsed: str = chain.last.invoke(merged)
        return parsed

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
