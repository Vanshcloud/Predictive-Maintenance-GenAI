"""
src/genai/assistant.py — Conversational Maintenance Q&A
========================================================

WHY THIS FILE EXISTS:
    A report answers the question the report's author thought to ask. A
    technician standing at the machine has different ones — "has this been
    building for a week or did it start today?", "when was comp3 last
    replaced?", "should I keep it running until the shift ends?".

    Day 7's `answer_question()` handles one of those. It cannot handle the
    second one, because "and how does that compare to yesterday?" only means
    anything if the assistant remembers what "that" was.

HOW IT WORKS:
    A session pins one machine's facts, gathered once, and holds a message
    history. The facts are *immutable for the session* — every turn is
    answered against the same DATA block, so a follow-up cannot drift onto a
    different machine or a re-scored probability mid-conversation.

    THE FAILURE MODE THIS IS BUILT AGAINST is different from the report's.
    A report confabulates a *cause*; a conversation confabulates *continuity*
    — the model agrees with a premise the user smuggled into their question
    ("so the temperature spike caused it?") because agreeing is conversational
    and disagreeing is not. Three defences:

      1. The facts are re-sent every turn, not summarised into history. The
         model never has to rely on its own earlier phrasing of a number.
      2. The system prompt states that declining is a complete answer, and
         names the four sensors that exist so "temperature" has a definite
         answer rather than a plausible one.
      3. History is capped. An unbounded conversation eventually pushes the
         DATA block out of the useful attention window, at which point the
         model is improvising from its own transcript.
"""

from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from src.genai.chains import get_llm
from src.genai.prompts import ASSISTANT_TEMPLATE, format_machine_facts
from src.utils.exceptions import PredictionError, ReportGenerationError
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Turns kept before the oldest are dropped. Each turn is a question plus an
# answer, and the DATA block is re-sent every time, so this bounds prompt
# growth rather than conversation length — the user can keep talking, the
# model just stops seeing the beginning.
DEFAULT_MAX_TURNS = 10


class MaintenanceAssistant:
    """Multi-turn Q&A about a single machine, grounded in its own data."""

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        provider: Optional[str] = None,
        temperature: float = 0.2,
        max_turns: int = DEFAULT_MAX_TURNS,
        **llm_kwargs: Any,
    ) -> None:
        """
        Args:
            llm: An explicit chat model. Injectable so tests need no API key.
            max_turns: Question/answer pairs retained in the prompt.
        """
        self.llm = (
            llm if llm is not None else get_llm(provider, temperature, **llm_kwargs)
        )
        self.chain = ASSISTANT_TEMPLATE | self.llm | StrOutputParser()
        self.max_turns = max_turns

        self._facts: Optional[str] = None
        self._record: Optional[Dict[str, Any]] = None
        self._history: List[BaseMessage] = []

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def start_session(self, record: Dict[str, Any]) -> None:
        """
        Pin one machine's facts for the conversation.

        Args:
            record: A `Predictor.explain_machine()` record. Pass
                `history_hours=24` when building it — without a trend, "has
                this been getting worse?" has no answer in the data and the
                model is left to guess at one.
        """
        required = ("machine_id", "failure_probability", "risk_level", "threshold")
        missing = [f for f in required if f not in record]
        if missing:
            raise ReportGenerationError(
                f"Cannot start a session: record is missing {missing}. "
                "Pass a record from Predictor.explain_machine()."
            )

        self._record = record
        self._facts = format_machine_facts(record)
        self._history = []

        has_trend = bool((record.get("context") or {}).get("recent_readings"))
        logger.info(
            f"Assistant session started for machine {record['machine_id']} "
            f"(risk={record['risk_level']}, trend data: "
            f"{'yes' if has_trend else 'no — history questions cannot be answered'})"
        )

    @property
    def machine_id(self) -> Any:
        """The machine this session is about, or None if not started."""
        return self._record["machine_id"] if self._record else None

    @property
    def history(self) -> List[BaseMessage]:
        """The retained conversation, oldest first."""
        return list(self._history)

    @property
    def turn_count(self) -> int:
        """Question/answer pairs exchanged in this session."""
        return len(self._history) // 2

    def reset(self) -> None:
        """Clear the conversation but keep the machine and its facts."""
        self._history = []
        logger.info("Assistant conversation cleared.")

    def end_session(self) -> None:
        """Drop everything, including the pinned machine."""
        self._facts = None
        self._record = None
        self._history = []

    # ------------------------------------------------------------------
    # Asking
    # ------------------------------------------------------------------

    def ask(self, question: str) -> str:
        """
        Ask a question about the pinned machine.

        Raises:
            ReportGenerationError: no session, empty question, or empty answer.
            LLMConnectionError: the provider is unreachable — the underlying
                prediction is unaffected.
        """
        if self._facts is None:
            raise ReportGenerationError(
                "No session started. Call start_session() with an "
                "explain_machine() record first."
            )
        if not question or not question.strip():
            raise ReportGenerationError("Question is empty.")

        question = question.strip()

        try:
            answer = self.chain.invoke(
                {
                    "machine_facts": self._facts,
                    "history": self._history,
                    "question": question,
                }
            )
        except Exception as e:
            # Reuse the report generator's classification so callers see one
            # consistent connectivity-vs-input distinction across the layer.
            from src.genai.chains import ReportGenerator

            raise ReportGenerator._wrap(
                e, f"question answering for machine {self.machine_id}"
            )

        if not answer or not answer.strip():
            raise ReportGenerationError("The model returned an empty answer.")

        answer = answer.strip()
        self._history.append(HumanMessage(content=question))
        self._history.append(AIMessage(content=answer))
        self._trim_history()

        logger.info(
            f"Machine {self.machine_id} — turn {self.turn_count}: "
            f"{len(question)} char question, {len(answer)} char answer"
        )
        return answer

    def _trim_history(self) -> None:
        """Keep only the most recent `max_turns` exchanges."""
        limit = self.max_turns * 2
        if len(self._history) > limit:
            dropped = len(self._history) - limit
            self._history = self._history[-limit:]
            logger.debug(f"Trimmed {dropped} message(s) from conversation history.")

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @classmethod
    def for_machine(
        cls,
        predictor: Any,
        dataset: Dict[str, Any],
        machine_id: Any,
        history_hours: int = 24,
        **kwargs: Any,
    ) -> "MaintenanceAssistant":
        """
        Build an assistant with a session already open on one machine.

        Defaults to 24 hours of trend so "is this getting worse?" is
        answerable from the data rather than from imagination.
        """
        record = predictor.explain_machine(
            dataset, machine_id, history_hours=history_hours
        )
        if not record:
            raise PredictionError(
                f"No prediction available for machine {machine_id!r}."
            )

        assistant = cls(**kwargs)
        assistant.start_session(record)
        return assistant
