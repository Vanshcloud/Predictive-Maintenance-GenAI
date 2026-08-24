"""
GenAI Package
=============
Turns grounded predictions into readable maintenance reports.

The prompts are the important part: every fact a report may cite is supplied
explicitly, because a model handed only a probability will confabulate a
fluent and entirely fabricated diagnosis. See `prompts.py`.
"""

from src.genai.assistant import MaintenanceAssistant
from src.genai.chains import ReportGenerator, get_llm
from src.genai.prompts import format_machine_facts

__all__ = [
    "MaintenanceAssistant",
    "ReportGenerator",
    "get_llm",
    "format_machine_facts",
]
