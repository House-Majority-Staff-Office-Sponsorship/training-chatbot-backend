"""ADK web wrapper for the conversational agent."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from google.adk.agents import LlmAgent

import config  # noqa: E402  — triggers .env load + ADK env vars
from agents.conversational import CONVERSATIONAL_INSTRUCTION  # noqa: E402


root_agent = LlmAgent(
    name="conversational_agent",
    model=config.GEN_FAST_MODEL,
    description="Handles conversational queries — greetings, small talk, and questions about the system.",
    instruction=CONVERSATIONAL_INSTRUCTION,
)
