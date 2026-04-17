"""ADK web wrapper for the intent_orchestrator agent."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from google.adk.agents import LlmAgent

import config  # noqa: E402
from agents.intent_orchestrator import INTENT_INSTRUCTION  # noqa: E402


root_agent = LlmAgent(
    name="intent_orchestrator",
    model=config.GEN_FAST_MODEL,
    description="Validates user queries for relevance and enriches them for downstream research agents.",
    instruction=INTENT_INSTRUCTION,
)
