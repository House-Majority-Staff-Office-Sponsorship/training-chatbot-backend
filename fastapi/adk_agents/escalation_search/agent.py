"""ADK web wrapper for the escalation_search agent.

For manual testing via `adk web`, provide the previous answer inline in
the user message, e.g.:

    PREVIOUS ANSWER:
    ---
    <paste the Flash quick-search answer here>
    ---

    USER QUESTION:
    <your question>
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from google.adk.agents import LlmAgent

import config  # noqa: E402
from agents.escalation_search import ESCALATION_INSTRUCTION  # noqa: E402
from agents.rag_tool import create_rag_retrieval_tool  # noqa: E402


rag_tool = create_rag_retrieval_tool(
    project=config.GCP_PROJECT,
    location=config.GCP_LOCATION,
    model=config.GEN_PRO_MODEL,
    rag_corpus=config.RAG_CORPUS,
)

root_agent = LlmAgent(
    name="escalation_search_agent",
    model=config.GEN_PRO_MODEL,
    description="Deep follow-up research agent that re-searches the RAG corpus with greater thoroughness after an initial answer was insufficient.",
    instruction=ESCALATION_INSTRUCTION,
    tools=[rag_tool],
)
