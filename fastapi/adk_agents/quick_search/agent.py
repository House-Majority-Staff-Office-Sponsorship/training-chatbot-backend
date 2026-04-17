"""ADK web wrapper for the quick_search agent."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from google.adk.agents import LlmAgent

import config  # noqa: E402
from agents.quick_search import QUICK_SEARCH_INSTRUCTION  # noqa: E402
from agents.rag_tool import create_rag_retrieval_tool  # noqa: E402


rag_tool = create_rag_retrieval_tool(
    project=config.GCP_PROJECT,
    location=config.GCP_LOCATION,
    model=config.GEN_FAST_MODEL,
    rag_corpus=config.RAG_CORPUS,
)

root_agent = LlmAgent(
    name="quick_search_agent",
    model=config.GEN_FAST_MODEL,
    description="Single-pass research agent that searches and compiles answers from the RAG corpus.",
    instruction=QUICK_SEARCH_INSTRUCTION,
    tools=[rag_tool],
)
