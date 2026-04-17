"""ADK web wrapper for the quiz_generator agent.

For manual testing via `adk web`, send a message like:

    Generate 5 multiple-choice quiz questions about: <topic>
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from google.adk.agents import LlmAgent

import config  # noqa: E402
from agents.quiz_generator import QUIZ_GENERATOR_INSTRUCTION  # noqa: E402
from agents.rag_tool import create_rag_retrieval_tool  # noqa: E402


rag_tool = create_rag_retrieval_tool(
    project=config.GCP_PROJECT,
    location=config.GCP_LOCATION,
    model=config.GEN_FAST_MODEL,
    rag_corpus=config.RAG_CORPUS,
)

root_agent = LlmAgent(
    name="quiz_generator",
    model=config.GEN_FAST_MODEL,
    description="Generates structured quiz questions from RAG corpus content.",
    instruction=QUIZ_GENERATOR_INSTRUCTION,
    tools=[rag_tool],
)
