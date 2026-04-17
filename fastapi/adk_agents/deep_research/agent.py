"""ADK web wrapper for the full deep_research pipeline (SequentialAgent)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import config  # noqa: E402
from agents.deep_research.pipeline import create_deep_research_agent  # noqa: E402


root_agent = create_deep_research_agent(
    fast_model=config.GEN_FAST_MODEL,
    advanced_model=config.GEN_PRO_MODEL,
    report_model=config.GEN_REPORT_MODEL,
    project=config.GCP_PROJECT,
    location=config.GCP_LOCATION,
    rag_corpus=config.RAG_CORPUS,
)
