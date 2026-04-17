"""
Programmatic runner for the deep research agent.

Uses ADK's InMemoryRunner to execute the multi-squad SequentialAgent pipeline.
Yields PipelineEvent dicts for SSE streaming.
"""

from __future__ import annotations

import json
import re
import time
from typing import AsyncGenerator, Literal, TypedDict

from models import ConversationMessage, format_conversation_history
from agents.runner_helper import run_agent_ephemeral, extract_usage_tokens
from agents.deep_research.pipeline import create_deep_research_agent


# ---------------------------------------------------------------------------
# Event types (match the TypeScript originals exactly)
# ---------------------------------------------------------------------------

class StepEvent(TypedDict):
    type: Literal["step"]
    field: str  # "enrichedQuery" | "researchQuestions" | "answer"
    value: str


class ResearchersInitEvent(TypedDict):
    type: Literal["researchers_init"]
    count: int
    labels: list[str]


class ResearcherDoneEvent(TypedDict):
    type: Literal["researcher_done"]
    index: int
    label: str
    value: str


class LogEvent(TypedDict):
    type: Literal["log"]
    agent: str
    message: str
    promptTokens: int
    responseTokens: int
    totalTokens: int
    timestamp: int
    researcherIndex: int | None


from typing import Union
PipelineEvent = Union[StepEvent, ResearchersInitEvent, ResearcherDoneEvent, LogEvent]


# Maps session state keys to SSE field names
STATE_KEY_MAP: dict[str, str] = {
    "enriched_query": "enrichedQuery",
    "research_questions": "researchQuestions",
    "final_report": "answer",
}

# Keys handled specially (not included in generic log detail)
HANDLED_KEY_RE = re.compile(r"^(researcher_log|researcher_count|researcher_labels|researcher_\d+)$")


async def stream_deep_research(
    query: str,
    *,
    project: str,
    location: str,
    fast_model: str,
    advanced_model: str,
    report_model: str,
    rag_corpus: str,
    conversation_history: list[ConversationMessage] | None = None,
) -> AsyncGenerator[PipelineEvent, None]:
    """
    Streaming runner — yields PipelineEvent dicts as each pipeline step
    writes its output to session state.
    """
    agent = create_deep_research_agent(
        fast_model=fast_model,
        advanced_model=advanced_model,
        report_model=report_model,
        project=project,
        location=location,
        rag_corpus=rag_corpus,
    )

    history_prefix = format_conversation_history(conversation_history or [])
    message = history_prefix + query

    researcher_labels: list[str] = []

    async for event in run_agent_ephemeral(agent, message, user_id="research-user", app_name="deep_research_app"):
        author = event.author or "unknown"
        prompt_tokens, response_tokens, total_tokens = extract_usage_tokens(event)

        if author != "user":
            state_delta = {}
            if hasattr(event, "actions") and event.actions:
                state_delta = getattr(event.actions, "state_delta", {}) or {}

            state_keys = list(state_delta.keys()) if state_delta else []
            unhandled_keys = [k for k in state_keys if not HANDLED_KEY_RE.match(k)]
            detail = f" → wrote [{', '.join(unhandled_keys)}]" if unhandled_keys else ""

            # Emit log event (skip noise: no tokens and no meaningful state writes)
            if total_tokens > 0 or unhandled_keys:
                yield LogEvent(
                    type="log",
                    agent=author,
                    message=f"{author}{detail}",
                    promptTokens=prompt_tokens,
                    responseTokens=response_tokens,
                    totalTokens=total_tokens,
                    timestamp=int(time.time() * 1000),
                    researcherIndex=None,
                )

        # Process state delta
        state_delta = {}
        if hasattr(event, "actions") and event.actions:
            state_delta = getattr(event.actions, "state_delta", {}) or {}

        if not state_delta:
            continue

        for key, value in state_delta.items():
            if not isinstance(value, str):
                continue

            # Researcher metadata: labels and count
            if key == "researcher_labels":
                try:
                    researcher_labels = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    researcher_labels = []
                count_val = state_delta.get("researcher_count")
                count = int(count_val) if count_val is not None else len(researcher_labels)
                yield ResearchersInitEvent(
                    type="researchers_init",
                    count=count,
                    labels=researcher_labels,
                )
                continue

            if key == "researcher_count":
                continue

            # Researcher log events
            if key == "researcher_log":
                try:
                    log = json.loads(value)
                    yield LogEvent(
                        type="log",
                        agent=log.get("agent", "researcher"),
                        message=log.get("message", ""),
                        promptTokens=log.get("promptTokens", 0),
                        responseTokens=log.get("responseTokens", 0),
                        totalTokens=log.get("totalTokens", 0),
                        timestamp=log.get("timestamp", int(time.time() * 1000)),
                        researcherIndex=log.get("researcherIndex"),
                    )
                except (json.JSONDecodeError, TypeError):
                    pass
                continue

            # Individual researcher completion
            researcher_match = re.match(r"^researcher_(\d+)$", key)
            if researcher_match:
                index = int(researcher_match.group(1))
                yield ResearcherDoneEvent(
                    type="researcher_done",
                    index=index,
                    label=researcher_labels[index] if index < len(researcher_labels) else f"Researcher {index + 1}",
                    value=value,
                )
                continue

            # Standard pipeline step
            if key in STATE_KEY_MAP:
                yield StepEvent(
                    type="step",
                    field=STATE_KEY_MAP[key],
                    value=value,
                )
