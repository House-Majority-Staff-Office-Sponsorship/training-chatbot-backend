"""
DynamicResearchSquad — custom BaseAgent that spawns one researcher
per research question and runs them all in parallel.

Reads `research_questions` from session state, parses questions,
creates an LlmAgent + RAG tool for each, runs them concurrently, and
emits individual completion events as each finishes.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import AsyncGenerator

from google.adk.agents import BaseAgent, LlmAgent

from agents.rag_tool import create_rag_retrieval_tool, RagTokenUsage
from agents.runner_helper import run_agent_ephemeral


@dataclass
class QuestionItem:
    question: str
    description: str


@dataclass
class ResearcherLog:
    agent: str
    message: str
    prompt_tokens: int
    response_tokens: int
    total_tokens: int
    timestamp: int


@dataclass
class ResearcherResult:
    index: int
    findings: str
    logs: list[ResearcherLog]


class DynamicResearchSquad(BaseAgent):
    """Dynamically spawns one researcher per research question and runs them all in parallel."""

    def __init__(self, *, project: str, location: str, model: str, rag_corpus: str):
        super().__init__(
            name="dynamic_research_squad",
            description="Dynamically spawns one researcher per research question and runs them all in parallel.",
        )
        self._project = project
        self._location = location
        self._model = model
        self._rag_corpus = rag_corpus

    def _parse_questions(self, raw: str) -> list[QuestionItem]:
        """
        Extract questions from the question expander output.
        Tries JSON first, then falls back to extracting question-like lines.
        Hard cap at 10 questions.
        """
        # 1. Try JSON extraction
        json_match = re.search(r'\{[\s\S]*"questions"\s*:\s*\[[\s\S]*\]\s*\}', raw)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                if isinstance(parsed.get("questions"), list) and parsed["questions"]:
                    items = []
                    for q in parsed["questions"][:10]:
                        if isinstance(q, dict) and isinstance(q.get("question"), str) and q["question"].strip():
                            items.append(QuestionItem(
                                question=q["question"].strip(),
                                description=q.get("description", "").strip() if isinstance(q.get("description"), str) else "",
                            ))
                    if items:
                        return items
            except (json.JSONDecodeError, KeyError):
                pass

        # 2. Fallback: extract question-like lines
        questions: list[QuestionItem] = []
        for line in raw.split("\n"):
            trimmed = line.strip()
            if "?" not in trimmed:
                continue
            # Strip leading markers
            cleaned = re.sub(
                r"^(?:[-*•]\s*|#{1,4}\s+|\d+[.)]\s*|[a-zA-Z][.)]\s*|\*{1,2}|[IVXLC]+[.)]\s*)+",
                "",
                trimmed,
            ).strip()
            if len(cleaned) > 10 and "?" in cleaned:
                questions.append(QuestionItem(question=cleaned, description=""))

        return questions[:10]

    async def _run_question_researcher(
        self, item: QuestionItem, enriched_query: str
    ) -> tuple[str, list[ResearcherLog]]:
        """Run a single question researcher and return its findings + logs."""
        rag_logs: list[ResearcherLog] = []

        def on_rag_usage(usage: RagTokenUsage):
            snippet = usage.query[:57] + "..." if len(usage.query) > 60 else usage.query
            rag_logs.append(ResearcherLog(
                agent="retrieve_from_rag",
                message=f'retrieve_from_rag → "{snippet}"',
                prompt_tokens=usage.prompt_tokens,
                response_tokens=usage.response_tokens,
                total_tokens=usage.total_tokens,
                timestamp=usage.timestamp,
            ))

        rag_tool = create_rag_retrieval_tool(
            project=self._project,
            location=self._location,
            model=self._model,
            rag_corpus=self._rag_corpus,
            on_token_usage=on_rag_usage,
        )

        safe_name = "researcher_" + re.sub(r"[^a-z0-9]+", "_", item.question.lower())[:50]

        agent = LlmAgent(
            name=safe_name,
            model=self._model,
            description=f'Researches: "{item.question}"',
            instruction=f"""You are a focused research agent for the House Majority Staff Office training documentation system. Your job is to thoroughly research ONE specific question.

QUESTION TO RESEARCH:
{item.question}

RESEARCH GUIDANCE:
{item.description}

OVERALL CONTEXT:
{enriched_query}

Instructions:
1. Formulate 2-3 targeted queries specifically about this question.
2. Call the retrieve_from_rag tool for EACH query.
3. Synthesize ALL results into a comprehensive, detailed answer.

CRITICAL — Evidence and source requirements:
- Always include specific policy numbers, section references, or document titles when found.
- Quote key definitions, rules, and requirements verbatim using quotation marks.
- Cite data points, dates, thresholds, and numerical values exactly as they appear in the source.
- If a finding lacks a specific reference, note that no source reference was found.
- If the RAG corpus has no relevant information for this question, clearly state that.
- The RAG tool returns a SOURCES section with each response. You MUST collect ALL source references from every RAG call.

FORMATTING:
- Write in plain text only. No markdown, no headers, no bold, no bullet points, no special formatting.
- Use simple numbered lists or line breaks to organize information.
- The final report composer will handle all formatting later.

Your response MUST end with a REFERENCES section listing ALL sources from the RAG responses, one per line:

REFERENCES:
Document/source title (URI if available)
Document/source title (URI if available)""",
            tools=[rag_tool],
            output_key="question_findings",
        )

        findings = ""
        agent_logs: list[ResearcherLog] = []

        async for event in run_agent_ephemeral(agent, item.question, user_id="question-researcher", app_name="researcher_app"):
            author = event.author or "unknown"
            if author != "user":
                usage = getattr(event, "usage_metadata", None)
                prompt_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
                response_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0
                total_tokens = getattr(usage, "total_token_count", 0) if usage else (prompt_tokens + response_tokens)

                state_delta = {}
                if hasattr(event, "actions") and event.actions:
                    state_delta = getattr(event.actions, "state_delta", {}) or {}

                state_keys = list(state_delta.keys()) if state_delta else []
                detail = f" → wrote [{', '.join(state_keys)}]" if state_keys else ""

                if total_tokens > 0 or state_keys:
                    agent_logs.append(ResearcherLog(
                        agent=author,
                        message=f"{author}{detail}",
                        prompt_tokens=prompt_tokens,
                        response_tokens=response_tokens,
                        total_tokens=total_tokens,
                        timestamp=int(time.time() * 1000),
                    ))

                if "question_findings" in state_delta:
                    findings = str(state_delta["question_findings"])

        # Merge RAG logs into timeline
        all_logs = sorted(agent_logs + rag_logs, key=lambda l: l.timestamp)
        return findings, all_logs

    async def _run_impl(self, ctx) -> AsyncGenerator:
        """
        Core implementation: parse questions, spawn parallel researchers,
        yield events as each completes.
        """
        research_questions = (ctx.session.state or {}).get("research_questions", "")
        enriched_query = (ctx.session.state or {}).get("enriched_query", "")

        questions = self._parse_questions(research_questions)

        # Fallback: if parsing fails, run a single generic researcher
        if not questions:
            findings, logs = await self._run_question_researcher(
                QuestionItem(question="General Research", description=research_questions),
                enriched_query,
            )

            # Yield researcher logs
            for log in logs:
                yield self._make_state_event({
                    "researcher_log": json.dumps({
                        "agent": log.agent,
                        "message": log.message,
                        "promptTokens": log.prompt_tokens,
                        "responseTokens": log.response_tokens,
                        "totalTokens": log.total_tokens,
                        "timestamp": log.timestamp,
                        "researcherIndex": 0,
                    })
                })

            if ctx.session.state is not None:
                ctx.session.state["section_findings_all"] = findings

            yield self._make_state_event({
                "researcher_count": "1",
                "researcher_labels": json.dumps(["General Research"]),
                "researcher_0": findings,
            })
            return

        # Emit researcher count + labels
        yield self._make_state_event({
            "researcher_count": str(len(questions)),
            "researcher_labels": json.dumps([q.question for q in questions]),
        })

        # Launch all researchers in parallel using asyncio
        results: dict[int, ResearcherResult] = {}
        done_queue: asyncio.Queue[ResearcherResult] = asyncio.Queue()

        async def run_and_enqueue(idx: int, item: QuestionItem):
            try:
                findings, logs = await self._run_question_researcher(item, enriched_query)
                await done_queue.put(ResearcherResult(index=idx, findings=findings, logs=logs))
            except Exception as e:
                await done_queue.put(ResearcherResult(
                    index=idx,
                    findings=f"[Research failed: {e}]",
                    logs=[],
                ))

        # Start all tasks
        tasks = [
            asyncio.create_task(run_and_enqueue(i, q))
            for i, q in enumerate(questions)
        ]

        # Yield events as each researcher completes (in completion order)
        ordered_findings: list[str] = [""] * len(questions)
        for _ in range(len(questions)):
            result = await done_queue.get()
            ordered_findings[result.index] = result.findings

            # Emit researcher logs with researcherIndex
            for log in result.logs:
                yield self._make_state_event({
                    "researcher_log": json.dumps({
                        "agent": log.agent,
                        "message": log.message,
                        "promptTokens": log.prompt_tokens,
                        "responseTokens": log.response_tokens,
                        "totalTokens": log.total_tokens,
                        "timestamp": log.timestamp,
                        "researcherIndex": result.index,
                    })
                })

            # Emit researcher_done
            yield self._make_state_event({
                f"researcher_{result.index}": result.findings,
            })

        # Wait for all tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)

        # Combine findings for the report compiler
        combined = "\n\n---\n\n".join(
            f"{questions[i].question}\n\n{ordered_findings[i]}"
            for i in range(len(questions))
        )
        if ctx.session.state is not None:
            ctx.session.state["section_findings_all"] = combined

    def _make_state_event(self, state_delta: dict):
        """Create an event with a state delta."""
        from google.adk.events import Event, EventActions
        return Event(
            author=self.name,
            actions=EventActions(state_delta=state_delta),
        )

    async def _run_async_impl(self, ctx) -> AsyncGenerator:
        async for event in self._run_impl(ctx):
            yield event

    async def _run_live_impl(self, ctx) -> AsyncGenerator:
        # Not used
        return
        yield  # Make it an async generator
