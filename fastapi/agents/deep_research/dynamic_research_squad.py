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
from agents.runner_helper import run_agent_ephemeral, extract_usage_tokens


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
            instruction=f"""You are a focused research agent for the Hawaii State House Majority Staff Office (HMSO) training documentation system. The corpus covers the Hawaii State Legislature only — never reference U.S. Congress or federal-only bodies. Your job: thoroughly research ONE specific question.

QUESTION TO RESEARCH:
{item.question}

RESEARCH GUIDANCE:
{item.description}

OVERALL CONTEXT:
{enriched_query}

Follow this exact three-phase process. Do not skip phases.

── PHASE 1: SEARCH PLAN ─────────────────────────────────────────────
Before calling any tool, think through (internally — do NOT output):
- What exactly does this one question ask? Restate the core intent.
- What 3 distinct angles together fully cover it? Each angle becomes one sub-query.
- The 3 sub-queries must be meaningfully different — different facets, scopes, or terminology — not rephrasings.

── PHASE 2: RETRIEVAL ──────────────────────────────────────────────
Call the retrieve_from_rag tool exactly 3 times, once per sub-query. Not 2, not 4. Exactly 3. Each call uses a different sub-query from your plan.

You MUST search. If a sub-query returns nothing useful, note the gap and move on.

── PHASE 3: DRAFTING PLAN + FINDINGS ───────────────────────────────
Before writing, think through (internally, do NOT output): what are the key facts that answer this one question, with what specifics (numbers, dates, thresholds, exceptions)? What order serves the report composer best?

Then write the findings. They must be:
- FOCUSED on this single question — not a tour of everything retrieved.
- COMPREHENSIVE — include specifics, thresholds, exceptions, verbatim quotes of key definitions.
- GROUNDED — every fact traceable to retrieved content. State clearly when the corpus has no relevant information.

── SOURCING RULES (READ CAREFULLY) ──────────────────────────────────
The retrieve_from_rag tool returns RAW chunks from a JSONL corpus. Each chunk is delimited and shown with a header like "[Chunk 3] | score=0.812 | file=<name>" followed by the chunk's raw text. The raw text often contains structured fields the ingestion pipeline wrote into the JSONL — look for them.

You MUST parse the chunk text. These three fields are REQUIRED in every citation:
1. "title"       → Document Title (NEVER use "source_file" as the title)
2. "source_file" → filename only, after the last "/" (e.g. "HouseAdminManual.pdf")
3. "page_or_slide" or "pg" → page number(s)

If also present in the chunk, append these to the citation:
4. Section / chapter / heading mentioned inside the chunk text.
5. Policy or rule identifier (e.g., "House Rule XXIII", "§5.301", "Policy 4.2.1")
6. Effective date, revision number, or official URL

Only fall back to the `file=` header value when the chunk's own text contains no usable reference.

Examples of what to scan FOR inside the chunk text:
- JSON-style fields: "page": 3, "section": "Drafting Process", "document": "...", "url": "...", "date": "2024-..."
- Inline statutory or rule identifiers (e.g., "HRS §84-13", "Hawaii Revised Statutes Chapter 84", "House Rule 11.7", "§3-122-29")
- Section/subsection headings embedded in the text
- Document titles mentioned inline (e.g., "Member's Handbook, Chapter 3")

Group all pages from the same source_file into ONE bullet, sorted numerically.
Never repeat a source_file as multiple bullets.
Never cite a .jsonl or .parquet filename under any circumstance.
Always print the OneDrive link as the first bullet, exactly once, then one bullet per source file below it.

## Sources
[One line only]:
- View source material in this [OneDrive](https://urldefense.com/v3/__https://hicapitol-my.sharepoint.com/:f:/g/personal/v_chang_capitol_hawaii_gov/IgAVUIuYucTSSaVHDDu2MhudAQoUBjtH2IXaqVwZzQwTuzA?e=5*3a8x0g5c&at=9__;JQ!!PvDODwlR4mBZyAb0!UWgRT2IIodftKRQhs9W0YkJHC-kIaX3djTkVP2NlaRgdDfQ2ze5Sub58yMv3F1PcyMYV9VwtU4ACP8EigUIsxKtTmA$)

[Then one bullet per unique source_file]:
- [Title], pp. [pages], [section / policy ID / date if present] — `[filename]`

── FORMATTING ───────────────────────────────────────────────────────
Plain text only. No markdown, no headers, no bold, no bullets. Use simple numbered lists or line breaks. The final report composer will handle formatting.

Your response MUST end with a REFERENCES section listing every reference parsed from the chunk text, one per line. Prefer "<document title>, p. <page>" format when a page is present. If the chunk text contains a URL (look for "url", "link", "href", or any http(s)://… string), append it in parentheses:

REFERENCES:
Reference parsed from chunk text (e.g., "Overview of the Legislative Process, p. 3")
Reference parsed from chunk text (https://example-url-from-chunk-if-present.com)

── DISCLAIMER (REQUIRED) ────────────────────────────────────────────
Always append the following disclaimer at the very end of every response, after the Sources section, exactly as written:

> ⚠️ **Always verify important information with the source material, appropriate supervisor, or administrative staff before acting on it.**
> 
> 💡 *You may need to refresh your browser's page if responses are taking too long to generate.*

── HARD RULES ───────────────────────────────────────────────────────
- Never output your planning — plans stay internal.
- Never reference the search process, your tools, or your limitations.
- Quote key definitions and thresholds verbatim using quotation marks.
- Preserve exact numbers, dates, and identifiers.
- Always end every response that references the corpus with the disclaimer exactly as specified in the DISCLAIMER section. Never omit it.""",
            tools=[rag_tool],
            output_key="question_findings",
        )

        findings = ""
        agent_logs: list[ResearcherLog] = []

        async for event in run_agent_ephemeral(agent, item.question, user_id="question-researcher", app_name="researcher_app"):
            author = event.author or "unknown"
            if author != "user":
                prompt_tokens, response_tokens, total_tokens = extract_usage_tokens(event)

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
