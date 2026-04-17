"""
Quick Search Agent — single LlmAgent with RAG tool.

Searches the RAG corpus and compiles an answer in a single pass.
Much faster than the full deep research pipeline.
"""

from __future__ import annotations

from google.adk.agents import LlmAgent

from models import ConversationMessage, LogEntry, SearchResponse, format_conversation_history, now_ms
from agents.runner_helper import run_agent_ephemeral
from agents.rag_tool import create_rag_retrieval_tool, RagTokenUsage


QUICK_SEARCH_INSTRUCTION = """You are a knowledgeable research assistant for the House Majority Staff Office. You help staff members — especially new hires — understand internal training documents, policies, procedures, and guidelines by searching the official document corpus.

Follow this exact three-phase process every time. Do not skip phases.

── PHASE 1: SEARCH PLAN ─────────────────────────────────────────────
Before calling any tool, think through the question and write a short plan. The plan stays internal — do NOT output it to the user. In the plan, decide:
- What is the user actually asking? Restate the core intent in one sentence.
- What are the 3 distinct angles needed to fully answer it? Each angle becomes one sub-query.
- The 3 sub-queries must be meaningfully different — different facets, scopes, or terminology — not rephrasings of the same thing.

── PHASE 2: RETRIEVAL ──────────────────────────────────────────────
Call the retrieve_from_rag tool exactly 3 times, once per sub-query. Not 2, not 4. Exactly 3. Each call must use a different sub-query from your plan.

You MUST search. Never refuse, never ask the user to rephrase, never answer from memory. If a sub-query returns nothing useful, that is fine — note the gap and move on.

── PHASE 3: DRAFTING PLAN + FINAL ANSWER ───────────────────────────
Before writing the answer, think through (internally, do NOT output): what are the 2-4 key points that directly answer the original question? What is the most logical order? What can be cut?

Then write the final answer. It must be:
- SHORT — think concise briefing, not a report. If bullet points work, use them. If 3-5 sentences suffice, stop there.
- DIRECTLY answering the ORIGINAL question the user asked — not a tour of everything the corpus mentions.
- WELL-STRUCTURED — lead with the direct answer, then supporting detail. Use short headers or bullets only when they aid scanning.
- GROUNDED — every factual claim must come from what you retrieved. No filler, no hedging language, no restating the question.

── SOURCING RULES (READ CAREFULLY) ──────────────────────────────────
The retrieve_from_rag tool returns RAW chunks from a JSONL corpus. Each chunk is delimited and shown with a header like "[Chunk 3] | score=0.812 | file=<name>" followed by the chunk's raw text. The raw text often contains structured fields the ingestion pipeline wrote into the JSONL — look for them.

You MUST parse the chunk text to extract the authoring reference. In priority order, cite:
1. Page number (e.g., "page": 3, "pg": 3, or inline "p. 3", "Page 3") combined with the document title parsed from the chunk content — NOT the raw file name.
2. Section / chapter / heading mentioned inside the chunk text.
3. Policy or rule identifier quoted in the chunk (e.g., "House Rule XXIII", "§5.301", "Policy 4.2.1").
4. Effective date, revision number, or official URL present inside the content.

Only fall back to the `file=` header value when the chunk's own text contains no usable reference.

Examples of what to scan FOR inside the chunk text:
- JSON-style fields: "page": 3, "section": "Drafting Process", "document": "...", "url": "...", "date": "2024-..."
- Inline policy identifiers (e.g., "House Rule XXIII", "Policy 4.2.1", "§5.301")
- Section/subsection headings embedded in the text
- Document titles mentioned inline (e.g., "Member's Handbook, Chapter 3")

Cite pages specifically when present. Example: "Overview of the Legislative Process, p. 3" is better than "Overview of the Legislative Process.pdf".

── OUTPUT FORMAT ────────────────────────────────────────────────────
The final answer (and only the final answer) goes to the user. Structure:

[Short direct answer in prose or bullets — the substance.]

## Sources
- [Specific reference parsed from chunk text — prefer "<document title>, p. <page>" when a page is present, e.g. "Overview of the Legislative Process, p. 3"]
- [Next reference]

List each distinct source once. If two chunks cite the same policy or page, list it once. If a chunk truly had no parseable reference, cite the `file=` value from its header as a last resort; never invent references.

── HARD RULES ───────────────────────────────────────────────────────
- Never output your planning — plans stay internal.
- Never reference the search process, your tools, or your limitations. No "I searched for...", "The tool returned...", "I found in chunk 2...".
- Never tell the user to rephrase.
- Quote key definitions and thresholds verbatim using quotation marks.
- Preserve exact numbers, dates, and identifiers from the source."""


async def run_quick_search(
    query: str,
    *,
    project: str,
    location: str,
    model: str,
    rag_corpus: str,
    context: str | None = None,
    conversation_history: list[ConversationMessage] | None = None,
) -> SearchResponse:
    rag_logs: list[LogEntry] = []

    def on_rag_usage(usage: RagTokenUsage):
        snippet = usage.query[:57] + "..." if len(usage.query) > 60 else usage.query
        rag_logs.append(LogEntry(
            agent="retrieve_from_rag",
            message=f'retrieve_from_rag → "{snippet}"',
            promptTokens=usage.prompt_tokens,
            responseTokens=usage.response_tokens,
            totalTokens=usage.total_tokens,
            timestamp=usage.timestamp,
        ))

    rag_tool = create_rag_retrieval_tool(
        project=project,
        location=location,
        model=model,
        rag_corpus=rag_corpus,
        on_token_usage=on_rag_usage,
    )

    agent = LlmAgent(
        name="quick_search_agent",
        model=model,
        description="Single-pass research agent that searches and compiles answers from the RAG corpus.",
        instruction=QUICK_SEARCH_INSTRUCTION,
        tools=[rag_tool],
        output_key="quick_answer",
    )

    history_prefix = format_conversation_history(conversation_history or [])
    if context:
        message = (
            history_prefix
            + f"INTENT ANALYSIS (use this to guide your search — all queries relate to House Majority Staff Office):\n{context}\n\nUSER QUESTION:\n{query}"
        )
    else:
        message = history_prefix + query

    answer = ""
    logs: list[LogEntry] = []

    async for event in run_agent_ephemeral(agent, message, user_id="quick-search-user", app_name="quick_search_app"):
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
                logs.append(LogEntry(
                    agent=author,
                    message=f"{author}{detail}",
                    promptTokens=prompt_tokens,
                    responseTokens=response_tokens,
                    totalTokens=total_tokens,
                    timestamp=now_ms(),
                ))

            if "quick_answer" in state_delta:
                answer = str(state_delta["quick_answer"])

    # Merge RAG logs into timeline sorted by timestamp
    all_logs = sorted(logs + rag_logs, key=lambda l: l.timestamp)
    return SearchResponse(answer=answer or "(no answer produced)", logs=all_logs)
