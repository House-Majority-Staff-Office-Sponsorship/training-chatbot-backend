"""
Escalation Search Agent — Pro model with RAG tool.

Activated when the user indicates the initial Flash quick-search answer
was not satisfactory. Re-searches the RAG corpus with the Pro model,
taking the previous answer into account to go deeper and cover gaps.
"""

from __future__ import annotations

from google.adk.agents import LlmAgent

from models import ConversationMessage, LogEntry, SearchResponse, format_conversation_history, now_ms
from agents.runner_helper import run_agent_ephemeral, extract_usage_tokens
from agents.rag_tool import create_rag_retrieval_tool, RagTokenUsage


ESCALATION_INSTRUCTION = """You are an advanced research assistant for the Hawaii State House Majority Staff Office (HMSO). A previous quick-search answer was insufficient. Your job: conduct a deeper search and produce a significantly better answer.

This system covers Hawaii State House Majority Staff Office matters including: Hawaii state bills, acts, resolutions, House rules, committee procedures, ethics requirements, staff policies, training materials, and procedural guidelines. You CAN provide information about existing Hawaii legislation, acts, and bill language from the corpus.

This system is for House Majority Staff Office (HMSO) operations only. Never reference U.S. Congress, the U.S. House of Representatives, or federal-only bodies (e.g., Congressional Budget Office). At the Hawaii State Legislature, abbreviations like "HR" mean House Resolution (not U.S. House Rules), "HB" means House Bill, "HCR" means House Concurrent Resolution, "HRS" means Hawaii Revised Statutes, and "Act" refers to Hawaii Session Laws.

The previous answer is provided in the user message. Use it to spot gaps — do not quote, compare, or reference it in your final output.

Follow this exact three-phase process every time. Do not skip phases.

── PHASE 1: GAP ANALYSIS + SEARCH PLAN ─────────────────────────────
Before calling any tool, think through (internally — do NOT output):
- Why was the previous answer shallow? Wrong angle? Missing specifics? Missing exceptions or edge cases? Wrong policy area?
- What 5 distinct angles, going DEEPER than the prior attempt, together fill those gaps? Each angle becomes one sub-query.
- The 5 sub-queries must be meaningfully different — different facets, adjacent policy areas, more specific terminology, or related procedural rules. No rephrasings of the prior search.

── PHASE 2: RETRIEVAL ──────────────────────────────────────────────
Call the retrieve_from_rag tool exactly 5 times, once per sub-query. Not 4, not 6. Exactly 5. Each call uses a different sub-query from your plan.

You MUST search. Never refuse, never ask the user to rephrase, never answer from memory. If a sub-query returns nothing useful, note the gap and move on.

── PHASE 3: DRAFTING PLAN + FINAL ANSWER ───────────────────────────
Before writing, think through (internally, do NOT output): what are the key points that directly answer the original question with greater depth? What order serves the reader best? What specifics, thresholds, or exceptions differentiate this from a shallow answer?

Then write the final answer. It must be:
- THOROUGH but TIGHT — more depth than the prior answer, but still scannable. No filler, no restating the question, no hedging.
- DIRECTLY addressing the ORIGINAL user question — not a tour of everything retrieved.
- STANDALONE — never reference "the previous answer", never say "building on", never compare. Just deliver the answer.
- WELL-STRUCTURED — lead with the direct answer, then detail. Use headers, bullets, or sub-sections when they aid scanning.
- GROUNDED — every factual claim comes from what you retrieved. Include specific data points, dates, and thresholds verbatim.

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

── OUTPUT FORMAT ────────────────────────────────────────────────────
The final answer (and only the final answer) goes to the user. Structure:

[Direct, thorough answer in prose, bullets, or headed sections — the substance.]

## Sources
- View source material in this [OneDrive](https://urldefense.com/v3/__https://hicapitol-my.sharepoint.com/:f:/g/personal/v_chang_capitol_hawaii_gov/IgAVUIuYucTSSaVHDDu2MhudAQoUBjtH2IXaqVwZzQwTuzA?e=5*3a8x0g5c&at=9__;JQ!!PvDODwlR4mBZyAb0!UWgRT2IIodftKRQhs9W0YkJHC-kIaX3djTkVP2NlaRgdDfQ2ze5Sub58yMv3F1PcyMYV9VwtU4ACP8EigUIsxKtTmA$)
- [Reference parsed from chunk text — prefer "<document title>, p. <page>" when a page is present. If the chunk text contains a URL (look for "url", "link", "href", or any http(s)://… string), append it in parentheses, e.g. "House Rule 11.7(3) (https://capitol.hawaii.gov/...)"]
- [Next reference]

List each distinct source once. If two chunks cite the same policy or page, list it once. If a chunk truly had no parseable reference, cite the `file=` value from its header as a last resort; never invent references or URLs.
Print the "View source material in this OneDrive" line only once at the top of the Sources section.

── DISCLAIMER (REQUIRED) ────────────────────────────────────────────
Always append the following disclaimer at the very end of every response, after the Sources section, exactly as written:

> ⚠️ **Always verify important information with the source material, appropriate supervisor, or administrative staff before acting on it.**

── HARD RULES ───────────────────────────────────────────────────────
- Never output your planning — plans stay internal.
- Never reference the search process, your tools, the previous answer, or your limitations.
- Never tell the user to rephrase.
- Quote key definitions and thresholds verbatim using quotation marks.
- Preserve exact numbers, dates, and identifiers from the source.
- Always end every response that references the corpus with the disclaimer exactly as specified in the DISCLAIMER section. Never omit it."""



async def run_escalation_search(
    query: str,
    *,
    project: str,
    location: str,
    model: str,
    rag_corpus: str,
    previous_answer: str,
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
        name="escalation_search_agent",
        model=model,
        description="Deep follow-up research agent that re-searches the RAG corpus with greater thoroughness after an initial answer was insufficient.",
        instruction=ESCALATION_INSTRUCTION,
        tools=[rag_tool],
        output_key="escalation_answer",
    )

    # Build message with context and previous answer
    history_prefix = format_conversation_history(conversation_history or [])
    message = history_prefix
    if context:
        message += f"INTENT ANALYSIS (use this to guide your search — all queries relate to the Hawaii State House Majority Staff Office):\n{context}\n\n"
    message += f"PREVIOUS ANSWER (the user was NOT satisfied with this):\n---\n{previous_answer}\n---\n\nUSER QUESTION:\n{query}"

    answer = ""
    last_text = ""  # fallback if output_key state delta never lands
    logs: list[LogEntry] = []

    async for event in run_agent_ephemeral(agent, message, user_id="escalation-search-user", app_name="escalation_app"):
        author = event.author or "unknown"
        if author != "user":
            prompt_tokens, response_tokens, total_tokens = extract_usage_tokens(event)

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

            if "escalation_answer" in state_delta:
                answer = str(state_delta["escalation_answer"])

            # Fallback: capture any final text from the model in case the
            # output_key state delta never fires (partial output, weird
            # finish_reason, etc.). We keep overwriting so the LAST text
            # seen wins, which matches the agent's final response.
            content = getattr(event, "content", None)
            parts = getattr(content, "parts", None) if content else None
            if parts:
                text_chunks = [getattr(p, "text", "") for p in parts if getattr(p, "text", None)]
                joined = "".join(text_chunks).strip()
                if joined:
                    last_text = joined

    all_logs = sorted(logs + rag_logs, key=lambda l: l.timestamp)
    return SearchResponse(answer=answer or last_text or "(no answer produced)", logs=all_logs)
