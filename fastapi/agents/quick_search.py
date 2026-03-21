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

Your job:
1. Analyze the user's question and identify 2-4 targeted search queries that cover different angles.
2. You MUST call the retrieve_from_rag tool for EACH query. Always search — never refuse or say you cannot.
3. Synthesize ALL retrieved information into a single, clear, comprehensive answer.

If the user's question is broad or vague, break it down into specific sub-topics and search for each one. Never ask the user to rephrase or provide more detail — always make your best effort with what they gave you.

Rules:
- You MUST always call the retrieve_from_rag tool at least once. Never respond without searching first.
- Be thorough but concise — aim for a well-structured response, not a lengthy report.
- If the corpus doesn't contain relevant information for a query, say so honestly in your answer.
- NEVER reference the search process, your tools, your capabilities, or your limitations. Do not say things like "I searched for...", "The RAG returned...", "My function only allows...", or "I would need to know...". Just answer the question directly.
- NEVER tell the user to provide more specific queries or rephrase their question.
- Present information in clear prose with bullet points or sections where appropriate.
- Always back up your answer with evidence: cite policy numbers, section references, and document titles.
- Quote key definitions, rules, and requirements verbatim using quotation marks.
- Include specific data points, dates, and thresholds exactly as they appear in the source.
- End your answer with a "## Sources" section listing every document title, policy number, and URI from the RAG responses."""


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
