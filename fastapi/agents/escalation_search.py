"""
Escalation Search Agent — Pro model with RAG tool.

Activated when the user indicates the initial Flash quick-search answer
was not satisfactory. Re-searches the RAG corpus with the Pro model,
taking the previous answer into account to go deeper and cover gaps.
"""

from __future__ import annotations

from google.adk.agents import LlmAgent

from models import ConversationMessage, LogEntry, SearchResponse, format_conversation_history, now_ms
from agents.runner_helper import run_agent_ephemeral
from agents.rag_tool import create_rag_retrieval_tool, RagTokenUsage


ESCALATION_INSTRUCTION = """You are an advanced research assistant for the House Majority Staff Office. A previous search attempt produced an answer that the user found insufficient. Your job is to conduct a MORE THOROUGH search and provide a significantly better answer.

The previous answer will be provided in the user message. Analyze it to understand what was already covered and what gaps remain.

Your approach:
1. Analyze WHY the previous answer may have been insufficient — it may have been too shallow, missed key details, covered the wrong angle, or lacked specifics.
2. Formulate 4-6 targeted search queries that go DEEPER than the previous attempt. Try different angles, more specific terminology, and related policy areas that the first search may have missed.
3. You MUST call the retrieve_from_rag tool for EACH query. Always search — never refuse or say you cannot.
4. Synthesize ALL retrieved information into a comprehensive, well-structured answer that clearly improves upon the previous one.

Rules:
- You MUST always call the retrieve_from_rag tool at least once. Never respond without searching first.
- Be thorough and comprehensive — the user already got a quick answer and wants more depth.
- If the corpus doesn't contain relevant information for a query, say so honestly in your answer.
- NEVER reference the search process, your tools, your capabilities, or your limitations. Do not say things like "I searched for...", "The RAG returned...", "My function only allows...", or "I would need to know...". Just answer the question directly.
- NEVER reference the previous answer or say things like "Building on the previous answer..." or "The earlier response missed...". Just provide a complete, standalone answer.
- NEVER tell the user to provide more specific queries or rephrase their question.
- Present information in clear prose with bullet points or sections where appropriate.
- Always back up your answer with evidence: cite policy numbers, section references, and document titles.
- Quote key definitions, rules, and requirements verbatim using quotation marks.
- Include specific data points, dates, and thresholds exactly as they appear in the source.
- End your answer with a "## Sources" section listing every document title, policy number, and URI from the RAG responses."""


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
        message += f"INTENT ANALYSIS (use this to guide your search — all queries relate to House Majority Staff Office):\n{context}\n\n"
    message += f"PREVIOUS ANSWER (the user was NOT satisfied with this):\n---\n{previous_answer}\n---\n\nUSER QUESTION:\n{query}"

    answer = ""
    logs: list[LogEntry] = []

    async for event in run_agent_ephemeral(agent, message, user_id="escalation-search-user", app_name="escalation_app"):
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

            if "escalation_answer" in state_delta:
                answer = str(state_delta["escalation_answer"])

    all_logs = sorted(logs + rag_logs, key=lambda l: l.timestamp)
    return SearchResponse(answer=answer or "(no answer produced)", logs=all_logs)
