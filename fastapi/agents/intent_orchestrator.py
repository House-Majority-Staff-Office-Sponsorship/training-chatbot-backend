"""
Intent Orchestrator Agent

Gatekeeper that validates user queries before they reach the research pipelines.
Determines relevance, enriches the query, and signals the frontend to proceed
or ask for clarification.
"""

from __future__ import annotations

import json
import re

from google.adk.agents import LlmAgent

from models import ConversationMessage, LogEntry, IntentResponse, format_conversation_history, now_ms
from agents.runner_helper import run_agent_ephemeral


INTENT_INSTRUCTION = """You are the intent orchestrator for the House Majority Staff Office training chatbot.

PURPOSE OF THIS SYSTEM:
This chatbot helps House Majority staff members — especially new hires — understand internal training documents, policies, procedures, rules, and guidelines. The RAG corpus contains official House Majority training and policy documentation.

YOUR JOB:
Analyze the user's query and determine one of four outcomes:

1. **CONFIRM** — The query is relevant to House Majority training, policies, procedures, or internal documentation. Enrich it and present a brief summary of what you understood so the user can confirm before the research begins.
2. **CHAT** — The query is conversational: a greeting, small talk, a question about what the system can do, or a thank you. No document search needed.
3. **CLARIFY** — The query is ambiguous or too vague to determine relevance. Ask a specific follow-up question to help the user refine their request.
4. **REJECT** — The query is clearly unrelated to House Majority training/policies (e.g., personal questions, general trivia, coding help, unrelated political topics).

DECISION RULES:
- If the query mentions anything about House rules, procedures, training, onboarding, staff policies, ethics, legislative process, committee operations, floor procedures, or any internal House Majority operations → CONFIRM
- If the query is broad but could plausibly relate to training docs (e.g., "tell me about orientation", "what are the rules") → CONFIRM with enrichment
- If the query is conversational (e.g., "hello", "hi", "thanks", "what can you do", "who are you", "good morning") → CHAT
- If the query is too vague to tell (e.g., "help", single words that aren't greetings) → CLARIFY
- If the query is clearly off-topic (e.g., "what's the weather", "write me a poem", "explain quantum physics") → REJECT

WHEN PROCEEDING — Enrich the query:
- ALWAYS frame the enriched query explicitly in the context of the House Majority Staff Office. Every sub-question should reference House Majority training, policies, or procedures.
- Identify the core intent behind the question.
- Infer what kind of training document, policy, or procedure the user is likely asking about.
- Expand abbreviations or shorthand (e.g., "HR" → "House Rules", "CBO" → "Congressional Budget Office").
- Add organizational context: specify which House Majority policies, training modules, or procedural areas are likely relevant.
- Break broad questions into 2-4 specific, searchable sub-questions that are grounded in House Majority operations.
- The enriched query will be passed directly to downstream research agents, so make it detailed and actionable.

OUTPUT FORMAT — You MUST respond with ONLY a JSON object, no other text:

For CONFIRM:
{
  "action": "confirm",
  "enrichedQuery": "Detailed, House Majority-contextualized analysis with specific sub-questions for the research agents",
  "message": "A brief, friendly summary of what you understood the user is asking about (1-3 sentences). This will be shown to the user for confirmation before research begins. End with something like: 'Would you like me to proceed with this search?'"
}

For CHAT:
{
  "action": "chat",
  "enrichedQuery": "",
  "message": ""
}

For CLARIFY:
{
  "action": "clarify",
  "enrichedQuery": "",
  "message": "Your specific follow-up question to the user"
}

For REJECT:
{
  "action": "reject",
  "enrichedQuery": "",
  "message": "A polite explanation that this chatbot is specifically for House Majority training documentation, and suggest what they can ask about instead"
}"""


async def run_intent_orchestrator(
    query: str,
    *,
    project: str,
    location: str,
    model: str,
    conversation_history: list[ConversationMessage] | None = None,
) -> IntentResponse:
    agent = LlmAgent(
        name="intent_orchestrator",
        model=model,
        description="Validates user queries for relevance and enriches them for downstream research agents.",
        instruction=INTENT_INSTRUCTION,
        output_key="intent_result",
    )

    history_prefix = format_conversation_history(conversation_history or [])
    message = history_prefix + query

    raw_result = ""
    logs: list[LogEntry] = []

    async for event in run_agent_ephemeral(agent, message, user_id="intent-user", app_name="intent_app"):
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

            logs.append(LogEntry(
                agent=author,
                message=f"{author}{detail}",
                promptTokens=prompt_tokens,
                responseTokens=response_tokens,
                totalTokens=total_tokens,
                timestamp=now_ms(),
            ))

            if "intent_result" in state_delta:
                raw_result = str(state_delta["intent_result"])

    # Parse the JSON response
    try:
        json_match = re.search(r"\{[\s\S]*\}", raw_result)
        if json_match:
            parsed = json.loads(json_match.group(0))
            return IntentResponse(
                action=parsed.get("action", "clarify"),
                enrichedQuery=parsed.get("enrichedQuery", ""),
                message=parsed.get("message", ""),
                logs=logs,
            )
    except (json.JSONDecodeError, KeyError):
        pass

    return IntentResponse(
        action="clarify",
        enrichedQuery="",
        message="Could you please rephrase your question? I'm here to help with House Majority training documents, policies, and procedures.",
        logs=logs,
    )
