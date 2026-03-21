"""
Conversational Agent — simple LLM chat with no tools.

Handles greetings, small talk, and general questions about
what the system can do. No RAG search, no research pipeline.
"""

from __future__ import annotations

from google.adk.agents import LlmAgent

from models import ConversationMessage, LogEntry, SearchResponse, format_conversation_history, now_ms
from agents.runner_helper import run_agent_ephemeral


CONVERSATIONAL_INSTRUCTION = """You are a friendly assistant for the House Majority Staff Office training chatbot.

You handle conversational messages — greetings, small talk, and questions about what this system can do.

About this system:
- This chatbot helps House Majority staff members understand internal training documents, policies, procedures, and guidelines.
- Users can ask questions about House rules, onboarding, staff policies, ethics, legislative process, committee operations, floor procedures, and other internal House Majority operations.
- The system searches an official document corpus to provide grounded, sourced answers.
- There are three search modes: Quick Search (fast single-pass), Quick Search Pro (higher quality), and Deep Research (thorough multi-agent pipeline).

Rules:
- Be friendly, helpful, and concise.
- If someone greets you, greet them back warmly and briefly explain what you can help with.
- If someone asks what you can do, explain the system's capabilities.
- If someone asks a question that sounds like it needs document research, suggest they try one of the search modes.
- Keep responses short — 2-4 sentences for greetings, a bit more if explaining capabilities.
- Do not make up information about House Majority policies or procedures."""


async def run_conversational(
    query: str,
    *,
    project: str,
    location: str,
    model: str,
    conversation_history: list[ConversationMessage] | None = None,
) -> SearchResponse:
    agent = LlmAgent(
        name="conversational_agent",
        model=model,
        description="Handles conversational queries — greetings, small talk, and questions about the system.",
        instruction=CONVERSATIONAL_INSTRUCTION,
        output_key="chat_answer",
    )

    history_prefix = format_conversation_history(conversation_history or [])
    message = history_prefix + query

    answer = ""
    logs: list[LogEntry] = []

    async for event in run_agent_ephemeral(agent, message, user_id="chat-user", app_name="conversational_app"):
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

            if "chat_answer" in state_delta:
                answer = str(state_delta["chat_answer"])

    return SearchResponse(answer=answer or "(no response)", logs=logs)
