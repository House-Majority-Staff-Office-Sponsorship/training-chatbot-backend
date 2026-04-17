"""
Helper to run agents using InMemoryRunner with ephemeral sessions.

The Python ADK v1.18+ requires explicit session creation before running.
This helper mirrors the TypeScript SDK's `runEphemeral()` behavior.
"""

from __future__ import annotations

from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part


def extract_usage_tokens(event) -> tuple[int, int, int]:
    """Return (prompt, response, total) token counts from an ADK event.

    Coerces None to 0 — newer Vertex SDK responses sometimes set the
    *_token_count fields to None on tool-only turns, which breaks the
    LogEntry pydantic int validator.
    """
    usage = getattr(event, "usage_metadata", None)
    if not usage:
        return 0, 0, 0
    prompt = getattr(usage, "prompt_token_count", 0) or 0
    response = getattr(usage, "candidates_token_count", 0) or 0
    total = getattr(usage, "total_token_count", 0) or (prompt + response)
    return prompt, response, total


async def run_agent_ephemeral(
    agent: BaseAgent,
    message: str,
    user_id: str = "ephemeral-user",
    app_name: str = "app",
) -> AsyncGenerator:
    """
    Creates an ephemeral session, runs the agent, and yields events.
    Equivalent to TypeScript's runner.runEphemeral().
    """
    runner = InMemoryRunner(agent=agent, app_name=app_name)

    session = await runner.session_service.create_session(
        app_name=app_name,
        user_id=user_id,
    )

    msg = Content(role="user", parts=[Part(text=message)])

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=msg,
    ):
        yield event
