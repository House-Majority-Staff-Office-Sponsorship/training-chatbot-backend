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
