"""
Pydantic models and shared utilities for the FastAPI backend.
Matches the exact request/response shapes the frontend expects.
"""

from __future__ import annotations

import time
from typing import Literal, Optional

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------

class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class LogEntry(BaseModel):
    agent: str
    message: str
    promptTokens: int
    responseTokens: int
    totalTokens: int
    timestamp: int
    researcherIndex: Optional[int] = None


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class IntentRequest(BaseModel):
    query: str
    conversationHistory: list[ConversationMessage] = []


class SearchRequest(BaseModel):
    query: str
    context: Optional[str] = None
    conversationHistory: list[ConversationMessage] = []


class EscalationRequest(BaseModel):
    query: str
    previousAnswer: str
    context: Optional[str] = None
    conversationHistory: list[ConversationMessage] = []


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class IntentResponse(BaseModel):
    action: Literal["confirm", "clarify", "reject", "chat"]
    enrichedQuery: str
    message: str
    logs: list[LogEntry]


class SearchResponse(BaseModel):
    answer: str
    logs: list[LogEntry]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_conversation_history(history: list[ConversationMessage]) -> str:
    """Format conversation history into a text block for agent context."""
    if not history:
        return ""
    lines = [
        f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}"
        for m in history
    ]
    return (
        "CONVERSATION HISTORY (use this for context — the user may refer to earlier messages):\n"
        + "\n".join(lines)
        + "\n\n"
    )


def now_ms() -> int:
    """Current time in milliseconds."""
    return int(time.time() * 1000)
